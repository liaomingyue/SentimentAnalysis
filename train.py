import json
import os
import time

import numpy as np
import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from torch import optim
from torch.nn import Embedding

from model.chat_dataset_reader import ChatDatasetReader
from model.sentiment_classifier import SentimentClassifier, SentimentClassifierPredictor

# ------------------- params ----------------------

# output_size = 11
# hidden_size = 256
# embedding_dim = 100
# bidirection = True
# attention_dim = 64
# attention_hop = 16
# connect_size = 1024
# dropout = 0.5
# batch_size = 32
# epochs = 15
# patience = 1

output_size = 11
hidden_size = 512
embedding_dim = 200
bidirection = True
attention_dim = 128
attention_hop = 32
connect_size = 2048
dropout = 0.5
batch_size = 64
epochs = 15
patience = 1

# 错误信息输出路径
wrong_path = os.path.join('outputs', 'wrong',
                          '{}_wrong.json'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))))
# 训练集，测试集和验证机路径
train_path = os.path.join('dataset', 'train.data')
eval_path = os.path.join('dataset', 'eval.data')
test_path = os.path.join('dataset', 'test.data')
# 模型的保存路径
model_path = os.path.join('outputs', '{}_model_bs{}_ep{}_hs{}_emb{}_ad{}_ah{}_cs{}_dp{}.tar.gz'.format(
    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())),
    batch_size, epochs,
    hidden_size,
    embedding_dim,
    attention_dim,
    attention_hop,
    connect_size,
    dropout))

# ------------------- load data ----------------------

# load data set
reader = ChatDatasetReader()
train_dataset = ensure_list(reader.read(train_path))
eval_dataset = ensure_list(reader.read(eval_path))

# get vocabulary获取train_dataset和eval_dataset的词汇量
vocab = Vocabulary.from_instances(train_dataset + eval_dataset,
                                  min_count={'sentence': 3})

# define embedding
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=embedding_dim)
word_embeddings = BasicTextFieldEmbedder({"sentence": token_embedding})

# ------------------- model ----------------------
# init model
model = SentimentClassifier(output_size=output_size,
                            hidden_size=hidden_size,
                            embedding_length=embedding_dim,
                            bidirection=bidirection,
                            attention_dim=attention_dim,
                            attention_hop=attention_hop,
                            connect_size=connect_size,
                            dropout=dropout,
                            word_embeddings=word_embeddings,
                            vocab=vocab).cuda()

# ------------------- train ----------------------
# init trainer
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=eval_dataset,
                  num_epochs=epochs,
                  patience=patience,  # stop training before loss raise
                  cuda_device=0
                  )

# start train
trainer.train()

# save params
torch.save(model.state_dict(), model_path)

# init predictor
model.load_state_dict(torch.load(model_path))

# ------------------- test ----------------------

wrong_content = []
predictor = SentimentClassifierPredictor(model, dataset_reader=reader)
with open(test_path, 'r', encoding='utf-8') as f:
    test_dataset = json.loads(f.read())

    acc_cnt = 0

    for test_item in test_dataset:
        logits = predictor.predict(test_item['sentence'])['logits']
        label_id = np.argmax(logits)
        label = model.vocab.get_token_from_index(label_id, 'labels')

        if label == test_item['label']:
            acc_cnt += 1
        else:
            wrong_content.append({'sentence': test_item['sentence'], 'label': test_item['label'], 'predict': label})

    test_acc = float(acc_cnt) / len(test_dataset)
    print('test accuracy:', test_acc)

with open(wrong_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(wrong_content, ensure_ascii=False))
