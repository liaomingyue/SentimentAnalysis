

import os
import time
import numpy as np

import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from torch.nn import Embedding

from model.chat_dataset_reader import ChatDatasetReader
from model.sentiment_classifier import SentimentClassifier, SentimentClassifierPredictor

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

wrong_path = os.path.join('outputs', 'wrong',
                          '{}_wrong.json'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))))
train_path = os.path.join('dataset', 'train.data')
eval_path = os.path.join('dataset', 'eval.data')
test_path = os.path.join('dataset', 'test.data')
# model_path = os.path.join('outputs', '{}_model_bs{}_ep{}_hs{}_emb{}_ad{}_ah{}_cs{}_dp{}.tar.gz'.format(
#     time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())),
#     batch_size, epochs,
#     hidden_size,
#     embedding_dim,
#     attention_dim,
#     attention_hop,
#     connect_size,
#     dropout))
model_path='outputs/2019-04-26-13-04-34_model_bs64_ep15_hs512_emb200_ad128_ah32_cs2048_dp0.5.tar.gz'

# load data set
reader = ChatDatasetReader()
train_dataset = ensure_list(reader.read(train_path))
eval_dataset = ensure_list(reader.read(eval_path))

# get vocabulary
vocab = Vocabulary.from_instances(train_dataset + eval_dataset,
                                  min_count={'sentence': 3})

# define embedding
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=embedding_dim)
word_embeddings = BasicTextFieldEmbedder({"sentence": token_embedding})


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

model.load_state_dict(torch.load(model_path))
predictor = SentimentClassifierPredictor(model, dataset_reader=reader)
while True:
    sentence1 = input('sentence1: ')
    logits = predictor.predict(sentence1)['logits']
    label_id = np.argmax(logits)
    label = model.vocab.get_token_from_index(label_id, 'labels')
    print(label)
