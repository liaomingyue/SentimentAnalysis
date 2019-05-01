import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common.util import JsonDict
from allennlp.common.util import ensure_list
from allennlp.data import Instance, DatasetReader
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.predictors.predictor import Predictor
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Embedding, CrossEntropyLoss

from model.chat_dataset_reader import ChatDatasetReader


@Model.register("sentiment_classifier")
class SentimentClassifier(Model):

    def __init__(self,
                 output_size,
                 hidden_size,
                 embedding_length,
                 bidirection,
                 attention_dim,
                 attention_hop,
                 connect_size,
                 dropout,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.hidden_size = hidden_size
        self.word_embeddings = word_embeddings  # embedding layer
        self.bilstm = nn.LSTM(embedding_length, hidden_size, bidirectional=bidirection)  # Bi-LSTM layer
        if bidirection:
            self.W_s1 = nn.Linear(2 * hidden_size, attention_dim)  # attention dimension: 64
            self.fc_layer = nn.Linear(attention_hop * 2 * hidden_size, connect_size)  # connected layer size: 2000
        else:
            self.W_s1 = nn.Linear(hidden_size, attention_dim)
            self.fc_layer = nn.Linear(attention_hop * hidden_size, connect_size)
        self.W_s2 = nn.Linear(attention_dim, attention_hop)  # attiontion hop: 16
        self.dropout = nn.Dropout(dropout)
        self.label = nn.Linear(connect_size, output_size)
        self.loss_function = CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

    def _attention_net(self, lstm_output):
        """
        Self-Attention layer
        :param lstm_output:
        :return:
        """
        # SELF ATTENTION
        attn_weight_matrix = self.W_s2(
            torch.tanh(self.W_s1(lstm_output)))  # batch_size x sequence_len x r ( attention hop)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)  # batch_size x r x sequence_len
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        # ATTENTION EXAMPLE
        # input: lstm_output final_state(h_n)
        # use single direction so num_layers * num_directions == 1
        # final_state.size() = (num_layers * num_directions, batch, hidden_size)
        # lstm_output.size() = (batch_size, num_seq, hidden_size)
        # hidden.size() = (batch_size, hidden_size)
        # attn_weights.size() = (batch_size, num_seq)
        # soft_attn_weights.size() = (batch_size, num_seq)
        # new_hidden_state.size() = (batch_size, hidden_size)
        #
        # hidden = final_state.squeeze(0)
        # attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)  # main difference toward self attention
        # soft_attn_weights = F.softmax(attn_weights, 1)
        # new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # return new_hidden_state

        return attn_weight_matrix

    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embeddings = self.word_embeddings(sentence)  # batch_size x sequence_len x embedding_size
        input = embeddings.permute(1, 0, 2)  # sequence_len x batch_size x embedding_size
        _, batch_size, _ = input.size()
        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda() # (2 x num_layer) x batch_size x hidden_size
        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()   # (2 x num_layer) x batch_size x hidden_size
        output, (h_n, c_n) = self.bilstm(input,
                                         (h_0, c_0))  # sequence_len x batch_size x (num_directions x hidden_size)
        output = output.permute(1, 0, 2)  # batch_size   x sequence_len x (num_directions x hidden_size)
        attn_weight_matrix = self._attention_net(output)  # batch_size x r x sequence_len
        hidden_matrix = torch.bmm(attn_weight_matrix,
                                  output)  # M = AH : batch_size x r x (num_directions x hidden_size)
        fc_out = self.fc_layer(
            hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))  # batch_size x 2000
        fc_out = self.dropout(fc_out)
        logits = self.label(fc_out)  # batch_size x label_num

        output = {"logits": logits}
        if label is not None:
            for metric in self.metrics.values():
                metric(logits, label)  # calculate accuracy
            output["loss"] = self.loss_function(logits, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        This method will be called by :class:`allennlp.training.Trainer` in order to compute and use model metrics for early stopping and model serialization.
        :param reset: A boolean `reset` parameter is passed, as frequently a metric accumulator will have some state which should be reset between epochs.
        :return: Returns a dictionary of metrics.
        """
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


@Predictor.register('sentiment_classifier')
class SentimentClassifierPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


if __name__ == '__main__':
    train_path = os.path.join('..', 'dataset', 'train.data')
    eval_path = os.path.join('..', 'dataset', 'eval.data')
    # load data set
    reader = ChatDatasetReader()
    train_dataset = ensure_list(reader.read(train_path))
    dev_dataset = ensure_list(reader.read(eval_path))

    # get vocabulary
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'sentence': 3})

    # define embedding
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300)
    word_embeddings = BasicTextFieldEmbedder({"sentence": token_embedding})

    # init model
    model = SentimentClassifier(output_size=8,
                                hidden_size=256,
                                embedding_length=100,
                                bidirection=True,
                                attention_dim=64,
                                attention_hop=16,
                                connect_size=1024,
                                dropout=0.5,
                                word_embeddings=word_embeddings,
                                vocab=vocab)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    iterator = BucketIterator(batch_size=10, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      num_epochs=5,
                      patience=3,  # stop training before loss raise
                      cuda_device=-1
                      )

    # start train
    trainer.train()

    # save params
    torch.save(model.state_dict(), 'model.tar.gz')

    # init predictor
    model.load_state_dict(torch.load('model.tar.gz'))
    predictor = SentimentClassifierPredictor(model, dataset_reader=reader)

    # start predict
    logits = predictor.predict('1 iyuadasdasda vcc iyu asd asd asd asd')['logits']
    label_id = np.argmax(logits)
    print(model.vocab.get_token_from_index(label_id, 'labels'))
