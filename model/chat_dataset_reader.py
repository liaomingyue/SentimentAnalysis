import json
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides


@DatasetReader.register("chat_data")  # Register a name for reader
class ChatDatasetReader(DatasetReader):
    """
    How to Design a Personal Data Reader.
    ---------- ---------- ---------- ----------
    Override Function Needed:
        ( __init__: init tokenizer and token indexer )
        _read: read data set file and get data instances list
        text_to_instance: turn a text field to instance
    ---------- ---------- ---------- ----------
    Usage:
        1. Get reader class instance: reader = ChatDatasetReader()
        2. Get data instances: ensure_list(reader.read('dataset.json'))
        3. get data content: What's the structure of a single instance ?
            instance.fields: text and label field of an instance
            instance.fields['key']: information of 'key' field
            For Text Field:
                instance.fields['key'].sentence: 'key' field's sentence
                instance.fields['key'].sentence[i].text: get a 'key'-field-token's text content
            For Label Field:
                instance.fields['key'].label: 'key' field's label
    ---------- ---------- ---------- ----------
    Multiplex:
        Change _read and text_to_instance functions to multiplex the reader.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        # self._tokenizer = tokenizer or WordTokenizer()
        self._tokenizer = JustSpacesWordSplitter()
        self._token_indexers = token_indexers or {"sentence": SingleIdTokenIndexer()}

    @overrides
    # 实现DatasetReader的read方法
    def _read(self, file_path):
        # cached_path: Input can be a local path or an url. If input is url, it will download the file then read it.
        with open(cached_path(file_path), "r", encoding='utf-8') as data_file:
            json_data = json.loads(data_file.read())
            for line in json_data:
                sentence = line['sentence']
                label = line['label']
                yield self.text_to_instance(sentence, label)  # generator (get/return a list of text instance)

    @overrides
    def text_to_instance(self, sentence: str, label: int = None) -> Instance:  # type: ignore
        # Text/Label field can be more than one
        # tokenized_sentence = self._tokenizer.tokenize(sentence)  # step 1: text to token list
        tokenized_sentence = self._tokenizer.split_words(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)  # step 2: token to sentence field
        # tokenized_example = self._tokenizer.tokenize(example)
        # example_field = TextField(tokenized_example, self._token_indexers)
        fields = {
            'sentence': sentence_field,
            # 'example': example_field,
        }
        if label is not None:  # when it use in predictor, label is init as None
            fields['label'] = LabelField(str(label))  # LabelField need a string input
        return Instance(fields)


if __name__ == '__main__':
    # NOTE:
    #   1. ensure_list: An Iterable may be a list or a generator.
    #       This ensures we get a list without making an unnecessary copy.
    #   2. goal:
    #       Call ChatDatasetReader.read() and get a  list of instance
    #   3. structure of test.json:
    #       A list of {'sentence': string, 'label':int}
    reader = ChatDatasetReader()
    instances = ensure_list(reader.read('D:\\PycharmProjects\\SentimentAnalysis\\dataset\\raw_data\\test.json'))
    for instance in instances:
        fields = instance.fields
        print('sentence', [t.text for t in fields["sentence"].tokens])
        for t in fields["sentence"].tokens:
            print(t.text)
        print('label', fields["label"].label)
