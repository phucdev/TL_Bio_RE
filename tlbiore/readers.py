import csv
import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tsv_dataset_reader")
class TsvDatasetReader(DatasetReader):
    """
    Text classification data reader
    The data is assumed to be tab separated: 'pair id', 'sentence', 'label'
    'metadata' is optional and only used for passing metadata to the model
    """

    def __init__(self,
                 pair_id: int = 0,
                 sentence: int = 1,
                 label: int = 2,
                 lazy: bool = False,
                 skip_header: bool = False,
                 delimiter: str = "\t",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        self.pair_id = pair_id
        self.sentence = sentence
        self.label = label
        self.skip_header = skip_header
        self.delimiter = delimiter
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path) as input_file:
            logger.info("Reading instances from TSV dataset at: %s", file_path)

            reader = csv.reader(input_file, delimiter="\t")
            if self._skip_header:
                next(reader)
            for example in reader:
                pair_id = example[self.pair_id]
                sentence = example[self.sentence]
                label = example[self.label] if len(example) > 2 else None
                yield self.text_to_instance(sentence, label, pair_id)

    @overrides
    def text_to_instance(self,
                         sentence: str,
                         label: str = None,
                         pair_id: str = None) -> Instance:  # type: ignore
        sentence_tokens = self._tokenizer.tokenize(sentence)  # TODO: check compatibility with BioBERT
        fields = {
            'sentence': TextField(sentence_tokens, self._token_indexers),
        }
        if label is not None:
            fields['label'] = LabelField(label)

        if pair_id is not None:
            fields['pair_id'] = MetadataField(pair_id)
        return Instance(fields)
