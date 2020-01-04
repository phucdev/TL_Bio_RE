import json
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


@DatasetReader.register("ppi_dataset_reader")
class PPIDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sentences with pairs of proteins from scientific publications, and creates a
    dataset suitable for biomedical relation extraction.
    Expected format for each input line: {"pair_id": "text", "sentence": "text", "label": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        pair_id: ``MetaDataField``
        sentence: ``TextField``
        label: ``LabelField``
    where the ``label`` indicates protein-protein-interaction.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):

        with open(cached_path(file_path), "r") as input_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            for line in input_file:
                line = line.strip("\n")
                if not line:
                    continue
                line_json = json.loads(line)
                pair_id = line_json['pair_id']
                sentence = line_json['sentence']
                label = str(line_json['label'])
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

        #if pair_id is not None:
        #    fields['pair_id'] = MetadataField(pair_id)
        return Instance(fields)
