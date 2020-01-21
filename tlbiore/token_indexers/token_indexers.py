from typing import List, Dict
import logging

from allennlp.data import TokenType
from transformers.tokenization_bert import BertTokenizer

from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer

logger = logging.getLogger(__name__)


@TokenIndexer.register("bert-pretrained-special")
class PretrainedBertSpecialIndexer(WordpieceIndexer):

    """
    A `TokenIndexer` corresponding to a pretrained BERT model.
    # Parameters
    pretrained_model : `str`
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.
        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py#L32
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If `use_starting_offsets` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase : `bool`, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase : `List[str]`, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : `bool`, optional (default=`True`)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(
        self,
        pretrained_model: str,
        use_starting_offsets: bool = False,
        do_lowercase: bool = True,
        never_lowercase: List[str] = None,
        max_pieces: int = 512,
        truncate_long_sequences: bool = True,
        additional_special_tokens: List[str] = None
    ) -> None:
        if "-cased" in pretrained_model and do_lowercase:
            logger.warning(
                "Your BERT model appears to be cased, but your indexer is lowercasing tokens."
            )
        elif "-uncased" in pretrained_model and not do_lowercase:
            logger.warning(
                "Your BERT model appears to be uncased, "
                "but your indexer is not lowercasing tokens."
            )

        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)
        if additional_special_tokens:
            bert_tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        super().__init__(
            vocab=bert_tokenizer.vocab,
            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            namespace="bert",
            use_starting_offsets=use_starting_offsets,
            max_pieces=max_pieces,
            do_lowercase=do_lowercase,
            never_lowercase=never_lowercase,
            start_tokens=["[CLS]"],
            end_tokens=["[SEP]"],
            separator_token="[SEP]",
            truncate_long_sequences=truncate_long_sequences,
        )

    def __eq__(self, other):
        if isinstance(other, PretrainedBertSpecialIndexer):
            for key in self.__dict__:
                if key == "wordpiece_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented

    def pad_token_sequence(self, tokens: Dict[str, List[TokenType]], desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, TokenType]:
        # deprecated, but PyCharm displays a warning
        pass
