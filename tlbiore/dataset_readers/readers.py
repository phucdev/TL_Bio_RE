import json
import os
import logging
from typing import Dict

import torch
from torch.utils.data import TensorDataset

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
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
        tokens: ``TextField``
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
                e1_span = line_json['e1_span']
                e2_span = line_json['e2_span']
                label = str(line_json['label'])
                yield self.text_to_instance(sentence, label, pair_id, e1_span, e2_span)

    @overrides
    def text_to_instance(self,
                         sentence: str,
                         label: str = None,
                         pair_id: str = None,
                         e1_span=None,
                         e2_span=None) -> Instance:  # type: ignore
        sentence_tokens = self._tokenizer.tokenize(sentence)  # TODO: check compatibility with BioBERT
        fields = {
            'text': TextField(sentence_tokens, self._token_indexers),
        }
        if label is not None:
            fields['label'] = LabelField(label)

        """if pair_id is not None:
            fields['pair_id'] = MetadataField(pair_id)
        if e1_span is not None:
            fields['e1_span'] = MetadataField(e1_span)
        if e2_span is not None:
            fields['e2_span'] = MetadataField(e2_span)"""
        return Instance(fields)


class PPIProcessor(object):
    """Processor for the PPI data set"""

    def __init__(self, args):
        self.args = args
        self.relation_labels = ["0", "1"]

    @classmethod
    def _read(cls, file_path: str):
        with open(cached_path(file_path), "r") as input_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            examples = []
            for (i, line) in enumerate(input_file):
                line = line.strip("\n")
                if not line:
                    continue
                line_json = json.loads(line)
                pair_id = line_json['pair_id']
                sentence = line_json['sentence']
                # e1_span = line_json['e1_span']
                # e2_span = line_json['e2_span']
                label = str(line_json['label'])
                if i % 500 == 0:
                    logger.info(sentence)
                examples.append({"pair_id": pair_id, "sentence": sentence, "label": label})
            return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._read(os.path.join(self.args.data_dir, file_to_read))


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer.tokenize(example['sentence'])

        e1_marker_indices = []
        e2_marker_indices = []

        for idx, token in enumerate(tokens):
            if token == "$":
                e1_marker_indices.append(idx + 1)  # account for CLS token, that will be added later on
            elif token == "#":
                e2_marker_indices.append(idx + 1)  # account for CLS token, that will be added later on

        # Make sure that we have both the start and end markers
        assert len(e1_marker_indices) % 2 == 0, "Error with missing markers $ for e1: {}".format(tokens)
        assert len(e2_marker_indices) % 2 == 0, "Error with missing markers # for e2: {}".format(tokens)

        # Collect indices of all the entity parts
        e1_marker_pairs = zip(e1_marker_indices[::2], e1_marker_indices[1::2])
        e2_marker_pairs = zip(e2_marker_indices[::2], e2_marker_indices[1::2])

        # Account for [CLS] and [SEP] with "- 2".
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        # Truncate
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for start, end in e1_marker_pairs:
            for i in range(start, end + 1):
                e1_mask[i] = 1
        for start, end in e2_marker_pairs:
            for i in range(start, end + 1):
                e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("pair_id: %s" % example['pair_id'])
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        features.append(
            {"input_ids": input_ids,
             "attention_mask": attention_mask,
             "token_type_ids": token_type_ids,
             "label_id": label_id,
             "e1_mask": e1_mask,
             "e2_mask": e2_mask})

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = PPIProcessor(args)  # or make it variable

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}'.format(args.task, mode))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)

    if args.use_positional_markers:  # TODO add to main.py parse arguments
        all_e1_mask = torch.tensor([f["e1_mask"] for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f["e2_mask"] for f in features], dtype=torch.long)  # add e2 mask
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids)

    return dataset
