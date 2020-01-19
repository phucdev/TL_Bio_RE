import csv
import json
import copy
import os
import logging
from typing import List, Dict

import torch
from torch.utils.data import TensorDataset

try:
    from tlbiore.utils import get_label
except ImportError:
    from utils import get_label

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        pair_id: Unique id for the example.
        sentence: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, pair_id, sentence, label):
        self.pair_id = pair_id
        self.sentence = sentence
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id,
                 e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PPIDatasetReader(object):
    """Processor for the PPI data set"""

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    def _read(self, file_path: str, set_type: str, quotechar: str = None):
        with open(file_path, "r", encoding="utf-8") as input_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(input_file, delimiter="\t", quotechar=quotechar)
            examples = []
            for (i, line) in enumerate(reader):
                pair_id = line[0]
                sentence = line[1]
                if not self.args.no_lower_case:
                    sentence = sentence.lower()
                label = 0 if set_type == "pred" else self.relation_labels.index(line[2])
                if i % 1000 == 0:
                    logger.info(line)
                if i % 500 == 0:
                    logger.info(sentence)
                examples.append(InputExample(pair_id=pair_id, sentence=sentence,
                                             label=label))
            return examples

    def get_examples(self, mode: str):
        """
        Args:
            mode: train, dev, test, pred
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'pred':
            file_to_read = self.args.pred_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._read(file_path=os.path.join(self.args.data_dir, file_to_read), set_type=mode)


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 use_positional_markers=False):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer.tokenize(example.sentence)

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

        e1_mask = None
        e2_mask = None
        if use_positional_markers:
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
            logger.info("pair_id: %s" % example.pair_id)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask))

    return features


def load_and_cache_examples(args, tokenizer, mode: str):
    processor = PPIDatasetReader(args)  # or make it variable

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
        elif mode == "pred":
            examples = processor.get_examples("pred")
        else:
            raise Exception("For mode, Only train, dev, test, pred is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                add_sep_token=args.add_sep_token,
                                                use_positional_markers=args.use_positional_markers)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    return dataset
