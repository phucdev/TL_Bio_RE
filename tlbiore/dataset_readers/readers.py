import json
import os
import logging
from typing import List, Dict

import torch
from torch.utils.data import TensorDataset

from tlbiore.utils import get_label

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PPIDatasetReader(object):
    """Processor for the PPI data set"""

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read(cls, file_path: str):
        with open(file_path, "r") as input_file:
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

    def get_examples(self, mode: str):
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


def convert_examples_to_features(examples: List[Dict], max_seq_len, tokenizer,
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

        tokens = tokenizer.tokenize(example['sentence'])

        e1_marker_pairs = None
        e2_marker_pairs = None
        if use_positional_markers:
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
            logger.info("pair_id: %s" % example['pair_id'])
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            if use_positional_markers:
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        example_features = {"input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "label_id": label_id}
        if use_positional_markers:
            example_features["e1_mask"] = e1_mask
            example_features["e2_mask"] = e2_mask

        features.append(example_features)

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
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                add_sep_token=args.add_sep_token,
                                                use_positional_markers=args.use_positional_markers)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)

    if args.use_positional_markers:
        all_e1_mask = torch.tensor([f["e1_mask"] for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f["e2_mask"] for f in features], dtype=torch.long)  # add e2 mask
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids)

    return dataset
