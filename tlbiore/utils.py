import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers.tokenization_bert import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["#", "$"]  # TODO what about "@PROTEIN$"? Resize vocab or manually add to vocab.txt?


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param args:
    :param output_file: prediction_file_path
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (np.asarray(preds) == np.asarray(labels)).mean()


def acc_and_f1(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)

    p, r, f1 = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=average)
    # f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }
