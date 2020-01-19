import os
import random
import logging

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from transformers.tokenization_bert import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["#", "$"]  # TODO what about "@PROTEIN$"? Resize vocab or manually add to vocab.txt?


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def get_predict_pair_ids(args):
    test_file = os.path.join(args.data_dir, args.test_file)
    df = pd.read_csv(test_file, delimiter='\t', header=None, names=['pair_id', 'sentence', 'span_e1', 'span_e2'])
    return list(df.pair_id.values)


def write_prediction(args, output_dir, preds):
    """
    For official evaluation script
    :param args:
    :param output_dir: prediction_dir_path
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    pair_ids = get_predict_pair_ids(args)

    output_file = os.path.join(output_dir, "predictions.csv")

    # TODO as a next step split AIMed and BioInfer into 2 output files
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair_id, pred in zip(pair_ids, preds):
            f.write("{}\t{}\n".format(pair_id, relation_labels[pred]))


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


def compute_metrics(preds, labels, average='macro'):
    assert len(preds) == len(labels)
    acc = (np.asarray(preds) == np.asarray(labels)).mean()

    p, r, f1 = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }
