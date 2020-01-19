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
    df = pd.read_csv(test_file, delimiter='\t', header=None, names=['pair_id', 'sentence'])
    return df.pair_id


def write_prediction(args, output_dir, predictions):
    """
    For official evaluation script
    :param args:
    :param output_dir: prediction_dir_path
    :param predictions: [0,1,0,0,1,...]
    """
    relation_labels = get_label(args)
    pair_ids = get_predict_pair_ids(args)
    pred_labels = [relation_labels[pred] for pred in predictions]
    assert len(pair_ids) == len(pred_labels), "Lengths of pair ids and predicted labels do not match: {} vs. {}"\
        .format(len(pair_ids), len(pred_labels))

    df = pd.DataFrame({"pair_ids": pair_ids, "pred_labels": pred_labels})
    aimed = df[df.pair_id.str.startswith("AIMed")]
    bioinfer = df[df.pair_id.str.startswith("BioInfer")]

    aimed.to_csv(os.path.join(output_dir, "aimed_predictions.csv"), sep='\t', index=False, header=False)
    bioinfer.to_csv(os.path.join(output_dir, "bioinfer_predictions.csv"), sep='\t', index=False, header=False)


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
