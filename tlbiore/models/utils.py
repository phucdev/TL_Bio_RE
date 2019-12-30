import pandas as pd
import numpy as np


MAX_LEN = 512


def read_tsv(file_path):
    return pd.read_csv(file_path, delimiter='\t', header=None, names=['id', 'sentence', 'label'])


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_precision(preds, labels):
    """
    Function to calculate the precision of our predictions vs labels: TP/(TP+FP)
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Collect comparison value for all the times that our model predicted 1
    positives = [pred == gold for pred, gold in zip(pred_flat,labels_flat) if pred == 1]
    tp = sum(positives)
    fp = len(positives) - tp
    return tp / (tp + fp)


def flat_recall(preds, labels):
    """
    Function to calculate the recall of our predictions vs labels: TP/(TP+FN)
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    tp_and_fn = labels_flat.sum()
    # Collect comparison value for all the times that our model predicted 1
    tp = sum([pred == gold for pred, gold in zip(pred_flat,labels_flat) if pred == 1])

    return tp / tp_and_fn


def f1_score(precision, recall):
    return 2.0*precision*recall/(precision+recall)
