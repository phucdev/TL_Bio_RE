import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split


def get_span(entity_no, spans):
    span_tuples = []
    for span in spans.split(','):
        limits = span.split('-')
        start = int(limits[0])
        end = int(limits[1]) + 1
        span_tuples.append((entity_no, start, end))
    return span_tuples


def split_sentence(span_list, sentence, include_entities=False):
    """
    Returns sentence blocks that are not to be replaced.
    """
    sentence_array: List[str] = []
    start_idx = 0
    for idx, triple in enumerate(span_list):
        sentence_array.append(sentence[start_idx:triple[1]])
        if include_entities:
            sentence_array.append(sentence[triple[1]:triple[2]])
        start_idx = triple[2]
    sentence_array.append(sentence[span_list[-1][2]:])
    return sentence_array


def train_dev_test_split(df, split_ratio=(0.8, 0.1, 0.1)):
    train_size, dev_size, test_size = split_ratio
    # TODO split at sentence level
    train, tmp = train_test_split(df, random_state=2018, train_size=train_size)
    dev, test = train_test_split(tmp, random_state=2018, test_size=test_size/(test_size+dev_size))
    return train, dev, test


def export_tsv(df, out):
    """
    Deletes span columns
    Then exports to out path
    """
    data = df.copy()[['p_id', 'sentence', 'label']]
    data.to_csv(out, sep='\t', index=False, header=False)
