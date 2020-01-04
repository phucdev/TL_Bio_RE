import pandas as pd
import os
from typing import List
from tlbiore.data.corpus import Span
from sklearn.model_selection import train_test_split


class SpanUtils:

    @staticmethod
    def equals(span_a: Span, span_b: Span):
        return span_a.start == span_b.start and span_a.end == span_b.end

    @staticmethod
    def contains(span_a: Span, span_b: Span):
        return span_a.start <= span_b.start and span_a.end >= span_b.end

    @staticmethod
    def intersects(span_a: Span, span_b: Span):
        return span_a.start < span_b.end and span_a.end > span_b.start

    @staticmethod
    def intersects_any(spans_a: List[Span], spans_b: List[Span]):
        for span_a in spans_a:
            for span_b in spans_b:
                if SpanUtils.intersects(span_a, span_b):
                    return True     # TODO or return all the intersecting span pairs?
        return False

    @staticmethod
    def get_span_with_no(entity_no: int, spans: List[Span]):
        return [(entity_no, span.start, span.end) for span in spans]


def split_sentence(span_list, sentence: str, include_entities=False):
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


def train_dev_test_split(object_list, split_ratio=(0.8, 0.1, 0.1)):
    """
    Performs train, dev, test split on a list
    :param object_list: list of documents, sentences or pairs
    :param split_ratio: train, dev, test ratio
    :return: train, dev, test
    """
    train_size, dev_size, test_size = split_ratio
    # TODO: add stratify option for split on pair level, investigate same option for document/sentence level?
    train, tmp = train_test_split(object_list, random_state=2018, train_size=train_size)
    dev, test = train_test_split(tmp, random_state=2018, test_size=test_size / (test_size + dev_size))

    if isinstance(train, pd.DataFrame) and isinstance(dev, pd.DataFrame) and isinstance(test, pd.DataFrame):
        return train, dev, test
    else:
        # TODO: add checks that train, dev, test are lists of documents or sentences
        train_examples = pd.concat([elem.get_examples() for elem in train]).reset_index(drop=True)
        dev_examples = pd.concat([elem.get_examples() for elem in dev]).reset_index(drop=True)
        test_examples = pd.concat([elem.get_examples() for elem in test]).reset_index(drop=True)

        return train_examples, dev_examples, test_examples


def export_tsv(df, out):
    """
    Deletes span columns
    Then exports to out path
    """
    data = df.copy()[['pair_id', 'sentence', 'label']]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_csv(out, sep='\t', index=False, header=False)


def export_jsonl(df, out):
    """
    Deletes span columns
    Then exports to out path
    """
    data = df.copy()[['pair_id', 'sentence', 'label']]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_json(out, orient='records', lines=True)
