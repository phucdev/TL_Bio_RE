import pandas as pd
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split


START = 1
END = 2


class SpanUtils:

    @staticmethod
    def equals(span_a, span_b) -> bool:
        return span_a[START] == span_b[START] and span_a[END] == span_b[END]

    @staticmethod
    def contains(span_a, span_b) -> bool:
        """
        :return: true if span_a contains span_b, false otherwise
        """
        return span_a[START] <= span_b[START] and span_a[END] >= span_b[END]

    @staticmethod
    def intersects(span_a, span_b) -> bool:
        return span_a[START] < span_b[END] and span_a[END] > span_b[START]

    @staticmethod
    def get_span_with_no(entity_no: int, spans: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        return [(entity_no, span[0], span[1]) for span in spans]

    @staticmethod
    def merge_span_lists(e1_spans, e2_spans):
        """
        :param e1_spans:
        :param e2_spans:
        :return: list of filtered and merged spans
        """

        spans = e1_spans.copy()
        spans.extend(e2_spans)

        filtered_spans = spans.copy()
        for idx1, span1 in enumerate(spans):
            for idx2, span2 in enumerate(spans):
                if idx2 <= idx1:
                    continue
                if SpanUtils.contains(span1, span2):
                    filtered_spans.remove(span2)
                elif SpanUtils.contains(span2, span1):
                    filtered_spans.remove(span1)
        filtered_spans.sort(key=lambda span: span[START])
        return filtered_spans


def split_sentence(span_list, sentence: str, include_entities=False) -> List[str]:
    """
    :param
    span_list : List of spans for each entity.
    sentence : Sentence string that will be split.
    include_entities : Whether or not to include the entities in the output array.

    :returns
    Array of sentence blocks that are not to be replaced.
    We currently divide the sentence into blocks based on the entity spans.
    For example:
        "Cytokines measurements during IFN-alpha treatment showed a trend to
        decreasing levels of IL-4 at 4, 12, and 24 weeks."
    with entities "IFN-alpha" and "IL-4" would return for include_entities=False:
        ["Cytokines measurements during ", " treatment showed a trend to decreasing levels of ",
        " at 4, 12, and 24 weeks."]
    and for include_entities=True:
        ["Cytokines measurements during ", "IFN-alpha", " treatment showed a trend to decreasing levels of ",
        "IL-4", " at 4, 12, and 24 weeks."]
    """
    sentence_array: List[str] = []
    start_idx = 0
    for idx, triple in enumerate(span_list):
        sentence_array.append(sentence[start_idx:triple[START]])
        if include_entities:
            sentence_array.append(sentence[triple[START]:triple[END]])
        start_idx = triple[END]
    sentence_array.append(sentence[span_list[-1][END]:])
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
        train_examples = pd.concat([elem.get_examples() for elem in train]).sample(frac=1).reset_index(drop=True)
        dev_examples = pd.concat([elem.get_examples() for elem in dev]).reset_index(drop=True)
        test_examples = pd.concat([elem.get_examples() for elem in test]).reset_index(drop=True)

        return train_examples, dev_examples, test_examples


def export_tsv(df: pd.DataFrame, out, with_label=True):
    data = df.drop('label', 1) if 'label' in df.columns and not with_label else df
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_csv(out, sep='\t', index=False, header=False)


def export_jsonl(df: pd.DataFrame, out, with_label=True):
    data = df.drop('label', 1) if 'label' in df.columns and not with_label else df
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_json(out, orient='records', lines=True)
