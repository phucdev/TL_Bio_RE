import os
import pickle
import pandas as pd
import numpy as np
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
    def contains_not_equal(span_a, span_b) -> bool:
        """
        :return: true if span_a contains span_b, but is not equal to span_b, false otherwise
        """
        return (span_a[START] < span_b[START] and span_a[END] >= span_b[END]) or \
               (span_a[START] == span_b[START] and span_a[END] > span_b[END])

    @staticmethod
    def intersects(span_a, span_b) -> bool:
        return span_a[START] < span_b[END] and span_a[END] > span_b[START]

    @staticmethod
    def get_split_points(span_list):
        # Flatten span_list and filter out duplicate indices
        collected_indices = np.asarray(span_list)[:, 1:]
        split_points = list(np.unique(collected_indices.flatten()))
        assert len(split_points) > 1
        return split_points

    @staticmethod
    def get_spans_with_no(entity_no: int, spans: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        return [(entity_no, span[0], span[1]) for span in spans]

    @staticmethod
    def merge_span_lists(e1_spans, e2_spans, rm_duplicates=False, rm_contained=False):
        """
        :param e1_spans: list of spans for first entity
        :param e2_spans: list of spans for second entity
        :param rm_contained:
        :param rm_duplicates: subcase of contained
        :return: list of merged spans (filtered if specified)
        """
        spans = e1_spans.copy()
        spans.extend(e2_spans)
        spans.sort(key=lambda span: span[START])

        if rm_duplicates or rm_contained:
            filtered_spans = get_deep_copy(spans)   # maybe .copy is enough?
            for idx1, span1 in enumerate(spans):
                for idx2, span2 in enumerate(spans):
                    if idx2 <= idx1:
                        continue
                    if rm_duplicates:
                        if SpanUtils.equals(span1, span2):
                            filtered_spans.remove(span2)
                            continue
                    if rm_contained:
                        if SpanUtils.contains(span1, span2):
                            filtered_spans.remove(span2)
                        elif SpanUtils.contains(span2, span1):
                            filtered_spans.remove(span1)
            return filtered_spans
        else:
            return spans


def split_sentence(span_list, sentence, include_entities=False) -> List[str]:
    """
    :param
    span_list: List of spans for each entity.
    sentence: Sentence string that will be split.
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

    {"pair_id":"AIMed.d3.s30.p0","sentence":"Temporally following this growth arrest, the cells develop a
    senescence morphology and express @PROTEIN$e-associated beta-galactosidase (SA-beta-gal).",
    "e1_span":[[95,104]],"e2_span":[[95,104]]}
    """
    sentence_array: List[str] = []

    start_idx = 0
    for triple in span_list:
        sentence_array.append(sentence[start_idx:triple[START]])
        if include_entities:
            sentence_array.append(sentence[triple[START]:triple[END]])
        start_idx = triple[END]
    sentence_array.append(sentence[span_list[-1][END]:])
    return sentence_array


def get_sentence_blocks(span_list, sentence: str) -> List[str]:
    """
    Similarly splits sentences at (unique) span indices, but always includes entities
    :param span_list: combined span list of both entities
    :param sentence:
    :return:
    """
    # Retrieve indices where to split sentence into blocks
    split_points = SpanUtils.get_split_points(span_list)

    sentence_array: List[str] = []

    start_idx = 0
    for split_point in split_points:
        sentence_array.append(sentence[start_idx:split_point])
        start_idx = split_point
    sentence_array.append(sentence[split_points[-1]:])

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


def get_deep_copy(obj):
    return pickle.loads(pickle.dumps(obj))


def export_tsv(df: pd.DataFrame, out, with_label=True):
    data = df.drop('label', 1) if 'label' in df.columns and not with_label else df
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_csv(out, sep='\t', index=False, header=False)


def export_jsonl(df: pd.DataFrame, out, with_label=True):
    data = df.drop('label', 1) if 'label' in df.columns and not with_label else df
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data.to_json(out, orient='records', lines=True)
