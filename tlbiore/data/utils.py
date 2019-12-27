from typing import List
from tlbiore.data.corpus import Span
from sklearn.model_selection import train_test_split


class SpanUtils:
    @staticmethod
    def get_spans(char_offset):
        spans = []
        for span in char_offset.split(','):
            limits = span.split('-')
            start = int(limits[0])
            end = int(limits[1]) + 1
            spans.append(Span(start, end))
        return spans

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
    # TODO using stratify option requires separate label data frame
    train, tmp = train_test_split(df, random_state=2018, train_size=train_size)
    dev, test = train_test_split(tmp, random_state=2018, test_size=test_size / (test_size + dev_size))
    return train, dev, test


def export_tsv(df, out):
    """
    Deletes span columns
    Then exports to out path
    """
    data = df.copy()[['p_id', 'sentence', 'label']]
    data.to_csv(out, sep='\t', index=False, header=False)
