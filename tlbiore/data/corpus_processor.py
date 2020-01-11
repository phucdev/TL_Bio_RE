from lxml import etree
from tlbiore.data.corpus import *
from tlbiore.data import utils


def process_corpora(file_list: List):
    assert len(file_list) == 2  # TODO: only works for 2 corpora
    corpora = [process_corpus(xml_file) for xml_file in file_list]
    corpora[0].documents.extend(corpora[1].documents)
    return corpora[0]


def process_corpus(xml_file, corpus_id='PPI_corpus'):
    corpus = Corpus(corpus_id)

    for _, doc in etree.iterparse(xml_file, events=("end",), tag='document'):
        document = Document(doc.attrib)
        for sent in doc:
            sentence = Sentence(sent.attrib)
            for entity in sent.findall('entity'):
                sentence.add_entity(Entity(entity.attrib))
            for pair in sent.findall('pair'):
                sentence.add_pair(Pair(pair.attrib))
            document.add_sentence(sentence)
            sent.clear()
        doc.clear()
        corpus.add_document(document)
    return corpus


def add_markers(example: pd.Series, e1_start: str, e1_end: str, e2_start: str, e2_end: str):
    example_copy = utils.get_deep_copy(example)

    # uncomment if we want no double markers for equal spans, when the same markers are used for both entities,
    # also pass as argument to merge_span_lists
    # rm_duplicates = True if e1_start == e2_start and e1_end == e2_end else False

    e1_spans = utils.SpanUtils.get_spans_with_no(1, example_copy['e1_span'])
    e2_spans = utils.SpanUtils.get_spans_with_no(2, example_copy['e2_span'])
    entity_spans = utils.SpanUtils.merge_span_lists(e1_spans, e2_spans)

    sentence_parts = utils.get_sentence_blocks(entity_spans, example_copy['sentence'])

    new_sentence = utils.get_deep_copy(sentence_parts)

    block_offset = 0
    char_offset = 0

    output_id = 'AIMed.d3.s33.p0'   # TODO

    for idx, part in enumerate(sentence_parts):
        span_no = len(''.join(sentence_parts[:idx]))    # or plus one?, span_start

        # TODO: how to deal with contained entities, cosmetic error (# $ e1 # $)
        for span_idx, span in enumerate(e1_spans):
            _, start, end = span
            if span_no == start:
                new_sentence.insert(idx + block_offset, e1_start)
                new_start = start+char_offset
                example_copy['e1_span'][span_idx] = (new_start, end)
                block_offset += 1
                char_offset += len(e1_start)
            elif span_no == end:
                new_sentence.insert(idx + block_offset, e1_end)
                char_offset += len(e1_end)
                new_start = example_copy['e1_span'][span_idx][0]
                new_end = end+char_offset
                example_copy['e1_span'][span_idx] = (new_start, new_end)
                block_offset += 1

        for span_idx, span in enumerate(e2_spans):
            _, start, end = span
            if span_no == start:
                new_sentence.insert(idx + block_offset, e2_start)
                new_start = start+char_offset
                example_copy['e2_span'][span_idx] = (new_start, end)
                block_offset += 1
                char_offset += len(e2_start)
            elif span_no == end:
                new_sentence.insert(idx + block_offset, e2_end)
                char_offset += len(e2_end)
                new_start = example_copy['e2_span'][span_idx][0]
                new_end = end+char_offset
                example_copy['e2_span'][span_idx] = (new_start, new_end)
                block_offset += 1

    example_copy['sentence'] = ''.join(new_sentence)
    return example_copy


def anonymize_entities(example: pd.Series, anon: str):
    """
    example: data frame
    anon: to anonymize entities with
    """
    example_copy = utils.get_deep_copy(example)

    e1_spans = utils.SpanUtils.get_spans_with_no(1, example_copy['e1_span'])
    e2_spans = utils.SpanUtils.get_spans_with_no(2, example_copy['e2_span'])
    merged_spans = utils.SpanUtils.merge_span_lists(e1_spans, e2_spans)
    filtered_spans = utils.SpanUtils.filter_span_list(merged_spans, rm_duplicates=True, rm_contained=True)

    sentence_parts: List[str] = utils.get_sentence_blocks(filtered_spans, example_copy['sentence'],
                                                          include_entities=False)

    idx = 1

    for filtered_span in filtered_spans:
        sentence_parts.insert(idx, anon)

        # Span correction code specifically for anonymization
        for span_idx, span in enumerate(e1_spans):
            if utils.SpanUtils.contains(filtered_span, span):
                new_start = len(''.join(sentence_parts[:idx]))
                new_end = new_start + len(anon)
                example_copy['e1_span'][span_idx] = (new_start, new_end)
        for span_idx, span in enumerate(e2_spans):
            if utils.SpanUtils.contains(filtered_span, span):
                new_start = len(''.join(sentence_parts[:idx]))
                new_end = new_start + len(anon)
                example_copy['e2_span'][span_idx] = (new_start, new_end)
        idx += 2    # +1 added anonymized entity, +1 for loop incrementation

    example_copy['sentence'] = ''.join(sentence_parts)
    return example_copy


def prepare_data_lin(df):
    """
    Add positional markers to entities
    """
    df_copy = utils.get_deep_copy(df)
    return df_copy.apply(add_markers, args=('ps ', ' pe', 'ps ', ' pe'), axis=1)


def prepare_data_ali(df):
    """
    Add different positional markers to entities
    """
    df_copy = utils.get_deep_copy(df)
    return df_copy.apply(add_markers, args=('$ ', ' $', '# ', ' #'), axis=1)


def prepare_data_lee(df, anon='@PROTEIN$'):
    """
    Anonymize entities with specified args
    """
    df_copy = utils.get_deep_copy(df)
    return df_copy.apply(anonymize_entities, anon=anon, axis=1)
