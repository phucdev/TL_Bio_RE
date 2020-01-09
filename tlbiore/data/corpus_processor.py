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
    entity_spans = utils.SpanUtils.get_span_with_no(1, example['e1_span'])
    entity_spans.extend(utils.SpanUtils.get_span_with_no(2, example['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, example['sentence'],
                                          include_entities=True)

    idx = 1
    new_e1_spans = []
    new_e2_spans = []

    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        start = len(''.join(sentence_parts[:idx]))  # ugly hack to recalculate start index
        sentence_parts.insert(idx, e1_start if entity_no == 1 else e2_start)
        idx += 2    # +1 start marker, +1 entity itself
        sentence_parts.insert(idx, e1_end if entity_no == 1 else e2_end)
        idx += 1    # +1 end marker
        end = len(''.join(sentence_parts[:idx]))  # ugly hack to recalculate end index
        idx += 1    # +1 increment for loop

        if entity_no == 1:
            new_e1_spans.append((start, end))
        else:
            new_e2_spans.append((start, end))

    example['sentence'] = ''.join(sentence_parts)
    example['e1_span'] = new_e1_spans
    example['e2_span'] = new_e2_spans

    return example


def anonymize_entities(example: pd.Series, anon: str):
    """
    example: data frame
    anon: to anonymize entities with
    """
    new_e1_spans = []
    new_e2_spans = []
    entity_spans = utils.SpanUtils.get_span_with_no(1, example['e1_span'])
    entity_spans.extend(utils.SpanUtils.get_span_with_no(2, example['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, example['sentence'])

    idx = 1
    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        start = len(''.join(sentence_parts[:idx]))  # ugly hack to recalculate start index
        sentence_parts.insert(idx, anon)
        idx += 1  # +1 anonymized entity
        end = len(''.join(sentence_parts[:idx]))  # ugly hack to recalculate end index
        idx += 1  # +1 increment for loop

        if entity_no == 1:
            new_e1_spans.append((start, end))
        else:
            new_e2_spans.append((start, end))

    example['sentence'] = ''.join(sentence_parts)
    example['e1_span'] = new_e1_spans
    example['e2_span'] = new_e2_spans

    example['sentence'] = ''.join(sentence_parts)
    return example


def prepare_data_lin(df):
    """
    Add positional markers to entities
    """
    return df.apply(add_markers, args=('ps ', ' pe', 'ps ', ' pe'), axis=1)


def prepare_data_ali(df):
    """
    Add different positional markers to entities
    """
    return df.apply(add_markers, args=('$ ', ' $', '# ', ' #'), axis=1)


def prepare_data_lee(df, anon='@PROTEIN$'):
    """
    Anonymize entities with specified args
    """
    return df.apply(anonymize_entities, anon=anon, axis=1)
