from lxml import etree
from tlbiore.data.corpus import *
from tlbiore.data import utils


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


def process_corpora(file_list: List):
    assert len(file_list) == 2  # TODO: only works for 2 corpora
    corpora = [process_corpus(xml_file) for xml_file in file_list]
    corpora[0].documents.extend(corpora[1].documents)
    return corpora[0]


def add_markers(example, e1_start, e1_end, e2_start, e2_end):
    entity_spans = utils.SpanUtils.get_span_with_no(1, example['e1_span'])
    entity_spans.extend(utils.SpanUtils.get_span_with_no(2, example['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, example['sentence'],
                                          include_entities=True)

    idx = 1
    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        sentence_parts.insert(idx, e1_start if triple[0] == 1 else e2_start)
        idx += 2
        sentence_parts.insert(idx, e1_end if triple[0] == 1 else e2_end)
        idx += 2  # increment for loop and for added elem

    example['sentence'] = ''.join(sentence_parts)

    return example


def anonymize_entities(example, anon1, anon2):
    """
    example: data frame
    anon1: to anonymize entity 1 with
    anon2: to anonymize entity 2 with
    """
    entity_spans = utils.SpanUtils.get_span_with_no(1, example['e1_span'])
    entity_spans.extend(utils.SpanUtils.get_span_with_no(2, example['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, example['sentence'])

    idx = 1
    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        sentence_parts.insert(idx, anon1 if triple[0] == 1 else anon2)
        idx += 2  # increment for loop and for added elem

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


def prepare_data_lee(df, anon1='@PROTEIN1$', anon2='@PROTEIN2$'):
    """
    Anonymize entities with specified args
    """
    return df.apply(anonymize_entities, anon1=anon1, anon2=anon2, axis=1)
