import pandas as pd
from lxml import etree

from tlbiore.data import utils


def get_char_offset(entities, e_id):
    for e in entities:
        if e.attrib['id'] == e_id:
            return e.attrib['charOffset']


def process_corpus(xml_file):
    p_id = []  # pair id
    label = []  # interaction
    sentence = []  # sentence text
    e1_span = []  # e1 charOffset
    e2_span = []  # e2 charOffset

    for _, doc in etree.iterparse(xml_file, events=("end",), tag='document'):
        # d_id.append(doc.attrib['id'])
        for sent in doc:
            entities = sent.findall('entity')
            for pair in sent.findall('pair'):
                attributes = pair.attrib
                p_id.append(attributes['id'])
                if attributes['interaction'] == 'True':
                    label.append(1)
                else:
                    label.append(0)
                sentence.append(sent.attrib['text'])
                e1_span.append(get_char_offset(entities, attributes['e1']))
                e2_span.append(get_char_offset(entities, attributes['e2']))
            sent.clear()
        doc.clear()

    d = {'p_id': p_id,
         'sentence': sentence, 'label': label,
         'e1_span': e1_span, 'e2_span': e2_span}
    df = pd.DataFrame(data=d)
    return df


def add_markers(pair, e1_start, e1_end, e2_start, e2_end):
    entity_spans = utils.get_span(1, pair['e1_span'])
    entity_spans.extend(utils.get_span(2, pair['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, pair['sentence'],
                                          include_entities=True)

    idx = 1
    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        sentence_parts.insert(idx, e1_start if triple[0] == 1 else e2_start)
        idx += 2
        sentence_parts.insert(idx, e1_end if triple[0] == 1 else e2_end)
        idx += 2  # increment for loop and for added elem
        # print(idx)
        # print(sentence_parts)

    pair['sentence'] = ''.join(sentence_parts)

    return pair


def anonymize_entities(pair, anon1, anon2):
    """
    pair: data frame
    anon1: to anonymize entity 1
    anon2: to anonymize entity 2
    """
    entity_spans = utils.get_span(1, pair['e1_span'])
    entity_spans.extend(utils.get_span(2, pair['e2_span']))

    # Idea is to generate span triples and then replace them
    entity_spans.sort(key=lambda trip: trip[1])

    sentence_parts = utils.split_sentence(entity_spans, pair['sentence'])

    idx = 1
    for triple in entity_spans:
        entity_no, _, _ = triple
        # TODO Special case in BioInfer with overlapping spans
        sentence_parts.insert(idx, anon1 if triple[0] == 1 else anon2)
        idx += 2  # increment for loop and for added elem

    pair['sentence'] = ''.join(sentence_parts)
    return pair


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
