import pandas as pd
from typing import List, Dict


def print_list(object_list):
    if not object_list:
        return 'None'
    else:
        return ', '.join(str(e) for e in object_list)


class Span:
    def __init__(self, start: int, end: int):
        self.start: int = start
        self.end: int = end


class Entity:
    def __init__(self, ent_attrib: Dict):
        self.id: str = ent_attrib['id']
        # self.orig_id: str = ent_attrib['origId'] or ent_attrib['seqId']
        self.text: str = ent_attrib['text']
        self.e_type: str = ent_attrib['type']
        self.char_offset = ent_attrib['charOffset']

    def __str__(self):
        return "Entity(%s, %s, %s, %s)" % (self.id, self.text, self.e_type, self.char_offset)

    def get_spans(self):
        spans = []
        for span in self.char_offset.split(','):
            limits = span.split('-')
            start = int(limits[0])
            end = int(limits[1]) + 1
            spans.append(Span(start, end))
        return spans


class Pair:
    def __init__(self, pair_attrib: Dict):
        self.id: str = pair_attrib['id']
        self.e1: Entity = pair_attrib['e1']
        self.e2: Entity = pair_attrib['e2']
        self.label: int = 1 if pair_attrib['interaction'] == 'True' else 0

    def __str__(self):
        return "Pair()(%s, %s, %s, %s)" % (self.id, self.e1, self.e2, self.label)


class Sentence:
    def __init__(self, sent_attrib: Dict):
        self.id = sent_attrib['id']
        # self.orig_id = sent_attrib['origId'] or sent_attrib['seqId']
        self.text = sent_attrib['text']
        self.entities: List[Entity] = []
        self.pairs: List[Pair] = []

    def __str__(self):
        return "Sentence(%s, %s, (%s), (%s))" % (self.id, self.text, print_list(self.entities[:2]),
                                                 print_list(self.pairs[:2]))

    def add_entity(self, entity: Entity):
        if entity not in self.entities:
            self.entities.append(entity)

    def get_entity(self, e_id):
        for entity in self.entities:
            if entity.id == e_id:
                return entity
        return None

    def add_pair(self, pair: Pair):
        if pair not in self.pairs:
            self.pairs.append(pair)

    def get_examples(self):
        p_id: List[str] = []
        sentence: List[str] = []
        label: List[int] = []
        e1_span: List = []
        e2_span: List = []
        for pair in self.pairs:
            p_id.append(pair.id)
            sentence.append(self.text)
            label.append(pair.label)
            e1_span.append(self.get_entity(pair.e1).get_spans())
            e2_span.append(self.get_entity(pair.e2).get_spans())
        example_df = pd.DataFrame({'p_id': p_id, 'sentence': sentence, 'label': label,
                                   'e1_span': e1_span, 'e2_span': e2_span}).astype({'label': 'int32'})
        return example_df


class Document:
    def __init__(self, doc_attrib: Dict):
        self.id = doc_attrib['id']
        # self.orig_id = doc_attrib['origId']
        self.sentences: List[Sentence] = []

    def __str__(self):
        return "Document(%s, (%s))" % (self.id, print_list(self.sentences[:2]))

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def get_sentences(self):
        return self.sentences

    def get_examples(self):
        sentence_examples = [sentence.get_examples() for sentence in self.sentences]
        return pd.concat(sentence_examples).reset_index(drop=True)


class Corpus:
    def __init__(self, c_id):
        self.id = c_id
        self.documents = []

    def add_document(self, document: Document):
        self.documents.append(document)

    def get_sentences(self):
        sentences = [document.get_sentences() for document in self.documents]
        return sentences

    def get_examples(self):
        document_examples = [document.get_examples() for document in self.documents]
        return pd.concat(document_examples).reset_index(drop=True)
