import pandas as pd
from typing import List


class Span:
    def __init__(self, start: int, end: int):
        self.start: int = start
        self.end: int = end


class Entity:
    def __init__(self, e_id: str, char_offset, orig_id: str, text: str, e_type: str):
        self.id: str = e_id
        self.spans: List[Span] = char_offset
        self.orig_id: str = orig_id
        self.text: str = text
        self.e_type: str = e_type


class Pair:
    def __init__(self, p_id: str, e1: Entity, e2: Entity, interaction: str):
        self.id: str = p_id
        self.e1: Entity = e1
        self.e2: Entity = e2
        self.label: int = 1 if interaction == 'True' else 0


class Sentence:
    def __init__(self, s_id: str, orig_id: str, text: str):
        self.id = s_id
        self.orig_id = orig_id
        self.text = text
        self.entities: List[Entity] = []
        self.pairs: List[Pair] = []

    def add_entity(self, entity: Entity):
        if entity not in self.entities:
            self.entities.append(entity)

    def add_pair(self, pair: Pair):
        if pair not in self.pairs:
            self.pairs.append(pair)

    def get_pairs(self):
        return self.pairs

    def get_examples(self):
        p_id = []
        sentence = []
        label = []
        e1_span = []
        e2_span = []
        for pair in self.pairs:
            p_id.append(pair.id)
            sentence.append(self.text)  # TODO: do transformation here?
            label.append(pair.label)
            e1_span.append(pair.e1.spans)
            e2_span.append(pair.e2.spans)
        return pd.DataFrame({'p_id': p_id, 'sentence': sentence, 'label': label,
                             'e1_span': e1_span, 'e2_span': e2_span})


class Document:
    def __init__(self, d_id, orig_id):
        self.orig_id = orig_id
        self.id = d_id
        self.sentences: List[Sentence] = []

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def get_sentences(self):
        return self.sentences

    def get_examples(self):
        return pd.concat([sentence.get_examples() for sentence in self.sentences])
