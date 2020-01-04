from overrides import overrides

import csv

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('bert-classifier')
class BertClassifierPredictor(Predictor):
    """"Predictor wrapper for the BertClassifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    @overrides
    def load_line(self, line: str) -> JsonDict:
        split_lines = line.split("\t")
        return {
            "pair_id": split_lines[0],
            "sentence": split_lines[1],
        }

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        pair_id = json_dict['pair_id']
        sentence = json_dict['sentence']
        return self._dataset_reader.text_to_instance(pair_id=pair_id, sentence=sentence)
