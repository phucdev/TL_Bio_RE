from typing import Dict, Optional, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from transformers.modeling_bert import BertModel


@Model.register("bert_for_ppi")
class BertForPPI(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.
    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.
    :param
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        verbose_metrics: bool = False,
        dropout: float = 0.0,
        num_labels: int = None,
        index: str = "bert",
        label_namespace: str = "labels",
        trainable: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size

        self._label_namespace = label_namespace

        if num_labels:
            out_features = num_labels
        else:
            out_features = vocab.get_vocab_size(namespace=self._label_namespace)

        self.num_classes = out_features
        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = torch.nn.Linear(in_features, out_features)
        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self._loss = torch.nn.CrossEntropyLoss()
        self._index = index
        initializer(self._classification_layer)

    def forward(  # type: ignore
        self, sentence: Dict[str, torch.LongTensor], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        sentence : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        input_ids = sentence[self._index]
        token_type_ids = sentence[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        _, pooled = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask
        )

        pooled = self._dropout(pooled)

        # apply classification layer
        logits = self._classification_layer(pooled)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(probs, label)
            self.label_accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            if self.verbose_metrics:
                metric_dict[name + '_P'] = metric_val[0]
                metric_dict[name + '_R'] = metric_val[1]
                metric_dict[name + '_F1'] = metric_val[2]
            sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names)
        average_f1 = sum_f1 / total_len
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)
        return metric_dict
