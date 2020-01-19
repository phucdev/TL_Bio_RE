from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as f
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AdamW
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tlbiore.models import utils
from tlbiore.dataset_readers import readers


class BertSequenceClassifier(pl.LightningModule):

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(BertSequenceClassifier, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.num_labels = 2
        self.model = BertForSequenceClassification.from_pretrained(args.pretrained_model_name,
                                                                   num_labels=self.num_labels)

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer

    def forward(self, *args, **kwargs):
        """We don't really need this function"""
        pass

    def training_step(self, batch, batch_nb):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_nb):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })
        return output

    def validation_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_nb):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })

        return output

    def test_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    @pl.data_loader
    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        return train_dataloader

    @pl.data_loader
    def tng_dataloader(self):
        # supposedly deprecated, but PyCharm seems to require this implementation
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        return train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        val_sampler = SequentialSampler(self.train_dataset)
        val_dataloader = DataLoader(self.train_dataset, sampler=val_sampler, batch_size=self.batch_size)
        return val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        test_sampler = SequentialSampler(self.train_dataset)
        test_dataloader = DataLoader(self.train_dataset, sampler=test_sampler, batch_size=self.batch_size)
        return test_dataloader


class RBert(pl.LightningModule):

    def __init__(self, args):
        super(RBert, self).__init__()
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, output_attention=True)

    def configure_optimizers(self):
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, )
        return optimizer

    def forward(self, input_ids, attention_mask):
        """
        Follows huggingface BertForSequenceClassification implementation
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        """
        With our configuration outputs is a tuple of:
        - outputs[0]: Sequence of hidden-states at the output of the last layer of the model.
        - outputs[1]: Last layer hidden-state of the first token of the sequence (classification token) further
            processed by a Linear layer and a Tanh activation function.
        - outputs[2]: Attention weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        """
        _, pooled_output, attn = outputs

        # randomly deactivate few neurons in nn to avoid overfitting
        pooled_output = self.dropout(pooled_output)

        # TODO add additional logic for layers (Alibaba) here
        logits = self.classifier(pooled_output)

        return logits, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask)

        # loss
        loss = f.cross_entropy(y_hat, labels)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask)

        # loss
        loss = f.cross_entropy(y_hat, labels)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), labels.cpu())     # TODO: check
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, labels = batch

        y_hat, attn = self.forward(input_ids, attention_mask)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), labels.cpu())
        test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(y_hat.cpu(), labels.cpu(),
                                                                                      labels=[1, 0])

        return {'test_acc': torch.tensor(test_acc), 'test_precision': torch.tensor(test_precision[0]),
                'test_recall': torch.tensor(test_recall[0]), 'test_fscore': torch.tensor(test_fscore[0])}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_test_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_test_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        avg_test_fscore = torch.stack([x['test_fscore'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'avg_test_precision': avg_test_precision,
                'avg_test_recall': avg_test_recall, 'avg_test_fscore': avg_test_fscore,
                'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def tng_dataloader(self):
        # supposedly deprecated, but PyCharm seems to require this implementation
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader