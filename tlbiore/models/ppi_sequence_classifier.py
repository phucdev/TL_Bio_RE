import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
from tlbiore.models import utils


MAX_LEN = 512   # Number used in original paper, however we have some sequences that have length > 800
EPOCHS = 4  # Number of training epochs (authors recommend between 2 and 4)
BIO_BERT = '/content/drive/My Drive/TransferLearning/biobert_v1.1._pubmed'
# SCI_BERT = '/content/drive/My Drive/TransferLearning/biobert_v1.1._pubmed'
LIN_DIRECTORY = '/content/drive/My Drive/TransferLearning/Trainingsdaten/lin'
LEE_DIRECTORY = '/content/drive/My Drive/TransferLearning/Trainingsdaten/lee'
ALI_DIRECTORY = '/content/drive/My Drive/TransferLearning/Trainingsdaten/ali'
E1_MARKER_ID = 1002     # "$" is 1002, "#" is 1001
E2_MARKER_ID = 1001


def preprocess(tokenizer: BertTokenizer, x: pd.DataFrame):
    sentences = x.sentence.value

    # TODO get position of entities & positional markers in input_ids
    # TODO handle case where sequence is longer than 512, e.g. use 512 window on relevant parts?
    # Tokenize input
    input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN) for sent in sentences]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert all of our data into torch tensors, the required data type for our model
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(x.label.value)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks, labels


def get_dataloader(data_directory):
    train_data = utils.read_tsv(data_directory+"/train.tsv")
    dev_data = utils.read_tsv(data_directory + "/dev.tsv")
    test_data = utils.read_tsv(data_directory + "/test.tsv")

    batch_size = 32

    # Import BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(BIO_BERT, do_lower_case=True)
    # special_tokens_dict = {'additional_special_tokens': ['@PROTEIN1$', '@PROTEIN2$', 'ps', 'pe']}
    # num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print('Added {} tokens'.format(num_added_tokens))

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because,
    # unlike a for loop, with an iterator the entire data set does not need to be loaded into memory

    train_data = TensorDataset(*preprocess(tokenizer, train_data))
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    dev_data = TensorDataset(*preprocess(tokenizer, dev_data))
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=batch_size)

    test_data = TensorDataset(*preprocess(tokenizer, test_data))
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    return train_dataloader, dev_dataloader, test_dataloader


class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        # TODO: see
        #  https://github.com/huggingface/transformers/blob/594ca6deadb6bb79451c3093641e3c9e5dcfa446/src/transformers/modeling_bert.py#L1099
        #  https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa?
        #  to add layers for ALI method

        # use pretrained BERT
        self.bert = BertForSequenceClassification.from_pretrained(BIO_BERT, output_attentions=True)

        train_dataloader, val_dataloader, test_dataloader = get_dataloader(LIN_DIRECTORY)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, )
        return optimizer

    def training_steps(self, batch):
        # batch
        input_ids, attention_mask, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask)

        # loss
        loss = nn.functional.cross_entropy(y_hat, label)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = nn.functional.cross_entropy(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = utils.flat_accuracy(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = utils.flat_accuracy(y_hat.cpu(), label.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self.dev_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self.test_dataloader
