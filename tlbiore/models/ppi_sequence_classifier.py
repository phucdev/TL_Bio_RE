import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as f
import pytorch_lightning as pl
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from transformers import AdamW
import pandas as pd
from sklearn.metrics import accuracy_score
from tlbiore.models import utils
import os


CWD = os.getcwd()
MAX_LEN = 512   # Number used in original paper, however we have some sequences that have length > 800
EPOCHS = 4  # Number of training epochs (authors recommend between 2 and 4)
BIO_BERT = CWD+'/biobert_v1.1._pubmed'
# SCI_BERT = '/content/drive/My Drive/TransferLearning/biobert_v1.1._pubmed'
LIN_DIRECTORY = CWD+'/data/lin'
LEE_DIRECTORY = CWD+'/data/lee'
ALI_DIRECTORY = CWD+'/data/ali'
E1_MARKER_ID = 1002     # "$" is 1002, "#" is 1001
E2_MARKER_ID = 1001


def preprocess(tokenizer: BertTokenizer, x: pd.DataFrame):
    sentences = x.sentence.values

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
    labels = torch.tensor(x.label.values)
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


class BertBiomedicalRE(pl.LightningModule):

    def __init__(self):
        super(BertBiomedicalRE, self).__init__()
        self.num_labels = 2
        # TODO: see
        #  https://github.com/huggingface/transformers/blob/594ca6deadb6bb79451c3093641e3c9e5dcfa446/src/transformers/modeling_bert.py#L1099
        #  https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa?
        #  to add layers for ALI method

        # use pretrained BERT
        self.bert = BertModel.from_pretrained(BIO_BERT, output_attentions=True)

        # fine tuner (2 classes)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # TODO: add additional layers here
        self.classifier = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.num_labels)

        train_dataloader, val_dataloader, test_dataloader = get_dataloader(LIN_DIRECTORY)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

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

    def training_steps(self, batch, batch_nb):
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
        val_acc = accuracy_score(y_hat.cpu(), labels.cpu())
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

        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader
