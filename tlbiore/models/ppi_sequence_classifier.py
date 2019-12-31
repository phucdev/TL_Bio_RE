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
        self.bert = BertModel.from_pretrained(BIO_BERT, output_attentions=True)

        # TODO: check number of input features, add layer for combination of the 3, softmax etc.
        # fine tuner
        self.cls_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.e1_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.e2_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.concat_layer = nn.Linear(self.bert.config.hidden_size*3, 1)

        train_dataloader, val_dataloader, test_dataloader = get_dataloader(LIN_DIRECTORY)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def forward(self, input_ids, attention_mask):

        # Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    
        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        def get_span_rep(marker_id, include_markers=True):
            search_list = np.asarray(input_ids)
            indices = np.where(search_list == marker_id)
            assert len(indices) % 2 == 0    # if that fails, we need to choose different positional markers

            rep = np.zeros_like(cls_rep)
            indices_pairs = indices.reshape(-1, 2)   # make start-end pairs
            for pair in indices_pairs:
                if not include_markers:
                    rep += np.average(cont_reps[:, pair[0]+1:pair[1]])
                else:
                    rep += np.average(cont_reps[:, pair[0]:pair[1]+1])
            rep /= len(indices_pairs)   # average to account for split entity
            return rep

        # Obtaining the representation of e1

        e1_rep = get_span_rep(E1_MARKER_ID)

        # Obtaining the representation of e2
        e2_rep = get_span_rep(E2_MARKER_ID)


    
        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)
    
        return logits


def configure_optimizers(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5,)
    return optimizer


def train(model, optimizer, train_dataloader, dev_dataloader, args: utils.Arguments):    # TODO: args as named tuple/dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(args.epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(args.gpu) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.gpu) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = utils.flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
