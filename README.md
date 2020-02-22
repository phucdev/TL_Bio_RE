# TL_Bio_RE
Transfer Learning for Biomedical Relation Extraction Seminar. Applying BioBERT &amp; SciBERT to Relation Extraction (protein-protein-interaction).

## Step 1: Clone the repository and prepare the data

Clone the repository, create a python virtual environment and install the requirements.

Download the train and test data (AIMed, BioInfer) into `TL_Bio_RE/data/raw`.
Use `Korpusdaten-Bearbeiten.ipynb` to process corpora, do train-dev-test split and transform data according to the papers:

- Lee, Jinhyuk, et al. "BioBERT: pre-trained biomedical language representation model for biomedical text mining." _arXiv preprint arXiv:1901.08746_ (2019).
- Lin, Chen, et al. "A BERT-based universal model for both within-and cross-sentence clinical temporal relation extraction." _Proceedings of the 2nd Clinical Natural Language Processing Workshop._ 2019.
- Wu, Shanchan, and Yifan He. "Enriching pre-trained language model with entity information for relation classification." _Proceedings of the 28th ACM International Conference on Information and Knowledge Management._ 2019.


(The text in the notebook is written in German, but the code is self-explanatory. Also keep in mind that we combined the corpora to create our train-dev-test split. We could have instead created a split for each corpora separately.)

## Step 2: Download pretrained BERT models

### Download BioBERT and convert to PyTorch model

Download BioBERT: [https://github.com/naver/biobert-pretrained](https://github.com/naver/biobert-pretrained)

We have chosen [BioBERT v1.1 (+ PubMed 1M)](https://drive.google.com/file/d/1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD/view). The archive contains TensorFlow checkpoints, the BERT configuration and the vocabulary.

It is based on the original BERT implementation with TensorFlow from Google. Since we want to work with `huggingface`, we converted the Tensorflow checkpoint to a PyTorch model.

Conversion based on: [https://github.com/huggingface/transformers/issues/457#issuecomment-518403170](https://github.com/huggingface/transformers/issues/457#issuecomment-518403170)

You need to adjust the path to the directory where you saved BioBERT.
```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers bert \
  $BERT_BASE_DIR/model.ckpt-1000000 \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```
Our approach was to save `pytorch_model.bin`, `bert_config.json` and `vocab.txt` to a new directory `biobert_v1.1._pubmed_pytorch`. We had to rename `bert_config.json` as `config.json` to be able to load it with the `transformers` package.

### Download SciBERT

Download SciBERT: [https://github.com/allenai/scibert](https://github.com/allenai/scibert)

We have chosen [SciBERT Scivocab Cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_cased.tar) (PyTorch HuggingFace) because BioBERT is based on BERT cased.

### BERT

No need to manually download the original BERT. The Hugging Face Transformers code will automatically handle that, when we load the model with `BertModel.from_pretrained('bert-base-cased', config=bert_config)`.

## Step 3: Train the model and perform tests

### Training
We have implemented a simple BertForSequenceClassification model (based on the [Hugging Face implementation](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)) for the approaches described by [Lee et al 2019](https://arxiv.org/pdf/1901.08746.pdf) and [Lin et al 2019](https://www.aclweb.org/anthology/W19-1908.pdf).

- Lee et al 2019 anonymize the entities (with `@PROTEIN$`)
- Lin et al 2019 add positional markers to the relevant entities (with `ps` and `pe`)

Use the `train_bert_simple.sh` script (while in the TL_BIO_RE directory) to train the BertForSequenceClassification model. 
You need to change the absolute paths in the script.

In addition, we implemented RBERT (based on the following repository: [https://github.com/monologg/R-BERT](https://github.com/monologg/R-BERT)) described by [Wu, S. and He, Y. 2019](https://dl.acm.org/doi/pdf/10.1145/3357384.3358119?download=true) and adapted the code to work with our corpora.

They add positional markers to the entities:
- `$` to mark the start and end of the first entity
- `#` to mark the start and end of the second entity

In addition to the CLS token (pooled output for the whole sentence) they use the averaged representations for the entities for the classification.

Use the `train_rbert.sh` script to train the RBERT model.

Our training parameters are as follows: 

| Parameter           | Value |
|---------------------|-------|
| Batch size          | 16    |
| Max sequence length\* | 286   |
| Learning rate       | 2e-5  |
| Number of epochs    | 5     |
| Dropout rate        | 0.1   |

\*Note: Corpora analysis yielded a maximum sequence length of `281` after BERT-tokenization. We chose the value `286` to allow slack.

![Unknown](https://user-images.githubusercontent.com/11077393/72663484-7fab7280-39f3-11ea-8ad3-e7b9213394a4.png)

### Evaluation and Predictions
Evaluation is done at the end of the training automatically, because of the `--do_eval` flag.
If you only want to do evaluation, you can remove the `--do_train` flag or use the `eval_rbert.sh` script. It simply loads a model and evaluates it on the test data.

The `pred_rbert.sh` script loads a model and writes its predictions into .csv files (one for each corpora: AIMed, BioInfer) according to the specifications of the seminar:
- Pair id
- Label (True/False)

## Results

Our results on our the hold-out sets:

### AIMed

| Model           | ACC   | P     | R     | F1    |
|-----------------|-------|-------|-------|-------|
| Lee-Bert        | 86.9  | 58.7  | 84.8  | 69.4  |
| Lee-SciBert     | 89.4  | 68.7  | 72.3  | 70.4  |
| Lee-BioBert     | 90.2  | 67.2  | **85.9**  | **75.4**  |
|                 |       |       |       |       |
| Lin-Bert        | 89.6  | 67.4  | 78.0  | 72.3  |
| Lin-SciBert     | 87.4  | 63.2  | 66.5  | 64.8  |
| Lin-BioBert     | 87.8  | 61.8  | 78.0  | 69.0  |
|                 |       |       |       |       |
| WuHe-Bert       | 86.8  | 60.3  | 72.3  | 65.7  |
| WuHe-SciBert    | **90.4**  | **71.1**  | 75.9  | 73.4  |
| WuHe-BioBert    | 89.7  | 69.9  | 71.7  | 70.8  |

### BioInfer

| Model           | ACC   | P     | R     | F1    |
|-----------------|-------|-------|-------|-------|
| Lee-Bert        | 85.0  | 74.1  | 65.9  | 69.7  |
| Lee-SciBert     | 87.4  | 83.1  | 64.9  | 72.9  |
| Lee-BioBert     | 86.8  | 81.3  | 64.5  | 71.9  |
|                 |       |       |       |       |
| Lin-Bert        | 84.8  | 71.9  | **68.7**  | 70.3  |
| Lin-SciBert     | 87.1  | 82.7  | 64.0  | 72.1  |
| Lin-BioBert     | 87.5  | 81.5  | 67.5  | 73.9  |
|                 |       |       |       |       |
| WuHe-Bert       | 84.8  | 75.4  | 62.6  | 68.4  |
| WuHe-SciBert    | **87.8**  | 83.1  | 67.1  | **74.2**  |
| WuHe-BioBert    | 86.5  | **83.9**  | 60.0  | 70.0  |
