# TL_Bio_RE
Transfer Learning for Biomedical Relation Extraction Seminar. Applying BioBERT &amp; SciBERT to Relation Extraction (protein-protein-interaction).

## Step 1: Clone the repository and prepare the data

Clone the repository, create a python virtual environment and install the requirements.

Download the train and test data (AIMed, BioInfer).
Use `Korpusdaten-Bearbeiten.ipynb` to process corpora, do train-dev-test split and transform data according to the papers:

- "A BERT-based Universal Model for Both Within- and Cross-sentence Clinical Temporal Relation Extraction" 
- "Enriching Pre-trained Language Model with Entity Information for Relation Classification"
- "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"

(The text in the notebook is written in German, but the code is self-explanatory.)

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

### Evaluation and Predictions
Evaluation is done at the end of the training automatically, because of the `--do_eval` flag.
If you only want to do evaluation, you can remove that flag or use the `eval_rbert.sh` script. It simply loads a model and evaluates it on the test data.

The `pred_rbert.sh` script loads a model and writes its prediction into .csv files (one for each corpora: AIMed, BioInfer) according to the specifications of the seminar:
- Pair id
- Label (True/False)

## Results

Our results on our sentencewise train, dev, test split so far:

### Lee (BertForSequenceClassification)

| Pretrained Model    | Bert Cased Base    | BioBert Cased           | SciBert Scivocab Cased | 
|-----------|--------------------|--------------------|------------------------|
| Accuracy  | 85.68 | 88.09   | 87.47      | 
| Precision | 77.51  | 81.82 | 81.13      | 
| Recall    | 76.17 | 79.13 | 77.16     | 
| F1        | 76.81 | 80.35 | 78.88     | 

### Lin (BertForSequenceClassification)

| Pretrained Model   | Bert Cased Base    | BioBert Cased           | SciBert Scivocab Cased |  
|-----------|--------------------|--------------------|------------------------|
| Accuracy  | 86.24 | 87.61 | 88.51     | 
| Precision | 78.37 | 80.79 | 83.19     | 
| Recall    | 77.31 | 78.96 | 78.46     | 
| F1        | 77.82  | 79.82 | 80.47     | 

### Alibaba (RBERT)

| Pretrained Model   | Bert Cased Base    | BioBert Cased      | SciBert Scivocab Cased |
|-----------|--------------------|--------------------|------------------------|
| Accuracy  | 87.2 | 88.58 | 88.71      |
| Precision | 80.43   | 83.06 | 82.98     |
| Recall    | 77.25 | 79.03 | 79.91     |
| F1        | 78.66 | 80.79 | 81.3     |
