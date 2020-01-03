# TL_Bio_RE
Transfer Learning for Biomedical Relation Extraction Seminar. Applying BioBERT &amp; SciBERT to Relation Extraction (protein-protein-interaction).

## Step 1: Clone the repository and prepare the data

Clone the repository, create a python virtual environment and install the requirements.

Use `Korpusdaten-Bearbeiten.ipynb` to process corpora, do train-dev-test split and transform data according to the papers:

- "A BERT-based Universal Model for Both Within- and Cross-sentence Clinical Temporal Relation Extraction" 
- "Enriching Pre-trained Language Model with Entity Information for Relation Classification"
- "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"

## Step 2: Download BioBERT and convert to PyTorch model

Download BioBERT: [https://github.com/naver/biobert-pretrained](https://github.com/naver/biobert-pretrained)

We have chosen [BioBERT v1.1 (+ PubMed 1M)](https://drive.google.com/file/d/1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD/view). The archive contains TensorFlow checkpoints, the BERT configuration and the vocabulary.

It is based on the original BERT implementation with TensorFlow from Google. Since we want to work with `huggingface`, we converted the Tensorflow checkpoint to a PyTorch model with the [`convert_tf_checkpoint_to_pytorch.py`](https://github.com/huggingface/transformers/blob/master/transformers/convert_tf_checkpoint_to_pytorch.py) function.

See: [https://github.com/huggingface/transformers/issues/457#issuecomment-518403170](https://github.com/huggingface/transformers/issues/457#issuecomment-518403170)

You may want to adjust the path to the directory where you saved BioBERT.
```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers bert \
  $BERT_BASE_DIR/model.ckpt-1000000 \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```
Our approach was to save `pytorch_model.bin`, `bert_config.json` and `vocab.txt` to a new directory `biobert`. We had to rename `bert_config.json` as `config.json` to be able to load it with the `transformers` package as `biobert`.

## Step 3: Train the model and perform tests

Use `BioBERT-RE.ipynb` to train the different models and perform tests.