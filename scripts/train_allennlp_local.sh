#!/usr/bin/env bash

# Run allennlp training locally

#
# edit these variables before running script
DATASET='sentencewise/lin'

#export BERT_VOCAB=/net/nfs.corp/s2-research/scibert/scibert_scivocab_uncased/vocab.txt
#export BERT_WEIGHTS=/net/nfs.corp/s2-research/scibert/scibert_scivocab_uncased/weights.tar.gz
export BERT_MODEL=models/biobert_v1.1._pubmed/

export DATASET_SIZE=$dataset_size

CONFIG_FILE=configs/re_bert.json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export TRAIN_PATH=data/$DATASET/train.jsonl
export DEV_PATH=data/$DATASET/dev.jsonl
export TEST_PATH=data/$DATASET/test.jsonl

export CUDA_DEVICE=0

export DROPOUT_RATE=0.1
export NUM_EPOCHS=4
export LEARNING_RATE=0.001

CUDA_VISIBLE_DEVICES=1 python -m allennlp.run train $CONFIG_FILE  --include-package tlbiore -s "$@"
