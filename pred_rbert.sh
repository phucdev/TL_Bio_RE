BERT=bert-base-cased
BIOBERT=/vol/fob-vol6/nebenf13/truongph/Models/biobert_v1.1._pubmed_pytorch
SCIBERT=/vol/fob-vol6/nebenf13/truongph/Models/scibert_scivocab_cased_pytorch

DATA_DIR=ali
MODEL_NAME=bert_ali
OUTPUT_DIR=ali/biobert

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 /vol/fob-vol6/nebenf13/truongph/TL_Bio_RE/tlbiore/main.py \
  --do_pred \
  --task=ppi \
  --data_dir=/vol/fob-vol6/nebenf13/truongph/TL_Bio_RE/data/ppi_hu/$DATA_DIR \
  --output_dir=/vol/fob-vol6/nebenf13/truongph/TL_Bio_RE/output/$OUTPUT_DIR \
  --pretrained_model_name=$BERT \
  --model_dir=/vol/fob-vol6/nebenf13/truongph/TL_Bio_RE/models/$MODEL_NAME \
  --max_seq_len=286 \
  --batch_size=16 \
  --num_train_epochs=4 \
  --use_positional_markers=True \
  --no_lower_case \
  --label_file=/vol/fob-vol6/nebenf13/truongph/Data/ppi_TL_HU/labels.txt
