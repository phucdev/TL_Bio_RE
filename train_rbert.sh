BERT=bert-base-cased
BIOBERT=/vol/fob-vol6/nebenf13/truongph/Models/biobert_v1.1._pubmed_pytorch
SCIBERT=/vol/fob-vol6/nebenf13/truongph/Models/scibert_scivocab_cased_pytorch

METHOD=ali
PRETRAINED=$BIOBERT
MODEL_NAME=biobert_ali
export PYTHONPATH=$PWD

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 $PWD/tlbiore/main.py \
  --do_train \
  --do_eval \
  --task=ppi \
  --data_dir=$PWD/data/ppi_hu/$METHOD \
  --pretrained_model_name=$PRETRAINED \
  --model_dir=$PWD/models/$MODEL_NAME \
  --model_name=$MODEL_NAME \
  --max_seq_len=286 \
  --batch_size=16 \
  --num_train_epochs=5 \
  --no_lower_case \
  --label_file=$PWD/data/ppi_hu/labels.txt
