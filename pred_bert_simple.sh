BERT=bert-base-cased
BIOBERT=/vol/fob-vol6/nebenf13/truongph/Models/biobert_v1.1._pubmed_pytorch
SCIBERT=/vol/fob-vol6/nebenf13/truongph/Models/scibert_scivocab_cased_pytorch

METHOD=lee
PRETRAINED=$BIOBERT
MODEL_NAME=biobert_lee
OUTPUT_DIR=lee/biobert
export PYTHONPATH=$PWD

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 $PWD/tlbiore/main.py \
  --do_pred \
  --task=ppi \
  --data_dir=$PWD/data/ppi_hu/$METHOD \
  --output_dir=$PWD/output/$OUTPUT_DIR \
  --pretrained_model_name=$PRETRAINED \
  --model_dir=$PWD/models/$MODEL_NAME \
  --model_name=$MODEL_NAME \
  --model=simple \
  --max_seq_len=286 \
  --batch_size=16 \
  --num_train_epochs=5 \
  --no_lower_case \
  --label_file=$PWD/data/ppi_hu/labels.txt
