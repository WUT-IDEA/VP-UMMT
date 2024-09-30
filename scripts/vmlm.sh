#!/bin/bash



tgt="$1"


SRC=en
TGT=${tgt}

FEAT_ROOT="/data1/home/turghun/project/images/coco2014-multi30k/features"

REGION_FEAT_PATH="$FEAT_ROOT/faster_oidv4_features"
GRID_FEAT_PATH="$FEAT_ROOT/resnet101/local"
GLOBAL_FEAT_PATH="$FEAT_ROOT/resnet50/global"


ROOT="/data1/home/turghun/project"

DATA_PATH="$ROOT/VMLM/data/mscoco/mscoco-multi30k/mono/${SRC}-${TGT}-uy"
DUMP_PATH=$ROOT/acmmm/models/${SRC}-${TGT}-uy
EXP_NAME="vmlm-mscoco-multi30k-region-grid-global-concat-select"

REGION=true
GRID=true
GLOBAL=true

EPOCH_SIZE=300000
BSZ=64

export CUDA_VISIBLE_DEVICES=3

python ../train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size $BSZ --bptt 256 \
  --optimizer 'adam,lr=0.0001' \
  --epoch_size ${EPOCH_SIZE} --max_epoch 100000 --fp16 false --keep_best_checkpoints 1 \
  --validation_metrics '_valid_vmlm_ppl' --stopping_criterion '_valid_vmlm_ppl,50' \
  --image_names $DATA_PATH  --region_feats_path $REGION_FEAT_PATH --grid_feats_path $GRID_FEAT_PATH\
  --global_feats_path $GLOBAL_FEAT_PATH --inputs_concat true --select_attn true  --num_of_regions 36 \
  --re_img_feats $REGION --gr_img_feats $GRID --gl_img_feats $GLOBAL \
  --only_vmlm true  --eval_vmlm true --region_mask_type mask --eval_from 0 --vse true