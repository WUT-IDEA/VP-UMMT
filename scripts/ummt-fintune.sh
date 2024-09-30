#!/bin/bash

# mlm or vmlm or alter_vmlm


tgt="$1"
PRE="$2"


SRC=zh
TGT=${tgt}

FEAT_ROOT="/data1/home/turghun/project/images/coco2014-multi30k/features"

REGION_FEAT_PATH="$FEAT_ROOT/faster_oidv4_features"
GRID_FEAT_PATH="$FEAT_ROOT/resnet101/local"
GLOBAL_FEAT_PATH="$FEAT_ROOT/resnet50/global"

ROOT="/data1/home/turghun/project"

DATA_PATH="$ROOT/VMLM/data/mscoco/multi30k/mono/${SRC}-${TGT}"
DUMP_PATH=$ROOT/acmmm/models/${SRC}-${TGT}

EXP_NAME="ummt-finetune-region-grid-${PRE}-select-concat"

MLM_PATH="${ROOT}/acmmm/pretrained/${SRC}-${TGT}/vmlm-mscoco-multi30k-region-concat-selective/best-valid_${PRE}_ppl.pth"

echo "-----------------model is initiaolized by ${MLM_PATH}---------------------"

REGION=false
GRID=true
GLOBAL=true

EPOCH_SIZE=14500
BSZ=64

export CUDA_VISIBLE_DEVICES=0

python ../train.py  --beam_size 8 --exp_name $EXP_NAME  --dump_path ${DUMP_PATH} \
    --data_path "${DATA_PATH}"  --reload_model $MLM_PATH,$MLM_PATH \
    --lgs "${SRC}-${TGT}"    --vae_steps "${SRC},${TGT}" --vbt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}" --word_shuffle 3  \
    --word_dropout 0.1  --word_blank 0.1  --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false \
    --batch_size $BSZ  --bptt 256 --emb_dim 512  --n_layers 6  --n_heads 8  \
    --dropout 0.2 --attention_dropout 0.1  --gelu_activation true \
    --inputs_concat true --select_attn true \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.998,lr=0.0001 --keep_best_checkpoints 1 \
    --epoch_size ${EPOCH_SIZE}  --eval_bleu true  --stopping_criterion "valid_${SRC}-${TGT}_mmt_bleu,10" \
    --validation_metrics "valid_${SRC}-${TGT}_mmt_bleu,valid_${TGT}-${SRC}_mmt_bleu" \
    --re_img_feats $REGION --gr_img_feats $GRID --gl_img_feats $GLOBAL --global_feats_path $GLOBAL_FEAT_PATH \
    --image_names $DATA_PATH  --region_feats_path $REGION_FEAT_PATH --grid_feats_path $GRID_FEAT_PATH --num_of_regions 36 \
    --eval_from 28 --vse true  



