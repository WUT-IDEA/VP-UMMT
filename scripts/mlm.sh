#!/bin/bash



tgt="$1"


SRC=zh
TGT=${tgt}

ROOT="/data1/home/turghun/project"

DATA_PATH="$ROOT/VMLM/data/mscoco/mscoco-multi30k/mono/${SRC}-${TGT}"
DUMP_PATH=$ROOT/acmmm/models/${SRC}-${TGT}
EXP_NAME="mlm-mscoco-multi30k"

EPOCH_SIZE=100000
BSZ=64

export CUDA_VISIBLE_DEVICES=1

python ../train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs "${SRC}-${TGT}" --clm_steps '' --mlm_steps "${SRC},${TGT}" \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size $BSZ --bptt 256 \
  --optimizer 'adam,lr=0.0001' \
  --tokens_per_batch 2000 --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics '_valid_mlm_ppl' --stopping_criterion '_valid_mlm_ppl,25' \
  --fp16 false --keep_best_checkpoints 11 --save_periodic 5
  
