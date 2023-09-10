#!/bin/bash
set -ex

# Runs the "345M" parameter model

# export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
HOSTFILE=./hostfile
MASTER_ADDR=10.0.2.192
MASTER_PORT=6000


CHECKPOINT_PATH=./test-checkpoints
VOCAB_FILE=./dataset/gpt2-vocab.json
MERGE_FILE=./dataset/gpt2-merges.txt
DATA_PATH=./dataset/test-data/<Specify path and file prefix>_text_document
BASE_PATH=./tmp
DS_CONFIG=./deepspeed.json

TP=2
PP=1
ZERO_STAGE=1


HIDDEN_SIZE=1024 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=1024 # e.g. llama-13b: 13824
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=16 # e.g. llama-13b: 40
SEQ_LENGTH=512
NUM_KV_HEADS=4 # llama2 70B uses GQA

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32 # e.g. llama: 4M tokens
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1


ENV_ARGS="WANDB_DISABLED=true TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=INFO"

DISTRIBUTED_ARGS="
    --hostfile $HOSTFILE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --include "10.0.2.192:0,1@10.0.2.191:6,7" \
"

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

PARALLELISM_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    $ds_args
"

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LR \
    --train-iters $TRAIN_STEPS \
    --min-lr $MIN_LR \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $GRAD_CLIP \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 \
    --bf16 \
    --num-key-value-heads $NUM_KV_HEADS \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

$ENV_ARGS deepspeed $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $PARALLELISM_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
