#!/bin/bash

set -ex

##################################
DATA_PATH=""
VOCAB_FILEPATH=""
NUM_PROCS=100

python3 tools/preprocess_data.py \
       --input $DATA_PATH \
       --json-keys text \
       --split-sentences \
       --keep-newlines \
       --tokenizer-type 
       --tokenizer-model 
       --vocab-file $VOCAB_FILEPATH \
       --vocab-size
       --merge-file
       --append-eod \
       --lang vi \
       --output-prefix llama2 \
       --workers $NUM_PROCS \
       --partitions 1 \
       --log-interval 1000 \
       

