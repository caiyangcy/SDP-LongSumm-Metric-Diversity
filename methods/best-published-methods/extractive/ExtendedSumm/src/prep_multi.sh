#!/usr/bin/env bash


############### Normal- experiments Longsumm #################

# Longsumm
DATASET=test
MODEL=Bert-Large # Sci-Bert Bert-Large
DATA_TYPE=abstractive

for idx in {0..9}
do
    BASE_DIR=REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_$idx/ExtendedSumm/
    RAW_PATH=$BASE_DIR/$DATA_TYPE/$DATASET/
    SAVE_JSON=$BASE_DIR/processed/$MODEL/$DATA_TYPE/${DATASET}_jsons
    BERT_DIR=$BASE_DIR/processed/$MODEL/$DATA_TYPE/${DATASET}_bert_files

    echo "Starting to write aggregated json files..."
    echo "-----------------"



    python3 preprocess.py -mode format_longsum_to_lines \
                        -save_path $SAVE_JSON/  \
                        -n_cpus 24 \
                        -keep_sect_num \
                        -shard_size 150 \
                        -log_file ../logs/preprocess.log \
                        -raw_path $RAW_PATH/ \
                        -dataset $DATASET \


    echo "-----------------"
    echo "Now starting to write torch files..."
    echo "-----------------"
    # scibert bert-base bert-large

    python3 preprocess.py -mode format_to_bert \
                        -bart \
                        -model_name bert-large \
                        -dataset $DATASET \
                        -raw_path $SAVE_JSON/ \
                        -save_path $BERT_DIR/ \
                        -n_cpus 24 \
                        -log_file ../logs/preprocess.log \
                        -lower \


done