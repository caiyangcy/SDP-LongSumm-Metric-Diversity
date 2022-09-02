#!/usr/bin/env bash

for idx in {1..9}
do
    echo "-----------------"
    echo "data idx: ${idx}"
    echo "-----------------"

    BERT_DIR=/home/caiyang/Documents/CSIRO-Data61-Summer/datasets/dataset_multiple_splits/split_$idx/ExtendedSumm/processed/Bert-Large/abstractive/

    MODEL_PATH=/home/caiyang/Documents/CSIRO-Data61-Summer/datasets/dataset/ExtendedSumm/model_bert_base/split_$idx/

    CHECKPOINT=/home/caiyang/Documents/CSIRO-Data61-Summer/replication/ExtendedSumm/model_bert_base/split_$idx/BEST_model_best.pt

    # scibert bert-base bert-large
    # exp_set

    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/test
    python3 train.py -task ext \
                    -mode test \
                    -test_batch_size 3000 \
                    -bert_data_path $BERT_DIR \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus 0 \
                    -max_pos 512 \
                    -max_length 600 \
                    -alpha 0.95 \
                    -exp_set test \
                    -pick_top \
                    -min_length 600 \
                    -finetune_bert False \
                    -result_path_test /home/caiyang/Documents/CSIRO-Data61-Summer/leaderboard_splits/split_$idx/baseline/ExtendedSumm/bert_base/abstractive/ \
                    -test_from $CHECKPOINT \
                    -model_name bert \
                    -val_pred_len 30 \
                    -section_prediction \
                    -alpha_mtl 0.50 \

done