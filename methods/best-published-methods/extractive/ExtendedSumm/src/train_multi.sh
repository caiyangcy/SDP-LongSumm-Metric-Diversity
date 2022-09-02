#!/usr/bin/env bash

for idx in {8..9}
do
    echo "-----------------"
    echo "data idx: ${idx}"
    echo "-----------------"
    python3 train.py -task ext \
                    -mode train \
                    -model_name bert \
                    -val_pred_len 30 \
                    -bert_data_path /home/caiyang/Documents/CSIRO-Data61-Summer/datasets/dataset_multiple_splits/split_$idx/ExtendedSumm/processed/Bert-Large/extractive/ \
                    -ext_dropout 0.1 \
                    -model_path /home/caiyang/Documents/CSIRO-Data61-Summer/replication/ExtendedSumm/model_bert_base/split_$idx/ \
                    -lr 2e-3 \
                    -visible_gpus 1 \
                    -report_every 50 \
                    -log_file /home/caiyang/Documents/CSIRO-Data61-Summer/replication/ExtendedSumm/logs/bert_datasets_dataidx_$idx.log \
                    -val_interval 1000 \
                    -save_checkpoint_steps 200000 \
                    -batch_size 1 \
                    -test_batch_size 5000 \
                    -max_length 600 \
                    -train_steps 200000 \
                    -alpha 0.95 \
                    -exp_set train \
                    -use_interval true \
                    -warmup_steps 1000 \
                    -max_pos 512 \
                    -result_path_test /home/caiyang/Documents/CSIRO-Data61-Summer/replication/ExtendedSumm/results_bert_base/train/split_$idx/ \
                    -accum_count 2 \
                    -section_prediction \
                    -alpha_mtl 0.50 \

done