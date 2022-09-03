#!/bin/bash
# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 run_summarization.py \
  --data_dir="REPLACE-BY-YOUR-PATH/datasets/dataset/LongSumm2021/abstractive/processed" \
  --output_dir="REPLACE-BY-YOUR-PATH/Longsumm_code/output_test" \
  --vocab_model_file="REPLACE-BY-YOUR-PATH/Longsumm_code/bigbird/vocab/pegasus.model" \
  --attention_type=block_sparse \
  --couple_encoder_decoder=False \
  --max_encoder_length=1024 \
  --max_decoder_length=640 \
  --num_attention_heads=16 \
  --num_hidden_layers=16 \
  --hidden_size=1024 \
  --intermediate_size=4096 \
  --block_size=64 \
  --scope=pegasus \
  --norm_type=prenorm \
  --hidden_act=relu \
  --use_bias=False \
  --rescale_embedding=True \
  --substitute_newline="<n>" \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --do_train=True \
  --do_eval=True \
  --init_checkpoint="REPLACE-BY-YOUR-PATH/Longsumm_code/pretrained/summarization_arxiv_pegasus_model.ckpt-300000"
