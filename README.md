# SDP-LongSumm-Metric-Diversity

This repo contains implementation for paper Investigating Metric Diversity for Evaluating Long Document Summarization at the third SDP workshop at COLING 2022.

Long document summarization, a challenging summarization scenario, is the focus of the re-cently proposed LongSumm shared task. One
of the limitations of this shared task has been its use of a single family of metrics for evaluation (the ROUGE metrics). In contrast, other fields, like text generation, employ multiple metrics. We replicated the LongSumm evalu- ation using multiple test set samples (vs. the single test set of the official shared task) and investigated how different metrics might complement each other in this evaluation framework. We show that under this more rigorous evaluation, (1) some of the key learnings from Longsumm 2020 and 2021 still hold, but the relative ranking of systems changes, and (2) the use of additional metrics reveals additional high-quality summaries missed by ROUGE, and (3) we show that SPICE is a candidate metric for summarization evaluation for LongSumm.



## References

The followings are the official repos for the models used in the paper:

  * Bigbird-Pegasus: [LongSumm 2021: Session based automatic summarization model for scientific document](https://aclanthology.org/2021.sdp-1.12/)
  * DGCNN: [LongSumm 2021: Session based automatic summarization model for scientific document](https://aclanthology.org/2021.sdp-1.12/)
  * SummaFormer: [Summaformers @ LaySumm 20, LongSumm 20](https://github.com/sayarghoshroy/Summaformers)
    * Meanwhile, the pretrain models are from [The PyTorch Implementation Of SummaRuNNer](https://github.com/hpzhao/SummaRuNNer)
  * BertSum-Multi: [On Generating Extended Summaries of Long Documents](https://github.com/Georgetown-IR-Lab/ExtendedSumm)


**The following implementation will not provide all scripts from the repos above. One shall download the repos above then replace corresponding folders/files by our implementation before replication.**



## LongSumm Dataset

Refer to [LongSumm](https://github.com/guyfe/LongSumm) on how to obtain and prepare the dataset. In particular, the [script](https://github.com/sayarghoshroy/Summaformers/blob/main/LongSumm_Processing/src/pdf-json.py) from SummaFormer repo may help with science-parse. The data shall be left in the `datasets/` folder: `datasets/talksumm` and `datasetes/LongSumm`. Our paper focus on abstractive dataset only.

## Dataset Partitions

We repeat the experiments 10 times on different random partitions of the datasets. 

```
(Optional): python3 data_split_multi.py --doc_dir="datasets/talksumm/data/json-output" --summ_dir="datasets/LongSumm/extractive_summaries/talksumm_summaries" --is_extractive
python3 data_split_multi.py --doc_dir="datasets/LongSumm/abstractive_summaries/json-output-clean" --summ_dir="datasets/LongSumm/abstractive_summaries/by_clusters"
```

The script will partition and dataset and save them into `datasets/dataset_multiple_splits/split_{idx}/original/extractive` and `datasets/dataset_multiple_splits/split_{idx}/original/abstractive`, where `idx` ranges from 0 to 9.




## Methods

### Oracles

(Or-TopK) Oracle-Top K Sentences Matching: `python3 oracle_topk_selection.py`

(Or-TopK-SS) Oracle-Surrounding Sentences: `python3 oracle_text_span.py`

(Or-TopK-PM) Oracle-Paragraph Matching: `python3 oracle_text_span.py`

(Or-SW) Oracle-only Stopwords: `python3 oracle_only_stopwords.py`

### Baseline Text Summarizers

RandN: `python3 random_N_selection.py`

LeadN: `python3 lead_N_selection.py`


### Extractive

#### DGCNN
Under `methods/best-published-methods/abstractive/Bigbird-Pegasus/model/DGCNN_extract`

  1. Preprocess data: `python3 Vector_convert_multi.py`
  2. Training: `python3 DGCNN_train_multi.py`
  3. Inference: `python3 result_pred.py`

#### SummaRuNNer
Under `methods/best-published-methods/extractive/SummaRuNNer/LongSumm_Processing`
  1. Preprocess data: `python3 prepare_data_multi.py`
Under `/home/caiyang/Documents/SDP-codes/SDP-LongSumm-Metric-Diversity/methods/best-published-methods/extractive/SummaRuNNer/LongSumm_Training_Inference`
  2. Training & inference: `python3 main_multi.py`

#### BERTSum-Multi
Under `methods/best-published-methods/extractive/ExtendedSumm/src`

  1. Preprocess data: `bash prep_multi.sh`
  2. Training: `bash train_multi.sh`
  3. Inference: `bash test_multi.sh`


### Abstractive

#### Bigbird-Pegasus
Under `methods/best-published-methods/abstractive/Bigbird-Pegasus/model/utils`
  1. Preprocess data: `python3 session_rank_multi.py`
Under `methods/best-published-methods/abstractive/Bigbird-Pegasus/bigbird/summarization`
  2. Inference: `python3 predict_multi.py`

#### BART
Under `methods/best-published-methods/abstractive/Bart`

  Training and inference: `python3 bart_multi.py`


## Evaluation Metrics

  * [LongSumm evaluation script](https://github.com/guyfe/LongSumm/blob/master/scripts/evaluation_script.py) for ROUGE*.
  * [E2E NLG Challenge Evaluation metrics](https://github.com/tuetschek/e2e-metrics) for BLEU, NIST, CIDEr, METEOR.
  * [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption) for SPICE.
  * [BERTScore](https://github.com/Tiiiger/bert_score).


## SummVis

The detailed installation of SummVis can be found at the SummVis repo under [Installation](https://github.com/robustness-gym/summvis#installation). If, by any chance, FileNotFoundError or data loading errors pop up, the following minor correction may be able to help (works at the time of 114 commits on 5 Nov 2021).

line 363 in summvis.py: 

```
# Before
dataset = load_dataset(str(path / filename)) 
# After
dataset = load_dataset(str(path)) 
```