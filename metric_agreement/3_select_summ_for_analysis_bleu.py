import numpy as np
import sys
import re
import csv
import os
import json
import pickle
from nltk.translate import bleu_score
from rouge_score import rouge_scorer

from transformers import BartTokenizer


def read_ref_and_summ(system_file, reference_file, doc_file=None):
    with open(reference_file) as f1:
        groudtruth = f1.readlines()
        groudtruth = [line for line in groudtruth if line != '\n']

    with open(system_file) as f2:
        submission = f2.readlines()

    documents = []
    summaries = []
    if doc_file:
        with open(doc_file) as f3:
            # documents = f3.readlines()
            # documents = [doc.strip() for doc in documents]

            docu_reader = csv.reader(f3, delimiter=',')
            for row in docu_reader:
                documents.append(  " ".join( eval(row[0]) ) )
                summaries.append(  row[1].strip() ) 

    return groudtruth, submission, documents, summaries

def bertsumm_clean(text):
    # all the bertsum related texts should be cleaned through this
    return text.replace("`` ", "\"").replace(" ''", "\"").replace(" ,", ",").replace(" :", ":").replace("( ", "(").replace(" )", ")").replace(" .", ".")



def impose_max_length(summary_text, max_tokens=150):
    #same tokenization as in rouge_score
    #https://github.com/google-research/google-research/blob/26a130831ee903cb97b7d04e71f227bbe24960b2/rouge/tokenize.py
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)


def evaluate_rouge2(documents, submission):
    metrics = ['rouge1', 'rouge2', 'rougeL']

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f":[], "rouge1_r":[], "rouge2_f":[], "rouge2_r":[], "rougeL_f":[], "rougeL_r":[]}
    # results = {"rouge1_f":[], "rouge1_r":[], "rouge1_p":[], "rouge2_f":[], "rouge2_r":[], "rouge2_p":[], "rougeL_f":[], "rougeL_r":[], "rougeL_p":[]}

    if len(submission) < len(documents):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)

    for idx, input_docs in enumerate(documents):
        submission_summary = submission[idx]

        submission_summary = impose_max_length(submission_summary, max_tokens=150)
        input_docs = impose_max_length(input_docs, max_tokens=len(input_docs))

        scores = scorer.score(input_docs.strip(), submission_summary.strip())
        for metric in metrics:
            results[metric+"_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)
            # results[metric + "_p"].append(scores[metric].precision)

    return results



# extractive: DGCNN, SummaRuNNer, BERTSum
# abstractive: Bart, Bigbird-Pegasus
top_agreement = dict()
bottom_agreement = dict()

for data_partition_idx in range(10):

    print("#"*60)
    print(f" ----- data_idx: {data_partition_idx} ----- ")
    print("#"*60)


    DGCNN_summ_path = f"../leaderboard_splits/split_{data_partition_idx}/baseline/DGCNN_train_on_ext/system_50.txt"
    DGCNN_ref_path = f"../leaderboard_splits/split_{data_partition_idx}/baseline/DGCNN_train_on_ext/reference_50.txt"
    DGCNN_doc_path = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/LongSumm2021/abstractive_test_for_extractive_prediction/data_test.txt"


    DGCNN_reference, DGCNN_system, DGCNN_doc, summaries_original = read_ref_and_summ(DGCNN_summ_path, DGCNN_ref_path, DGCNN_doc_path)


    DGCNN_ref_reorder = []

    for _, src_ref in enumerate(summaries_original):
        for idx, ref in enumerate(DGCNN_reference):
            if src_ref[:100].lower() in ref.lower():
                DGCNN_ref_reorder.append(idx)
                break

    assert len(set(DGCNN_ref_reorder)) == 22, "There has been documents not updated"

    DGCNN_reference = [DGCNN_reference[idx] for idx in DGCNN_ref_reorder]
    DGCNN_system = [DGCNN_system[idx] for idx in DGCNN_ref_reorder]
    DGCNN_doc = [DGCNN_doc[idx] for idx in DGCNN_ref_reorder]



    scores = [   bleu_score.sentence_bleu( [ DGCNN_reference[idx] ], hyp ) for idx, hyp in enumerate(DGCNN_system)  ]
    dgcnn_top5 = np.argsort(scores)[::-1][:5]
    dgcnn_last5 = np.argsort(scores)[:5]



    # # ########################################################################################################################################################################################################


    document_path = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/LongSumm2021/abstractive/processed/test.json"
    with open(document_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        Bigbird_Small_No_doc = [d['document'].strip() for d in dataset]

    system_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/Bigbird-Pegasus-Doc/system_trunc.txt"
    reference_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/Bigbird-Pegasus-Doc/reference.txt"

    Bigbird_Small_No_reference, Bigbird_Small_No_system, _ , _ = read_ref_and_summ(system_file, reference_file)

    doc_file = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/LongSumm2021/Bigbird-Pegasus-Actual-Input-Doc/doc.txt"
    with open(doc_file, "r") as f:
        Bigbird_Small_No_doc = f.readlines()


    Bigbird_Small_No_ref_reorder = []

    for _, src_ref in enumerate(DGCNN_reference):
        for idx, ref in enumerate(Bigbird_Small_No_reference):
            if src_ref[:100].lower() in ref.lower():
                Bigbird_Small_No_ref_reorder.append(idx)
                break

    assert len(set(Bigbird_Small_No_ref_reorder)) == 22, "There has been documents not updated"

    Bigbird_Small_No_reference = [Bigbird_Small_No_reference[idx] for idx in Bigbird_Small_No_ref_reorder]
    Bigbird_Small_No_system = [Bigbird_Small_No_system[idx] for idx in Bigbird_Small_No_ref_reorder]
    Bigbird_Small_No_doc = [Bigbird_Small_No_doc[idx] for idx in Bigbird_Small_No_ref_reorder]


    scores = [   bleu_score.sentence_bleu( [ Bigbird_Small_No_reference[idx] ], hyp ) for idx, hyp in enumerate(Bigbird_Small_No_system)  ]
    bigbird_top5 = np.argsort(scores)[::-1][:5]
    bigbird_last5 = np.argsort(scores)[:5]


    # # # # ########################################################################################################################################################################################################


    system_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/SummaFormer_train_on_ext/system.txt"
    reference_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/SummaFormer_train_on_ext/reference.txt"

    Summaformer_reference, Summaformer_system, _ , _ = read_ref_and_summ(system_file, reference_file)

    doc_file = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/SummaFormer/SummaFormer-Actual-Input-Doc/doc.txt"
    with open(doc_file, "r") as f:
        Summaformer_docs = f.readlines()


    Summaformer_ref_reorder = []

    for _, src_ref in enumerate(DGCNN_reference):
        for idx, ref in enumerate(Summaformer_reference):
            if src_ref[:100].lower() in ref.lower():
                Summaformer_ref_reorder.append(idx)
                break

    assert len(set(Summaformer_ref_reorder)) == 22, f"There has been documents not updated, actual length: {len(set(Summaformer_ref_reorder))}"

    Summaformer_reference = [Summaformer_reference[idx] for idx in Summaformer_ref_reorder]
    Summaformer_system = [Summaformer_system[idx] for idx in Summaformer_ref_reorder]


    scores = [   bleu_score.sentence_bleu( [ Summaformer_reference[idx] ], hyp ) for idx, hyp in enumerate(Summaformer_system)  ]
    summa_top5 = np.argsort(scores)[::-1][:5]
    summa_last5 = np.argsort(scores)[:5]

    # # ########################################################################################################################################################################################################

    system_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/ExtendedSumm/bert_base/abstractive/test_source.txt"
    reference_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/ExtendedSumm/bert_base/abstractive/test_target.txt"

    BERTSumm_reference, BERTSumm_system, _ , _ = read_ref_and_summ(system_file, reference_file)

    doc_file = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/ExtendedSumm/processed/Bert-Large/abstractive/test_jsons/test.0.json"
    with open(doc_file, "r") as f:
        BERTSum_doc = json.load(f)

    BERTSum_sents = [ data['src'] for data in BERTSum_doc ]
    BERTSum_doc = [  ]

    for doc_sents in BERTSum_sents:
        doc = []
        sents_cnt = 0
        for sents in doc_sents:
            tokens = sents[0]
            if len(tokens) < 5 or len(tokens) > 150:
                continue

            doc.append( " ".join(tokens) )

            if len(doc) >= 600:
                break

        doc_text = " ".join(doc)

        BERTSum_doc.append(doc_text.strip())


    BERTSumm_ref_reorder = []

    for i, src_ref in enumerate(DGCNN_reference):
        src_ref_split = src_ref.split()
        src_ref_split = [word.strip().lower() for word in src_ref_split]
        src_ref_split = " ".join(src_ref_split)
        src_ref_split = src_ref_split.replace(" ", "")

        for idx, ref in enumerate(BERTSumm_reference):

            ref = bertsumm_clean(ref) 

            ref = ref.replace(" ", "")

            if src_ref_split[:50] in ref or src_ref_split[50:100] in ref:
                BERTSumm_ref_reorder.append(idx)

                break


    assert len(set(BERTSumm_ref_reorder)) == 22, f"There has been documents not updated, actual length: {len(set(BERTSumm_ref_reorder))}, { sorted( list(BERTSumm_ref_reorder) ) }"

    BERTSumm_reference = [BERTSumm_reference[idx] for idx in BERTSumm_ref_reorder]
    BERTSumm_system = [BERTSumm_system[idx] for idx in BERTSumm_ref_reorder]



    scores = [   bleu_score.sentence_bleu( [ BERTSumm_reference[idx] ], hyp ) for idx, hyp in enumerate(BERTSumm_system)  ]
    bertsum_top5 = np.argsort(scores)[::-1][:5]
    bertsum_last5 = np.argsort(scores)[:5]

    ########################################################################################################################################################################################################

    system_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/Bart/system_bart-large.txt"
    reference_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/Bart/reference_bart-large.txt"


    Bart_reference, Bart_system, _ , _ = read_ref_and_summ(system_file, reference_file, )

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large") # facebook/bart-large facebook/bart-large-cnn

    base_path = "../Bart/"

    with open(f"{base_path}/multi_tokenized/tokenized_data_bart-large_split_{data_partition_idx}.pickle", 'rb') as handle:
        data_tokenized = pickle.load(handle)
        _, _, test_data_tokenized = data_tokenized


    tokenized_inputs = test_data_tokenized[0]
                
    Bart_docs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in tokenized_inputs['input_ids'] ] 


    Bart_ref_reorder = []

    for i, src_ref in enumerate(DGCNN_reference):
        src_ref_split = src_ref.split()

        for idx, ref in enumerate(Bart_reference):
            if src_ref[:100].lower() in ref.lower():
                Bart_ref_reorder.append(idx)
                break

    assert len(set(Bart_ref_reorder)) == 22, f"There has been documents not updated: {Bart_ref_reorder}, length: {len(set(Bart_ref_reorder))}"

    Bart_reference = [Bart_reference[idx] for idx in Bart_ref_reorder]
    Bart_system = [Bart_system[idx] for idx in Bart_ref_reorder]
    Bart_docs = [Bart_docs[idx] for idx in Bart_ref_reorder ]

    scores = [   bleu_score.sentence_bleu( [ Bart_reference[idx] ], hyp ) for idx, hyp in enumerate(Bart_system)  ]
    bart_top5 = np.argsort(scores)[::-1][:5]
    bart_last5 = np.argsort(scores)[:5]


    entry_labels = [ 'dgcnn', 'bigbird', 'summa', 'bertsum', 'bart' ]
    tops = [ dgcnn_top5, bigbird_top5, summa_top5, bertsum_top5, bart_top5 ]
    bottoms = [ dgcnn_last5, bigbird_last5, summa_last5, bertsum_last5, bart_last5 ]

    top_agreement[data_partition_idx] = tops
    bottom_agreement[data_partition_idx] = bottoms


savepath = "../leaderboard_splits/metric_agreement/stats/bleu.pickle"
with open(savepath, "wb") as f:
    pickle.dump((top_agreement, bottom_agreement), f)
