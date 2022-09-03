import numpy as np
import sys
import re
import csv
import os
import json
import pickle
from bert_score import BERTScorer
from transformers import BartTokenizer
from rouge_score import rouge_scorer


def read_ref_and_summ(system_file, reference_file, doc_file):
    with open(reference_file) as f1:
        groudtruth = f1.readlines()
        groudtruth = [line for line in groudtruth if line != '\n']

    with open(system_file) as f2:
        submission = f2.readlines()
        submission = [line for line in submission if line != '\n']

    with open(doc_file) as f2:
        documents = f2.readlines()
        documents = [line for line in documents if line != '\n']


    return groudtruth, submission, documents



def bertsumm_clean(text):
    # all the bertsum related texts should be cleaned through this
    return text.replace("`` ", "\"").replace(" ''", "\"").replace(" ,", ",").replace(" :", ":").replace("( ", "(").replace(" )", ")").replace(" .", ".")

def save_top_and_last(documents, reference, system, top5, last5, scores, savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    with open(f"{savepath}/documents_top5.txt", "w") as f:
        doc_to_save = [ documents[i].strip().encode('unicode_escape').decode()  for i in top5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/documents_last5.txt", "w") as f:
        doc_to_save = [ documents[i].strip().encode('unicode_escape').decode() for i in last5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/reference_top5.txt", "w") as f:
        doc_to_save = [ reference[i].strip().encode('unicode_escape').decode()  for i in top5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/reference_last5.txt", "w") as f:
        doc_to_save = [ reference[i].strip().encode('unicode_escape').decode()  for i in last5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/system_top5.txt", "w") as f:
        doc_to_save = [ system[i].strip().encode('unicode_escape').decode()  for i in top5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/system_last5.txt", "w") as f:
        doc_to_save = [ system[i].strip().encode('unicode_escape').decode()  for i in last5]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/scores.txt", "w") as f:
        top5_scores = list(  map( str, [ scores[i] for i in top5] ) )
        top5_scores = "Top5: " + ", ".join(top5_scores)
        f.write(top5_scores+"\n")

        last5_scores = list(  map( str, [ scores[i] for i in last5] ) )
        last5_scores = "Last5: " + ", ".join(last5_scores)

        f.write(last5_scores)


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

        submission_summary = impose_max_length(submission_summary, max_tokens=600)
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



    system_file = f"../leaderboard_splits/metric_agreement/spice_prep/DGCNN/system_{data_partition_idx}.txt"
    reference_file = f"../leaderboard_splits/metric_agreement/spice_prep/DGCNN/reference_{data_partition_idx}.txt"
    doc_path = f"../leaderboard_splits/metric_agreement/spice_prep/DGCNN/documents_{data_partition_idx}.txt"


    DGCNN_reference, DGCNN_system, DGCNN_doc = read_ref_and_summ(system_file, reference_file, doc_path)


    score_path = f"../leaderboard_splits/metric_agreement/spice_prep/Bart/Bart_spice_{data_partition_idx}.txt"
    with open(score_path, "r") as f:
        scores = f.readlines()
        scores = scores[0]
        scores = scores.split(",")
        scores = [  float(s.strip()) for s in scores if s.strip() != "" ]

    dgcnn_top5 = np.argsort(scores)[::-1][:5]
    dgcnn_last5 = np.argsort(scores)[:5]

 
    # ########################################################################################################################################################################################################


    system_file = f"../leaderboard_splits/metric_agreement/spice_prep/Bigbird-Pegasus/system_{data_partition_idx}.txt"
    reference_file = f"../leaderboard_splits/metric_agreement/spice_prep/Bigbird-Pegasus/reference_{data_partition_idx}.txt"
    doc_path = f"../leaderboard_splits/metric_agreement/spice_prep/Bigbird-Pegasus/documents_{data_partition_idx}.txt"

    Bigbird_Small_No_reference, Bigbird_Small_No_system, Bigbird_Small_No_doc = read_ref_and_summ(system_file, reference_file, doc_path)


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



    score_path = f"../leaderboard_splits/metric_agreement/spice_prep/Bigbird-Pegasus/Bigbird-Pegasus_spice_{data_partition_idx}.txt"
    with open(score_path, "r") as f:
        scores = f.readlines()
        scores = scores[0]
        scores = scores.split(",")
        scores = [  float(s.strip()) for s in scores if s.strip() != "" ]

    scores = [scores[idx] for idx in Bigbird_Small_No_ref_reorder]


    bigbird_top5 = np.argsort(scores)[::-1][:5]
    bigbird_last5 = np.argsort(scores)[:5]


    # # # # ########################################################################################################################################################################################################


    system_file = f"../leaderboard_splits/metric_agreement/spice_prep/SummaRuNNer/system_{data_partition_idx}.txt"
    reference_file = f"../leaderboard_splits/metric_agreement/spice_prep/SummaRuNNer/reference_{data_partition_idx}.txt"
    doc_path = f"../leaderboard_splits/metric_agreement/spice_prep/SummaRuNNer/documents_{data_partition_idx}.txt"

    Summaformer_reference, Summaformer_system, Summaformer_docs = read_ref_and_summ(system_file, reference_file, doc_path)

    Summaformer_ref_reorder = []

    for _, src_ref in enumerate(DGCNN_reference):
        for idx, ref in enumerate(Summaformer_reference):
            if src_ref[:100].lower() in ref.lower():
                Summaformer_ref_reorder.append(idx)
                break

    assert len(set(Summaformer_ref_reorder)) == 22, f"There has been documents not updated, actual length: {len(set(Summaformer_ref_reorder))}"

    Summaformer_reference = [Summaformer_reference[idx] for idx in Summaformer_ref_reorder]
    Summaformer_system = [Summaformer_system[idx] for idx in Summaformer_ref_reorder]

    score_path = f"../leaderboard_splits/metric_agreement/spice_prep/SummaRuNNer/SummaRuNNer_spice_{data_partition_idx}.txt"
    with open(score_path, "r") as f:
        scores = f.readlines()
        scores = scores[0]
        scores = scores.split(",")
        scores = [  float(s.strip()) for s in scores if s.strip() != "" ]

    scores = [scores[idx] for idx in Summaformer_ref_reorder]

    summa_top5 = np.argsort(scores)[::-1][:5]
    summa_last5 = np.argsort(scores)[:5]


    # # ########################################################################################################################################################################################################

    system_file = f"../leaderboard_splits/metric_agreement/spice_prep/BERTSum/system_{data_partition_idx}.txt"
    reference_file = f"../leaderboard_splits/metric_agreement/spice_prep/BERTSum/reference_{data_partition_idx}.txt"
    doc_path = f"../leaderboard_splits/metric_agreement/spice_prep/BERTSum/documents_{data_partition_idx}.txt"

    BERTSumm_reference, BERTSumm_system, BERTSum_doc = read_ref_and_summ(system_file, reference_file, doc_path)

    
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


    score_path = f"../leaderboard_splits/metric_agreement/spice_prep/BERTSum/BERTSum_spice_{data_partition_idx}.txt"
    with open(score_path, "r") as f:
        scores = f.readlines()
        scores = scores[0]
        scores = scores.split(",")
        scores = [  float(s.strip()) for s in scores if s.strip() != "" ]

    scores = [scores[idx] for idx in BERTSumm_ref_reorder]


    bertsum_top5 = np.argsort(scores)[::-1][:5]
    bertsum_last5 = np.argsort(scores)[:5]

    ########################################################################################################################################################################################################

    system_file = f"../leaderboard_splits/metric_agreement/spice_prep/Bart/system_{data_partition_idx}.txt"
    reference_file = f"../leaderboard_splits/metric_agreement/spice_prep/Bart/reference_{data_partition_idx}.txt"
    doc_path = f"../leaderboard_splits/metric_agreement/spice_prep/Bart/documents_{data_partition_idx}.txt"


    Bart_reference, Bart_system, _ = read_ref_and_summ(system_file, reference_file, doc_path)

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

    score_path = f"../leaderboard_splits/metric_agreement/spice_prep/Bart/Bart_spice_{data_partition_idx}.txt"
    with open(score_path, "r") as f:
        scores = f.readlines()
        scores = scores[0]
        scores = scores.split(",")
        scores = [  float(s.strip()) for s in scores if s.strip() != "" ]

    scores = [scores[idx] for idx in Bart_ref_reorder]


    bart_top5 = np.argsort(scores)[::-1][:5]
    bart_last5 = np.argsort(scores)[:5]


    entry_labels = [ 'dgcnn', 'bigbird', 'summa', 'bertsum', 'bart' ]
    tops = [ dgcnn_top5, bigbird_top5, summa_top5, bertsum_top5, bart_top5 ]
    bottoms = [ dgcnn_last5, bigbird_last5, summa_last5, bertsum_last5, bart_last5 ]

    top_agreement[data_partition_idx] = tops
    bottom_agreement[data_partition_idx] = bottoms


savepath = "../leaderboard_splits/metric_agreement/stats/spice.pickle"
with open(savepath, "wb") as f:
    pickle.dump((top_agreement, bottom_agreement), f)
