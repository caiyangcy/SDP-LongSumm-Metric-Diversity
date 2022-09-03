import numpy as np
import sys
import re
import csv
import os
import json
import pickle
from bert_score import BERTScorer
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
    return text.replace("`` ", "\"").replace(" ''", "\"").replace(" ,", ",").replace(" :", ":").replace("( ", "(").replace(" )", ")")



def save_for_spice(documents, reference, system, savepath, data_idx):
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    with open(f"{savepath}/documents_{data_idx}.txt", "w") as f:
        doc_to_save = [ doc.strip().encode('unicode_escape').decode()  for doc in documents]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)

    with open(f"{savepath}/reference_{data_idx}.txt", "w") as f:
        doc_to_save = [ ref.strip().encode('unicode_escape').decode()  for ref in reference]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)


    with open(f"{savepath}/system_{data_idx}.txt", "w") as f:
        doc_to_save = [ sys.strip().encode('unicode_escape').decode()  for sys in system]
        doc_to_write = "\n".join(doc_to_save)
        f.write(doc_to_write)




# extractive: DGCNN, SummaRuNNer, BERTSum
# abstractive: Bart, Bigbird-Pegasus


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


    savepath = f"../leaderboard_splits/metric_agreement/spice_prep/DGCNN"
    save_for_spice(DGCNN_doc, DGCNN_reference, DGCNN_system, savepath, data_partition_idx)


    ########################################################################################################################################################################################################


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



    savepath = f"../leaderboard_splits/metric_agreement/spice_prep/Bigbird-Pegasus"
    save_for_spice(Bigbird_Small_No_doc, Bigbird_Small_No_reference, Bigbird_Small_No_system, savepath, data_partition_idx)


    # # ########################################################################################################################################################################################################


    system_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/SummaFormer_train_on_ext/system.txt"
    reference_file = f"../leaderboard_splits/split_{data_partition_idx}/baseline/SummaFormer_train_on_ext/reference.txt"

    Summaformer_reference, Summaformer_system, _ , _ = read_ref_and_summ(system_file, reference_file)

    doc_file = f"../datasets/dataset_multiple_splits/split_{data_partition_idx}/SummaFormer/SummaFormer-Actual-Input-Doc/doc.txt"
    with open(doc_file, "r") as f:
        Summaformer_docs = f.readlines()



    savepath = f"../leaderboard_splits/metric_agreement/spice_prep/SummaRuNNer"
    save_for_spice(Summaformer_docs, Summaformer_reference, Summaformer_system, savepath, data_partition_idx)


    print("\n\n Summaformer_reference[14]: ", Summaformer_reference[14])



    ########################################################################################################################################################################################################

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



    savepath = f"../leaderboard_splits/metric_agreement/spice_prep/BERTSum"
    save_for_spice(BERTSum_doc, BERTSumm_reference, BERTSumm_system, savepath, data_partition_idx)


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

    savepath = f"../leaderboard_splits/metric_agreement/spice_prep/Bart"
    save_for_spice(Bart_docs, Bart_reference, Bart_system, savepath, data_partition_idx)
