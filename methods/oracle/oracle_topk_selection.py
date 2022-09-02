import random
import os
import json
import pandas as pd
from nltk import tokenize
from difflib import SequenceMatcher
import numpy as np

SEED = 123
random.seed(SEED)


substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]


def clean(file):
    if 'import json' in file : return False
    file = file.split('{')
    if len(file) == 1: return False
    return '{' + '{'.join(file[1:])


def revert_subs(text):
    text_reverted = text
    for subs_tuple in substitutions:
        text_reverted = text_reverted.replace(subs_tuple[1], subs_tuple[0])

    return text_reverted


##################################### abstractive summaries #####################################

for data_idx in range(10):

    print(f" ----- idx: {data_idx} -----")

    longsumm_abs_docs = f"../../datasets/dataset_multiple_splits/split_{data_idx}/original/abstractive/test/document"
    longsumm_abs_summs = f"../../datasets/dataset_multiple_splits/split_{data_idx}/original/abstractive/test/summary"

    oracle_summaries = []
    real_summaries = []


    for subdir, dirs, files in os.walk(longsumm_abs_summs):
        print(f"{len(files)} files in total")
        for file in files:
            
            summ_path = subdir + os.sep + file

            summ_path = summ_path.replace('`', '').replace("'", '')

            with open(summ_path, 'r') as file:
                summ_info = json.load(file)
                doc_id = str(summ_info['id'])

                if doc_id == "39155988": # notice that this file is not processed successfully
                    continue
                        
            doc_path = longsumm_abs_docs + os.sep + doc_id + ".json"

            try:
                with open(doc_path, 'r') as f:

                    text = clean(f.read())
                    if text:
                        text = json.loads(text)#['metadata']
                        for key in text.keys() - ['title','sections','references','abstractText']:
                            text.pop(key)

                        
                        doc = text["abstractText"]

                        if text["sections"] is not None:
                            for section in text['sections']:
                                doc += ' ' + section['text'] 

                        doc_replaced = doc 
                        for subs_tuple in substitutions:
                            doc_replaced = doc_replaced.replace(subs_tuple[0], subs_tuple[1])
                        doc_replaced = doc_replaced.replace("\n", " ")
                        
                        doc = tokenize.sent_tokenize(doc_replaced)

                        random.shuffle(doc)


                        oracle_selection = []

                        summ_sents = summ_info['summary']
                        summ_replaced = []
                        for summ_sent in summ_sents:
                            for subs_tuple in substitutions:
                                summ_sent = summ_sent.replace(subs_tuple[0], subs_tuple[1])
                            summ_replaced.append(summ_sent)
                        
                        for summ_sent in summ_replaced:
                            if summ_sent in doc:
                                summ_sent_reverted = revert_subs(summ_sent)
                                oracle_selection.append(summ_sent_reverted)

                            else:
                                summ_ratios = [SequenceMatcher(None, sent, summ_sent).ratio() for sent in doc]

                                # greey approach in terms of the best matching
                                # idx = summ_ratios.index(max(summ_ratios))
                                # labels[idx] = 1

                                idx = summ_ratios.index(max(summ_ratios))
                                sent_picked = doc[idx]

                                summ_sent_reverted = revert_subs(sent_picked)
                                oracle_selection.append(summ_sent_reverted)
                                    

                        oracle_summaries.append( " ".join(sent.strip() for sent in oracle_selection)  )
                        
                        real_summaries.append( " ".join(sent.strip() for sent in summ_info['summary'])  )

            except Exception as e:
                print(f"{doc_path} error: {e}")



    abs_pred_path = f"../../leaderboard_splits/split_{data_idx}/baseline/oracle"

    with open(f"{abs_pred_path}/oracle_abstractive_selection.txt", "w") as f:
        selection_to_write = "\n".join(oracle_summaries)
        f.write(selection_to_write)

    with open(f"{abs_pred_path}/abstractive_groudtruth.txt", "w") as f:
        truth_to_write = "\n\n".join(real_summaries)
        f.write(truth_to_write)