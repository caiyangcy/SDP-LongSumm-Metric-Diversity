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


def topk_similar(doc, sents, topk=3):
    selected_idx = set()
    topk_selection = []
    for summ_sent in sents:
        summ_ratios = [SequenceMatcher(None, sent, summ_sent).ratio() for sent in doc]

        top_indices = np.argsort(summ_ratios)[::-1][:topk]

        for idx in top_indices:
            if idx in selected_idx: 
                continue

            selected_idx.add(idx)
            sent_picked = doc[idx]

            summ_sent_reverted = revert_subs(sent_picked)
            topk_selection.append(summ_sent_reverted)
            
    return topk_selection

def context(doc, sents):
    # take the sentece before and after the most similar sentence
    selected_idx = set()
    context_selection = []

    for summ_sent in sents:
        summ_ratios = [SequenceMatcher(None, sent, summ_sent).ratio() for sent in doc]

        idx = summ_ratios.index(max(summ_ratios))

        context_idx = [max(idx-1, 0), idx, min(idx+1, len(doc)-1)]
        for idx in context_idx:
            if idx in selected_idx: 
                continue

            selected_idx.add(idx)
            sent_picked = doc[idx]

            summ_sent_reverted = revert_subs(sent_picked)
            context_selection.append(summ_sent_reverted)
            
    return context_selection

def paragraph_direct_matching(doc, sents):
    # two ways of selecting the paragraph
    # 1. directly matching paragraph with the sentence
    # 2. picking up the paragraph where the sentece belongs to

    doc = doc.split("\n")
    paragraph_selection = []
    for summ_sent in sents:
        summ_ratios = [SequenceMatcher(None, pgh, summ_sent).ratio() for pgh in doc]

        idx = summ_ratios.index(max(summ_ratios))
        pgh_picked = doc[idx]

        summ_pgh_reverted = revert_subs(pgh_picked)
        summ_pgh_reverted.replace("\n", " ").strip()
        paragraph_selection.append(summ_pgh_reverted)

    return paragraph_selection


def paragraph_belong(doc, sents):
    doc = doc.split("\n")
    paragraph_sents = [tokenize.sent_tokenize(pgh) for pgh in doc]
    paragraph_sents = [pgh_sent for pgh_sent in paragraph_sents if len(pgh_sent) > 0]
    paragraph_selection = []

    for summ_sent in sents:
        pgh_max = []
        for pgh_sent in paragraph_sents:
            summ_ratios = [SequenceMatcher(None, pgh, summ_sent).ratio() for pgh in pgh_sent]
            # print("pgh_sent: ", pgh_sent)
            # print("summ ratios: ", summ_ratios)
            pgh_max.append(max(summ_ratios))

        max_idx = np.argmax(pgh_max)
        pgh = doc[max_idx]

        pgh_reverted = revert_subs(pgh)
        pgh_reverted = pgh_reverted.replace("\n", " ").strip()
        paragraph_selection.append(pgh_reverted)

    return paragraph_selection


##################################### abstractive summaries #####################################

for data_idx in range(10):

    print(f" ----- idx: {data_idx} -----")

    longsumm_abs_docs = f"../../datasets/dataset_multiple_splits/split_{data_idx}/original/abstractive/test/document"
    longsumm_abs_summs = f"../../datasets/dataset_multiple_splits/split_{data_idx}/original/abstractive/test/summary"

    topk_summaries = []
    context_summaries = []
    pgh_direct_summaries = []
    pgh_belong_summaries = []

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


                
                    oracle_selection = []

                    summ_sents = summ_info['summary']
                    summ_replaced = []
                    for summ_sent in summ_sents:
                        for subs_tuple in substitutions:
                            summ_sent = summ_sent.replace(subs_tuple[0], subs_tuple[1])
                        summ_replaced.append(summ_sent)
                    


                    pgh_direct_selection = paragraph_direct_matching(doc_replaced, summ_replaced)

                    pgh_belong_selection = paragraph_belong(doc_replaced, summ_replaced)


                    doc_replaced = doc_replaced.replace("\n", " ")
                    
                    doc = tokenize.sent_tokenize(doc_replaced)

                    random.shuffle(doc)

                    topk_selection = topk_similar(doc, summ_sents, 3)
                    context_selection = context(doc, summ_sents)


                    topk_selection_sent = " ".join(sent.strip() for sent in topk_selection)
                    topk_selection_sent_split = topk_selection_sent.split()
                    if len(topk_selection_sent_split) > 600:
                        topk_selection_sent = " ".join(sent for sent in topk_selection_sent_split[:600])


                    context_selection_sent = " ".join(sent.strip() for sent in context_selection)
                    context_selection_sent_split = context_selection_sent.split()
                    if len(context_selection_sent_split) > 600:
                        context_selection_sent = " ".join(sent for sent in context_selection_sent_split[:600])


                    pgh_direct_selection_sent = " ".join(sent.strip() for sent in pgh_direct_selection)
                    pgh_direct_selection_sent_split = pgh_direct_selection_sent.split()
                    if len(pgh_direct_selection_sent_split) > 600:
                        pgh_direct_selection_sent = " ".join(sent for sent in pgh_direct_selection_sent_split[:600])


                    pgh_belong_selection_sent = " ".join(sent.strip() for sent in pgh_belong_selection)
                    pgh_belong_selection_sent_split = pgh_belong_selection_sent.split()
                    if len(pgh_belong_selection_sent_split) > 600:
                        pgh_belong_selection_sent = " ".join(sent for sent in pgh_belong_selection_sent_split[:600])

                    topk_summaries.append( topk_selection_sent )
                    context_summaries.append( context_selection_sent )
                    pgh_direct_summaries.append( pgh_direct_selection_sent )
                    pgh_belong_summaries.append( pgh_belong_selection_sent )



    abs_pred_path = f"../../leaderboard_splits/split_{data_idx}/baseline/oracle"

    with open(f"{abs_pred_path}/oracle_abstractive_top3.txt", "w") as f:
        selection_to_write = "\n".join(topk_summaries)
        f.write(selection_to_write)


    with open(f"{abs_pred_path}/oracle_abstractive_context.txt", "w") as f:
        selection_to_write = "\n".join(context_summaries)
        f.write(selection_to_write)


    with open(f"{abs_pred_path}/oracle_abstractive_pgh_direct.txt", "w") as f:
        selection_to_write = "\n".join(pgh_direct_summaries)
        f.write(selection_to_write)


    with open(f"{abs_pred_path}/oracle_abstractive_pgh_belong.txt", "w") as f:
        selection_to_write = "\n".join(pgh_belong_summaries)
        f.write(selection_to_write)