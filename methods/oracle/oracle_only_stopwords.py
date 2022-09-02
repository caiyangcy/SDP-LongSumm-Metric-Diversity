import random
import os
import json
import pandas as pd
from nltk import tokenize
from difflib import SequenceMatcher
import numpy as np
from nltk.corpus import stopwords

SEED = 123
random.seed(SEED)


longsumm_ext_docs = "../../datasets/dataset/original/extractive/test/document"
longsumm_ext_summs = "../../datasets/dataset/original/extractive/test/summary"

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

for idx in range(10):

    print(f" ----- idx: {idx} -----")

    longsumm_abs_docs = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/test/document"
    longsumm_abs_summs = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/test/summary"

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

                        summ_sents = summ_info['summary']
                        summ_replaced = []
                        for summ_sent in summ_sents:
                            for subs_tuple in substitutions:
                                summ_sent = summ_sent.replace(subs_tuple[0], subs_tuple[1])
                            summ_replaced.append(summ_sent)

                        summ_replaced = " ".join(summ_replaced) 
                        
                        summ_words = tokenize.word_tokenize(summ_replaced)

                        oracle_selection = []
                        
                        stop_words = set(stopwords.words('english'))

                        for word in summ_words:
                            if word in stop_words:
                                oracle_selection.append(word)
                                    
                        oracle_summaries.append( " ".join(sent.strip() for sent in oracle_selection)  )
                        
            except Exception as e:
                print(f"{doc_path} error: {e}")



    abs_pred_path = f"../../leaderboard_splits//split_{idx}/baseline/oracle"

    with open(f"{abs_pred_path}/oracle_abstractive_selection_stopwords_from_summ.txt", "w") as f:
        selection_to_write = "\n".join(oracle_summaries)
        f.write(selection_to_write)
