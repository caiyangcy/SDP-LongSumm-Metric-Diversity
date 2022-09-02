# Random selection baseline
# 1. No preprocessing
# 2. Simple preprocessing
# 3. talksumm preprocessing
# 4. use abstract or not


import random
import os
import json
import pandas as pd
from nltk import tokenize


SEED = 123
random.seed(SEED)

RANDOM_K = 3
# RANDOM_K = 5
# RANDOM_K = 10


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


##################################### abstractive summaries #####################################


for idx in range(10):

    print(f" ----- idx: {idx} -----")

    longsumm_abs_docs = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/test/document"
    longsumm_abs_summs = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/test/summary"

    test_predictions = []
    test_summaries = []

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
            # doc_path = doc_path.replace('`', '').replace("'", '')

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

                        selected_sentences = " ".join(doc[:RANDOM_K])

                        test_predictions.append(selected_sentences)
                        
                        test_summaries.append( " ".join(sent.strip() for sent in summ_info['summary'])  )

            except Exception as e:
                print(f"{doc_path} error: {e}")



    abs_pred_path = f"../../leaderboard_splits/split_{idx}/baseline/random/"

    if not os.path.exists(abs_pred_path):
        os.mkdir(abs_pred_path)

    with open(f"{abs_pred_path}/random_{RANDOM_K}_abstractive_selection.txt", "w") as f:
        selection_to_write = "\n".join(test_predictions)
        f.write(selection_to_write)

    with open(f"{abs_pred_path}/abstractive_groudtruth.txt", "w") as f:
        truth_to_write = "\n\n".join(test_summaries)
        f.write(truth_to_write)