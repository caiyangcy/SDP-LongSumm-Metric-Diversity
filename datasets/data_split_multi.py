"""
    This file is used to split the datasets into "training" and "blind test" datasets. 
    The "training" set is like the datasets available to participants during the competitions.
    The "blind test" set is like the evaluation set used in the competition.
    Here the datasets refer to the raw output from science-parse and summary provided.
"""

import os
import random
import pandas as pd
import json
import argparse
import pickle


SEED = 42
SPLIT = 22


def clean(file):
    if 'import json' in file : return False
    file = file.split('{')
    if len(file) == 1: return False
    return '{' + '{'.join(file[1:])

def prepare_doc_and_summ_extrative(args):
    doc_dir = args.doc_dir
    summ_dir = args.summ_dir

    document = []
    summary = []

    for subdir, _, files in os.walk(doc_dir):

        for file in files:

            doc_path = subdir + os.sep + file
            doc_path = doc_path

            doc_name = file.rsplit('.', 1)[0]
            
            # with open(doc_path, 'r') as f:
            #     text = clean(f.read())
            #     if text:
            #         text = json.loads(text)['metadata']
            #         for key in text.keys() - ['title','sections','references','abstractText']:
            #             text.pop(key)
            #     else:
            #         continue
            
            document.append( doc_path )

            summ_path = summ_dir + os.sep + doc_name + '.txt' # json file for summary
            if not os.path.isfile(summ_path):
                summ_path = summ_dir + os.sep + doc_name + ' .txt' # json file for summary with space - this is due some files contain a space

            # summ = pd.read_csv(summ_path, sep='\t', header=None)
            
            # summ_sents = summ.iloc[:,-1].values
            summary.append( summ_path ) # " ".join( sent for sent in summ_sents ).strip()  )
           
    print('Scanning done!')

    return document, summary



def prepare_doc_and_summ_abstrative(args):
    doc_dir = args.doc_dir
    summ_dir = args.summ_dir

    document = []
    summary = []

    missing_docs = [] # note since we did not download all the pdfs then there will be missing documents

    for subdir, _, files in os.walk(summ_dir):

        for file in files:
            
            summ_path = subdir + os.sep + file

            summ_path = summ_path.replace('`', '').replace("'", '')

            with open(summ_path, 'r') as file:
                summ_info = json.load(file)
                doc_id = str(summ_info['id'])

                if doc_id == "39155988": # notice that this file is not processed successfully
                    continue
                
                # summ = " ".join(s for s in summ_info['summary']).strip()
            
            doc_path = doc_dir + os.sep + doc_id + ".json"
            doc_path = doc_path.replace('`', '').replace("'", '')

            try:
                with open(doc_path, 'r') as file:
                    document.append( doc_path )  
                    summary.append( summ_path )
            
            except Exception as e:
                # print(f"{summ_path} summary goes wrong, {e}")
                missing_docs.append(doc_path)

    print('Scanning done!')
    print(f"{len(missing_docs)} missing in total.") # should be 46

    return document, summary



def save_docs_and_summ(doc_paths, summ_paths, save_path):

    if not save_path.endswith("/"):
        save_path += "/"

    for idx, doc_path in enumerate(doc_paths):
        summ_path = summ_paths[idx]

        doc_name = doc_path.rsplit("/", 1)[-1]  #doc_path.rsplit(".", 1)[0].split("/")[-1]
        summ_name = summ_path.rsplit("/", 1)[-1] #summ_path.rsplit(".", 1)[0].split("/")[-1]

        doc_save = save_path + "document/" + doc_name
        summ_save = save_path + "summary/" + summ_name

        with open(doc_path, 'r') as f:
            # text = f.read()

            with open(doc_path, 'r') as f:
                text = clean(f.read())
                if text:
                    text = json.loads(text)['metadata']
                    for key in text.keys() - ['title','sections','references','abstractText']:
                        text.pop(key)
                else:
                    continue
        
        with open(doc_save, "w") as f:
            json.dump(text, f)
        

        if summ_name.endswith(".json"): # abstractive summary
            with open(summ_path, 'r') as file:
                summ = json.load(file)
            
            with open(summ_save, "w") as f:
                json.dump(summ, f)
        else:
            summ = pd.read_csv(summ_path, sep='\t', header=None)

            summ.to_csv(summ_save, sep='\t', index=False, header=False)
    
    print("Done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--doc_dir', help='link to the folder that contains the documents')
    parser.add_argument('--summ_dir', help='link to the folder that contains the summaries')
    parser.add_argument('--is_extractive', action='store_true', help='whether preprocessing on extractive docs')

    args = parser.parse_args()


    for idx, SEED in enumerate([10, 15, 32, 38, 44, 48, 123, 222, 456, 555]):

        random.seed(SEED)

        is_extractive = args.is_extractive
        if is_extractive:
            parent_folder = f"dataset_multiple_splits/split_{idx}/original/extractive"
            os.makedirs(parent_folder, mode=0o775, exist_ok=True)
            document, summary = prepare_doc_and_summ_extrative(args)
        else:
            parent_folder = f"dataset_multiple_splits/split_{idx}/original/abstractive"
            os.makedirs(parent_folder, mode=0o775, exist_ok=True)
            document, summary = prepare_doc_and_summ_abstrative(args)


        os.makedirs(parent_folder+"/avail/document", mode=0o775, exist_ok=True)
        os.makedirs(parent_folder+"/avail/summary", mode=0o775, exist_ok=True)
        os.makedirs(parent_folder+"/test/document", mode=0o775, exist_ok=True)
        os.makedirs(parent_folder+"/test/summary", mode=0o775, exist_ok=True)

        doc_summ = list(zip(document, summary))
        random.shuffle(doc_summ)
        document, summary = zip(*doc_summ)
        document = list(document)
        summary = list(summary)

        train_x, train_y = document[SPLIT:], summary[SPLIT:]
        val_x, val_y = document[:SPLIT], summary[:SPLIT]

        savepath = f"{parent_folder}/avail"
        save_docs_and_summ(train_x, train_y, savepath)

        savepath = f"{parent_folder}/test"
        save_docs_and_summ(val_x, val_y, savepath)
