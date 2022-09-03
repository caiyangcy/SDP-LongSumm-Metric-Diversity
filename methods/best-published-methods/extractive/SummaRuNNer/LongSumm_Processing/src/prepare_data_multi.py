import json , os , pickle, argparse
import random
from rouge_score import rouge_scorer
from nltk import tokenize
import pandas as pd
from json_dict import clean
from tqdm import tqdm 
from difflib import SequenceMatcher
from multiprocessing import Process, Queue, Manager
import numpy as np

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]


def process_data(papers, process_num, return_dict, extractive=True):
    updated_data = []

    if extractive:
        for ind, file in tqdm(enumerate(papers)):
            if file["sections"] is None or file['summary'] is None:
                continue
            updated_data.append(update_format_extractive(file))
    else:
        for ind, file in tqdm(enumerate(papers)):
            section_missing = file["sections"] is None or file['summary'] is None
            updated_data.append(update_format_abstractive(file, section_missing))

    return_dict[process_num] = updated_data


def abstractive_nonparsable():
    abstractive_paper = prepare_doc_and_summ_abstrative(abstractive_doc_dir, abstractive_summ_dir)

    for ind, file in enumerate(abstractive_paper):
        if file["sections"] is None or file['summary'] is None:
            print("ind: ", ind)
            print(file)


def get_datapoints_parallel(num_processes, extractive_doc_dir, extractive_summ_dir, abstractive_doc_dir, abstractive_summ_dir, data_type, idx):
    extractive_paper = prepare_doc_and_summ_extrative(extractive_doc_dir, extractive_summ_dir)
    abstractive_paper = prepare_doc_and_summ_abstrative(abstractive_doc_dir, abstractive_summ_dir)

    manager = Manager()
    return_dict = manager.dict()

    processes = []

    print('---- multi processing starts on extractive dataset ----')
    print(f" ---- extractive paper number: {len(extractive_paper)} ----")

    avg =  int( np.ceil( len(extractive_paper) / num_processes ) )
    last = 0

    for w in range(num_processes):
        if last < len(extractive_paper):
            p = Process(target=process_data, args=(extractive_paper[ last: last+avg ], w, return_dict, True))

            p.daemon = True
            p.start()
            processes.append(p)

            last += avg 

    for proc in processes:
        proc.join()

    extractive_updated_data = []
    for proc_data in return_dict.values():
        extractive_updated_data += proc_data

    print(f" ---- extractive paper processed: {len(extractive_updated_data)} ----")

    manager = Manager()
    return_dict = manager.dict()

    processes = []

    print('---- multi processing starts on abstractive dataset ----')

    avg =  int(len(abstractive_paper) / num_processes)+1
    last = 0
    print(f" ---- abstractive paper number: {len(abstractive_paper)} avg: {avg} ----")

    for w in range(num_processes):

        if last < len(abstractive_paper):
            p = Process(target=process_data, args=(abstractive_paper[ last: last+avg ], w, return_dict, False ))

            p.daemon = True
            p.start()
            processes.append(p)

            last += avg 

    for proc in processes:
        proc.join()

    abstractive_updated_data = []
    for proc_data in return_dict.values():
        abstractive_updated_data += proc_data

    print(f" ---- abstractive paper processed: {len(abstractive_updated_data)} ----")

    print(f" ---- saving data ---- ")

    if data_type == "avail":

        ext_train, ext_val = extractive_updated_data[ : int(len(extractive_updated_data)*0.9) ], extractive_updated_data[ int(len(extractive_updated_data)*0.9): ]
        abs_train, abs_val = abstractive_updated_data[ : int(len(abstractive_updated_data)*0.9) ], abstractive_updated_data[ int(len(abstractive_updated_data)*0.9): ]

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_train.txt','w') as f:
            json.dump(ext_train, f)

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_eval.txt','w') as f:
            json.dump(ext_val, f)


        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abstractive/LongSumm_abstractive_withlabels_avail_train.txt','w') as f:
            json.dump(abs_train, f)
        
        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abstractive/LongSumm_abstractive_withlabels_avail_eval.txt','w') as f:
            json.dump(abs_val, f)
        
        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abs_ext/LongSumm_withlabels_avail_train.txt','w') as f:
            json.dump(ext_train+abs_train, f)

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abs_ext/LongSumm_withlabels_avail_eval.txt','w') as f:
            json.dump(ext_val+abs_val, f)

    else:

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/extractive/LongSumm_extractive_withlabels_test.txt','w') as f:
            json.dump(extractive_updated_data, f)

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abstractive/LongSumm_abstractive_withlabels_test.txt','w') as f:
            json.dump(abstractive_updated_data, f)

        with open(f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abs_ext/LongSumm_withlabels_test.txt','w') as f:
            json.dump(extractive_updated_data+abstractive_updated_data, f)
        


###################################################################################################################
###################################################################################################################
###################################################################################################################

def abstractive_labeling(doc, true_summ):
    doc = tokenize.sent_tokenize(doc)
    # print(len(doc))
    summ = ''
    cur_score = 0
    labels = [ 0 for x in range(len(doc))]
    while True:
        for ind,x in enumerate(doc):
            if labels[ind] == 1 : continue
            cur_summ = summ + ' ' + doc[ind]
            score = scorer.score(cur_summ, true_summ)['rouge1'].fmeasure
            if score > cur_score :
                cur_score = score
                summ = cur_summ
                labels[ind] = 1
                break
        else : break
    return labels


def update_format_abstractive(paper, section_missing=False):
    updated_batch = {}

    updated_batch['doc'] = ''

    if "abstractText" in paper and paper["abstractText"]:
        updated_batch['doc'] += paper['abstractText'] 

    if not section_missing:
        for section in paper['sections']:
            updated_batch['doc'] += ' ' + section['text'] 
    # else:
    #     updated_batch['doc'] +=  paper['abstractText'] 

    updated_batch['summaries'] = paper['summary'].strip() #' '.join( summ.strip() for summ in paper['summary'] )

    for subs_tuple in substitutions:
        # replace the problematic tokens with temporary substitution
        updated_batch['doc'] = updated_batch['doc'].replace(subs_tuple[0], subs_tuple[1])
        updated_batch['summaries'] = updated_batch['summaries'].replace(subs_tuple[0], subs_tuple[1])

    updated_batch['labels'] = abstractive_labeling(updated_batch['doc'], updated_batch['summaries'])


    for subs_tuple in substitutions:
        # replace the problematic tokens with temporary substitution
        updated_batch['doc'] = updated_batch['doc'].replace(subs_tuple[1], subs_tuple[0])
        updated_batch['summaries'] = updated_batch['summaries'].replace(subs_tuple[1], subs_tuple[0])

    return updated_batch
    


def prepare_doc_and_summ_abstrative(doc_dir, summ_dir):

    document = []

    missing_docs = [] # note since we did not download all the pdfs then there will be missing documents

    for subdir, dirs, files in os.walk(summ_dir):

        for file in files:
            
            summ_path = subdir + os.sep + file

            summ_path = summ_path.replace('`', '').replace("'", '')

            with open(summ_path, 'r') as file:
                summ_info = json.load(file)
                doc_id = str(summ_info['id'])

                if doc_id == "39155988": # notice that this file is not processed successfully
                    continue
                
                summ = " ".join(s for s in summ_info['summary']).strip()
            
            doc_path = doc_dir + os.sep + doc_id + ".json"
            doc_path = doc_path.replace('`', '').replace("'", '')

            try:
                with open(doc_path, 'r') as f:

                    text = clean(f.read())
                    if text:
                        text = json.loads(text)#['metadata']
                        for key in text.keys() - ['title','sections','references','abstractText']:
                            text.pop(key)

                        if text["sections"] is None:
                            print(f"NOTE: {doc_id} has empty sections.")
                        text['summary'] = summ
                        document.append(text)

            except Exception as e:
                missing_docs.append(doc_path)

    print('Scanning done!')
    print(f"{len(missing_docs)} missing in total.") # should be 46

    return document


###################################################################################################################
###################################################################################################################
###################################################################################################################


def extractive_labeling(doc, summ):
    doc_replaced = doc 
    for subs_tuple in substitutions:
        doc_replaced = doc_replaced.replace(subs_tuple[0], subs_tuple[1])

    summ_replaced = []
    for summ_sent in summ:
        for subs_tuple in substitutions:
            summ_sent = summ_sent.replace(subs_tuple[0], subs_tuple[1])
        summ_replaced.append(summ_sent)


    doc = tokenize.sent_tokenize(doc_replaced)
    labels = [0]*len(doc)

    for summ_sent in summ_replaced:

        if summ_sent in doc:
            # only set once - assume that there are only one matching
            # idx = doc.index(summ_sent)
            # labels[idx] = 1

            # set multiple times - assume there are more than one matching
            ind = [idx for idx in range(len(doc)) if doc[idx] == summ_sent]
            for idx in ind:
                labels[idx] = 1

        else:
            summ_ratios = [SequenceMatcher(None, sent, summ_sent).ratio() for sent in doc]

            # greey approach in terms of the best matching
            # idx = summ_ratios.index(max(summ_ratios))
            # labels[idx] = 1
            if max(summ_ratios) < 0.8:
                continue
            # greey approach in terms of number of summary sentences
            arg_ind = np.argsort(summ_ratios)[::-1]
            for idx in arg_ind:
                if not labels[idx]:
                    labels[idx] = 1
                    break 

    return labels



def update_format_extractive(paper):
    updated_batch = {}

    updated_batch['doc'] = ''

    if "abstractText" in paper and paper["abstractText"]:
        updated_batch['doc'] += paper['abstractText'] 

    for section in paper['sections']:
        updated_batch['doc'] += ' ' + section['text'] 

    updated_batch['summaries'] = ' '.join(paper['summary'])

    updated_batch['labels'] = extractive_labeling(updated_batch['doc'], paper['summary'])

    return updated_batch


def prepare_doc_and_summ_extrative(doc_dir, summ_dir):

    document = []

    for subdir, dirs, files in os.walk(doc_dir):

        for file in files:

            doc_path = subdir + os.sep + file
            doc_path = doc_path#.replace('`', '').replace("'", '')

            doc_name = file.rsplit('.', 1)[0]
            
            with open(doc_path, 'r') as f:
                text = clean(f.read())
                if text:
                    text = json.loads(text)#['metadata']
                    for key in text.keys() - ['title','sections','references','abstractText']:
                        text.pop(key)
                else:
                    continue
            
            summ_path = summ_dir + os.sep + doc_name + '.txt' # json file for summary
            if not os.path.isfile(summ_path):
                summ_path = summ_dir + os.sep + doc_name + ' .txt' # json file for summary with space - this is due some files contain a space

            summ = pd.read_csv(summ_path, sep='\t', header=None)
            
            summ_sents = list( summ.iloc[:,-1].values )
            # summ_sents = " ".join( sent for sent in summ_sents ).strip()
            text['summary'] = summ_sents
            document.append( text )

    print('Scanning done!')

    return document




if __name__ == '__main__':

    data_type = "avail"

    for idx in range(10):
        print(f"idx: {idx}")

        extractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/extractive/{data_type}/document"
        extractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/extractive/{data_type}/summary"

        abstractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/abstractive/{data_type}/document"
        abstractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/abstractive/{data_type}/summary"

        get_datapoints_parallel(20, extractive_doc_dir, extractive_summ_dir, abstractive_doc_dir, abstractive_summ_dir, data_type, idx)


    data_type = "test"

    for idx in range(10):
        print(f"idx: {idx}")

        extractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/extractive/{data_type}/document"
        extractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/extractive/{data_type}/summary"

        abstractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/abstractive/{data_type}/document"
        abstractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/original/abstractive/{data_type}/summary"

        get_datapoints_parallel(20, extractive_doc_dir, extractive_summ_dir, abstractive_doc_dir, abstractive_summ_dir, data_type, idx)