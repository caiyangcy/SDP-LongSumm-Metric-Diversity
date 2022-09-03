import json , os , pickle, argparse
import random
from rouge_score import rouge_scorer
from nltk import tokenize
import pandas as pd
from tqdm import tqdm 
from difflib import SequenceMatcher
from multiprocessing import Process, Queue, Manager
import numpy as np

## summaformer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
scorer_L = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def clean(file):
    if 'import json' in file : return False
    file = file.split('{')
    if len(file) == 1: return False
    return '{' + '{'.join(file[1:])

substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]

def replace_subs(text):
    for subs_tuple in substitutions:
        text = text.replace(subs_tuple[0], subs_tuple[1])
    return text

def revert_subs(text):
    for subs_tuple in substitutions:
        text = text.replace(subs_tuple[1], subs_tuple[0])
    return text



def process_data(papers, process_num, return_dict, extractive=True, data_type="avail"):
    updated_data = []

    eval_idx = int(len(papers)*0.9)

    if extractive:
        for ind, file in enumerate(papers):
            if file["sections"] is None or file['summary'] is None:
                continue
            
            is_eval = ind >= eval_idx
            updated_data.append(update_format_extractive(file, data_type, is_eval))


    else:
        for ind, file in enumerate(papers):
            section_missing = file["sections"] is None or file['summary'] is None

            is_eval = ind >= eval_idx
            updated_data.append(update_format_abstractive(file, section_missing, data_type, is_eval))

    return_dict[process_num] = updated_data


def abstractive_nonparsable():
    abstractive_paper = prepare_doc_and_summ_abstrative(abstractive_doc_dir, abstractive_summ_dir)

    for ind, file in enumerate(abstractive_paper):
            if file["sections"] is None or file['summary'] is None:
                print("ind: ", ind)
                print(file)


def get_datapoints_parallel(num_processes, extractive_doc_dir, extractive_summ_dir, abstractive_doc_dir, abstractive_summ_dir, data_type):
    extractive_paper = prepare_doc_and_summ_extrative(extractive_doc_dir, extractive_summ_dir)
    abstractive_paper = prepare_doc_and_summ_abstrative(abstractive_doc_dir, abstractive_summ_dir)

    # manager = Manager()
    # return_dict = manager.dict()

    # processes = []

    # print('---- multi processing starts on extractive dataset ----')
    # print(f" ---- extractive paper number: {len(extractive_paper)} ----")

    # avg =  int( np.ceil( len(extractive_paper) / num_processes ) )
    # last = 0

    # for w in range(15):
    #     if last < len(extractive_paper):
    #         p = Process(target=process_data, args=(extractive_paper[ last: last+avg ], w, return_dict, True, data_type))

    #         p.daemon = True
    #         p.start()
    #         processes.append(p)

    #         last += avg 

    # for proc in processes:
    #     proc.join()

    # extractive_updated_data = []
    # for proc_data in return_dict.values():
    #     extractive_updated_data += proc_data


    # print(f" ---- extractive paper processed: {len(extractive_updated_data)} ----")


    manager = Manager()
    return_dict = manager.dict()

    processes = []

    print('---- multi processing starts on abstractive dataset ----')

    avg =  int(len(abstractive_paper) / num_processes)
    last = 0
    print(f" ---- abstractive paper number: {len(abstractive_paper)} ----")

    for w in range(25):

        if last < len(abstractive_paper):
            p = Process(target=process_data, args=(abstractive_paper[ last: last+avg ], w, return_dict, False, data_type))

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

    # print(f" ---- saving data ---- ")

    # if data_type == "avail":

    #     ext_train, ext_val = extractive_updated_data[ : int(len(extractive_updated_data)*0.9) ], extractive_updated_data[ int(len(extractive_updated_data)*0.9): ]
    #     abs_train, abs_val = abstractive_updated_data[ : int(len(abstractive_updated_data)*0.9) ], abstractive_updated_data[ int(len(abstractive_updated_data)*0.9): ]

    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/extractive/LongSumm_extractive_withlabels_avail_train.txt','w') as f:
    #         json.dump(ext_train, f)

    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/extractive/LongSumm_extractive_withlabels_avail_eval.txt','w') as f:
    #         json.dump(ext_val, f)


    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abstractive/LongSumm_abstractive_withlabels_avail_train.txt','w') as f:
    #         json.dump(abs_train, f)
        
    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abstractive/LongSumm_abstractive_withlabels_avail_eval.txt','w') as f:
    #         json.dump(abs_val, f)
        
    #     # with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abs_ext/LongSumm_withlabels_avail_train.txt','w') as f:
    #     #     json.dump(ext_train+abs_train, f)

    #     # with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abs_ext/LongSumm_withlabels_avail_eval.txt','w') as f:
    #     #     json.dump(ext_val+abs_val, f)

    # else:

    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/extractive/LongSumm_extractive_withlabels_test.txt','w') as f:
    #         json.dump(extractive_updated_data, f)

    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abstractive/LongSumm_abstractive_withlabels_test.txt','w') as f:
    #         json.dump(abstractive_updated_data, f)

    #     with open('REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abs_ext/LongSumm_withlabels_test.txt','w') as f:
            # json.dump(extractive_updated_data+abstractive_updated_data, f)
        


###################################################################################################################
###################################################################################################################
###################################################################################################################

def abstractive_labeling(doc, true_summ):
    # note this function here is not greedy any longer since it's time consuming

    doc_sents = [sent[1] for sent in doc]
    labels = [0]*len(doc)

    for summ_sent in true_summ:

        if summ_sent in doc_sents:

            # set multiple times - assume there are more than one matching
            ind = [idx for idx in range(len(doc)) if doc[idx] == summ_sent]
            for idx in ind:
                labels[idx] = 1

        else:
            summ_ratios = [scorer.score(sent, summ_sent)['rouge1'].fmeasure for sent in doc_sents]

            # greey approach in terms of number of summary sentences
            arg_ind = np.argsort(summ_ratios)[::-1]
            for idx in arg_ind:
                if not labels[idx]:
                    labels[idx] = 1
                    break 
    
    doc_updated = [  [ sent[0], sent[1], labels[idx] ] for idx, sent in enumerate(doc) ]
    return doc_updated



def update_format_abstractive(paper, section_missing=False, data_type='avail', is_eval=False):

    doc = []

    if "abstractText" in paper and paper["abstractText"]:
        abstract = paper['abstractText'] 
        abstract = replace_subs(abstract)
        abstract_tok = tokenize.sent_tokenize(abstract)

        doc += [("abstract", sent) for sent in abstract_tok]

    if not section_missing:
        for section in paper['sections']:
            text = section['text'] 
            text = replace_subs(text)
            text_tok = tokenize.sent_tokenize(text)

            doc += [(section["heading"], sent) for sent in text_tok]


    summary = ' '.join( summ.strip() for summ in paper['summary'] )
    summary_replaced = replace_subs(summary)
    summary_replaced_sents = [ replace_subs(summ) for summ in paper['summary'] ]

    doc_updated = extractive_labeling(doc, summary_replaced_sents)


    extended_summ_format = {}
    doc_id = paper['doc_id']
    extended_summ_format['id'] = doc_id
    extended_summ_format['abstract'] = paper['abstractText'] if "abstractText" in paper else ""

    # print("paper['summary']: ", paper['summary'])
    extended_summ_format['gold'] = [ tokenize.word_tokenize(summ) for summ in paper['summary']]
    # print("gold summary: ", extended_summ_format['gold'])

    sentences = []
    section_id = {}

    for  sent in doc_updated:
        words = tokenize.word_tokenize(sent[1])
        words = [revert_subs(w) for w in words]

        section = sent[0]

        if section is None: 
            section = "Other"

        section = section.replace(".", "").lower()
        section = ''.join(i for i in section if not i.isdigit())
        if section.endswith("s"):
            section = section[:-1]
        if section in section_id:
            sec_id = section_id[section]
        else:
            sec_id = max(section_id.values()) + 1 if len(section_id) > 0 else 0
            section_id[section] = sec_id

        label = sent[2]

        rouge_L = scorer_L.score(sent[1], summary_replaced)['rougeL'].fmeasure

        info = [words, section, rouge_L, sent[1], label, sec_id]
        sentences.append(info)

    extended_summ_format['sentences'] = sentences

    if data_type != "test":
        usage_type = "val" if is_eval else "train"
        path = f'REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abstractive/{usage_type}/{doc_id}.json'
    else:
        path = f'REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/abstractive/test/{doc_id}.json'
    with open(path,'w') as f:
        json.dump(extended_summ_format, f)


    return extended_summ_format
    


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
                
                summ = summ_info['summary'] #" ".join(s for s in summ_info['summary']).strip()
            
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
                        text['doc_id'] = doc_id
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
    doc_sents = [sent[1] for sent in doc]
    labels = [0]*len(doc)

    for summ_sent in summ:

        if summ_sent in doc_sents:
            # only set once - assume that there are only one matching
            # idx = doc.index(summ_sent)
            # labels[idx] = 1

            # set multiple times - assume there are more than one matching
            ind = [idx for idx in range(len(doc)) if doc[idx] == summ_sent]
            for idx in ind:
                labels[idx] = 1

        else:
            summ_ratios = [SequenceMatcher(None, sent, summ_sent).ratio() for sent in doc_sents]

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
    
    doc_updated = [  [ sent[0], sent[1], labels[idx] ] for idx, sent in enumerate(doc) ]
    return doc_updated


"""

"id" (String): the paper ID

"abstract" (String): the abstract text of the paper. This field is different from "gold" field for the datasets that have different ground-truth than the abstract.

"gold" (List <List<>>): the ground-truth summary of the paper, where the inner list is the tokens associated with each gold summary sentence.

"sentences" (List <List<>>): the source sentences of the full-text. The inner list contains 5 indices, each of which represents different fields of the source sentence:

    Index [0]: tokens of the sentences (i.e., list of tokens).
    Index [1]: textual representation of the section that the sentence belongs to.
    Index [2]: Rouge-L score of the sentence with the gold summary.
    Index [3]: textual representation of the sentences.
    Index [4]: oracle label associated with the sentence (0, or 1).
    Index [5]: the section id assigned by sequential sentence classification package. For more information, please refer to this repository

"""


def update_format_extractive(paper, data_type, is_eval):

    doc = []

    if "abstractText" in paper and paper["abstractText"]:
        abstract = paper['abstractText'] 
        abstract = replace_subs(abstract)
        abstract_tok = tokenize.sent_tokenize(abstract)

        doc += [("abstract", sent) for sent in abstract_tok]

    for section in paper['sections']:
        text = section['text'] 
        text = replace_subs(text)
        text_tok = tokenize.sent_tokenize(text)

        doc += [(section["heading"], sent) for sent in text_tok]


    summary = ' '.join( summ.strip() for summ in paper['summary'] )
    summary_replaced = replace_subs(summary)
    summary_replaced_sents = [ replace_subs(summ) for summ in paper['summary'] ]

    doc_updated = extractive_labeling(doc, summary_replaced_sents)


    extended_summ_format = {}
    doc_id = paper['doc_id']
    extended_summ_format['id'] = doc_id
    extended_summ_format['abstract'] = paper['abstractText'] if "abstractText" in paper else ""


    extended_summ_format['gold'] = [ tokenize.word_tokenize(summ) for summ in paper['summary']]

    sentences = []
    section_id = {}

    for sent in doc_updated:
        words = tokenize.word_tokenize(sent[1])
        words = [revert_subs(w) for w in words]

        section = sent[0]
        if section is None: 
            section = "Other"

        section = section.replace(".", "").lower()
        section = ''.join(i for i in section if not i.isdigit())
        if section.endswith("s"):
            section = section[:-1]

        if section in section_id:
            sec_id = section_id[section]
        else:
            sec_id = max(section_id.values()) + 1 if len(section_id) > 0 else 0
            section_id[section] = sec_id

        label = sent[2]

        rouge_L = scorer_L.score(sent[1], summary_replaced)['rougeL'].fmeasure

        info = [words, section, rouge_L, sent[1], label, sec_id]
        sentences.append(info)

    extended_summ_format['sentences'] = sentences

    if data_type != "test":
        usage_type = "val" if is_eval else "train"
        path = f'REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/extractive/{usage_type}/{doc_id}.json'
    else:
        path = f'REPLACE-BY-YOUR-PATH/datasets/dataset/ExtendedSumm/extractive/test/{doc_id}.json'
    with open(path,'w') as f:
        json.dump(extended_summ_format, f)

    return extended_summ_format






def prepare_doc_and_summ_extrative(doc_dir, summ_dir):

    document = []

    for subdir, dirs, files in os.walk(doc_dir):

        for file in files:

            doc_path = subdir + os.sep + file
            doc_path = doc_path

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
            text['summary'] = summ_sents
            text['doc_id'] = doc_name
            document.append( text )

    print('Scanning done!')

    return document




if __name__ == '__main__':

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    data_type = "test"
    extractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset/original/extractive/{data_type}/document"
    extractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset/original/extractive/{data_type}/summary"

    abstractive_doc_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset/original/abstractive/{data_type}/document"
    abstractive_summ_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset/original/abstractive/{data_type}/summary"

    get_datapoints_parallel(15, extractive_doc_dir, extractive_summ_dir, abstractive_doc_dir, abstractive_summ_dir, data_type)
