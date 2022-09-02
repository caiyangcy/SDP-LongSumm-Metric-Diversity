import enum
import json
import os
from re import T
import numpy as np
from transformers import RobertaTokenizer, TFRobertaModel, BertConfig
import random
import tensorflow as tf
from tqdm import tqdm
# import transformers
from nltk.tokenize import sent_tokenize
from nltk import tokenize

import tensorflow.keras.backend as K
import pickle, argparse

import pandas as pd 
from difflib import SequenceMatcher

import nltk
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def window_score(single_window_data: str, gt_rm_sw: list, metric='recall'):
    """compute all scores of a session with all ground truth """

    if metric == 'recall':
        win_rm_sw = word_token(single_window_data, "article")
        score = len(set(win_rm_sw) & set(gt_rm_sw)) / len(set(win_rm_sw))

    if metric == 'precision':
        win_rm_sw = word_token(single_window_data, "article")
        score = len(set(win_rm_sw) & set(gt_rm_sw)) / len(set(gt_rm_sw))

    return score


def word_token(raw_str, str_type):
    """token replacement"""
    if str_type == "article":
        sents = nltk.sent_tokenize(raw_str)
        words = []
        for sent in sents:
            words.extend(nltk.word_tokenize(sent))
    if str_type == "sent":
        words = nltk.word_tokenize(raw_str)
    if str_type == "no_stop":
        return nltk.word_tokenize(raw_str)

    return [w for w in words if w.lower() not in stop_words]


def replace_subs(text):
    text_replaced = text 
    for subs_tuple in substitutions:
        text_replaced = text_replaced.replace(subs_tuple[0], subs_tuple[1])

    return text_replaced

def revert_subs(text):
    text_replaced = text 
    for subs_tuple in substitutions:
        text_replaced = text_replaced.replace(subs_tuple[1], subs_tuple[0])

    return text_replaced

def prepare_doc_and_summ_extractive(args):
    doc_dir = args.doc_dir
    summ_dir = args.summ_dir

    document = []
    summary = []

    for subdir, dirs, files in os.walk(doc_dir):
        print('files: ', files)
        for file in files:

            doc_path = subdir + os.sep + file
            doc_path = doc_path#.replace('`', '').replace("'", '')

            doc_name = file.rsplit('.', 1)[0]
            
            with open(doc_path, 'r') as file:
                document.append( file.read().strip() )  
            
            summ_path = summ_dir + os.sep + doc_name + '.txt' # json file for summary
            if not os.path.isfile(summ_path):
                summ_path = summ_dir + os.sep + doc_name + ' .txt' # json file for summary with space - this is due some files contain a space

            summ = pd.read_csv(summ_path, sep='\t', header=None)
            
            summ_sents = summ.iloc[:,-1].values
            summary.append( list(summ_sents)  )
           
    print('Scanning done!')

    return document, summary


def prepare_doc_and_summ_abstractive(args):
    doc_dir = args.doc_dir
    summ_dir = args.summ_dir

    document = []
    summary = []

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
                summary.append( summ )
            
            doc_path = doc_dir + os.sep + doc_id + ".txt"
            doc_path = doc_path.replace('`', '').replace("'", '')

            try:
                with open(doc_path, 'r') as file:
                    document.append( file.read().strip() )  
            except Exception as e:
                # print(f"{summ_path} summary goes wrong, {e}")
                missing_docs.append(doc_path)

    print('Scanning done!')
    print(f"{len(missing_docs)} missing in total.") # should be 46

    return document, summary


def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        # print('before: ', x.shape)
        x = x[:length]
        # print('after: ', x.shape)
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        # print('before: ', x.shape)
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        # print('after: ', x.shape)
        outputs.append(x)
    return np.array(outputs)


def extractive_labeling(doc, summ):
    doc_replaced = doc 
    for subs_tuple in substitutions:
        doc_replaced = doc_replaced.replace(subs_tuple[0], subs_tuple[1])

    doc_replaced = replace_subs(doc)
    summ_replaced = [replace_subs(summ_sent) for summ_sent in summ]

    # summ_replaced = []
    # for summ_sent in summ:
    #     for subs_tuple in substitutions:
    #         summ_sent = summ_sent.replace(subs_tuple[0], subs_tuple[1])
    #     summ_replaced.append(summ_sent)

    doc = sent_tokenize(doc_replaced)
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




# def abstractive_labeling(doc, summ,):

#     # print("doc: ", doc)

#     doc_replaced = doc 
#     for subs_tuple in substitutions:
#         doc_replaced = doc_replaced.replace(subs_tuple[0], subs_tuple[1])

#     doc_replaced = replace_subs(doc)
#     summ_replaced = replace_subs(summ) 

#     split_doc = sent_tokenize(doc_replaced)
#     labels = [0]*len(split_doc)
#     split_summary = sent_tokenize(summ_replaced) 
        
#     gt_rm_sw = [ word_token(s, "sent") for s in split_summary ]

#     for summ_sent_words in gt_rm_sw:
#         scores = np.array( [ window_score(it, summ_sent_words, metric="recall") for it in split_doc ]  )

#         arg_ind = np.argsort(scores)[::-1]

#         for ind in arg_ind:
#             if not labels[ind]:
#                 labels[ind] = 1
#                 break

#     return labels



def abstractive_labeling(doc, true_summ):
    doc = tokenize.sent_tokenize(doc)
    # print(len(doc))
    summ = ''
    cur_score = 0
    labels = [ 0 for x in range(len(doc))]

    while True:
        for ind,x in enumerate(doc):
            if labels[ind]:
                 continue
            cur_summ = summ + ' ' + x
            score = scorer.score(cur_summ, true_summ)['rouge1'].fmeasure
            if score > cur_score :
                cur_score = score
                summ = cur_summ
                labels[ind] = 1
                break
        else:
            break
    return labels





def main(args, idx):

    pretrained_path = '../../replication/Longsumm_code/pretrained/roberta/'
    config_path = os.path.join(pretrained_path, 'config.json')
    
    tokenizer = RobertaTokenizer(vocab_file=pretrained_path + 'vocab.json',
                                 merges_file=pretrained_path + 'merges.txt',
                                 lowercase=True, add_prefix_space=True)


    config = BertConfig.from_json_file(config_path)
    config.output_hidden_states = True
    bert_model = TFRobertaModel.from_pretrained(pretrained_path, config=config)

    
    average_pooling = tf.keras.layers.GlobalAveragePooling1D()

    def conver_token(texts):
        token_ = []
        trunc_docs = []
        for text in tqdm(texts):

            text_replaced = replace_subs(text)
            text_sents = tokenize.sent_tokenize(text_replaced)

            if len(text_sents) > 400:
                text_sents = text_sents[:400]

            text_sents_reverted = [revert_subs(sent) for sent in text_sents]
            text_reverted = " ".join(text_sents_reverted)
            trunc_docs.append( text_sents_reverted )

            token = tokenizer(text_reverted, max_length=256,truncation=True,padding=True, return_tensors="tf")
            vecotor = bert_model(token)[0]
            # print('vecotor shape: ', vecotor.shape)
            pooling = average_pooling(vecotor, mask=token['attention_mask'])
            # print('pooling shape: ', pooling.shape)
            token_.append(pooling)
        return token_, trunc_docs

    if args.is_extractive:
        docus, summs = prepare_doc_and_summ_extractive(args)
    else:
        docus, summs = prepare_doc_and_summ_abstractive(args)

    print('docus length: ', len(docus) )
    print('summs length: ', len(summs) )


    ########### To speed up debugging, only use 2 documents ##############
    data_type = "avail" if "avail" in args.doc_dir else "test"

    if args.is_extractive:
        savepath = args.doc_dir.rsplit("/", 1)[0] + "/processed" 
        labels = []

        for idx, doc in tqdm(enumerate(docus)):
            summ = summs[idx]
            label = extractive_labeling(doc, summ)
            labels.append(label)

        with open(f"{savepath}/labels_{data_type}.pickle", 'wb') as handle:
            pickle.dump(labels, handle)
    else:
        savepath = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive_test_for_extractive_prediction/"

        labels = []

        for idx, doc in tqdm(enumerate(docus)):
            summ = summs[idx]
            label = abstractive_labeling(doc, summ)
            labels.append(label)

        with open(f"{savepath}/labels_{data_type}.pickle", 'wb') as handle:
            pickle.dump(labels, handle)


    data_type = "avail" if "avail" in args.doc_dir else "test"

    pooling, trunc_docs = conver_token(docus)
    print(len(pooling))
    b = sequence_padding(pooling, length=400)
    print(b.shape)

    if args.is_extractive:
        summs_merged = [  " ".join(s for s in summ) for summ in summs ] # summs is a list of lists of summaries; merge each sublists into a string.
    else:
        summs_merged = summs
    data_merged = [ [trunc_docs[idx], summs_merged[idx]] for idx in range(len(trunc_docs))]

    with open(f"{savepath}/data_{data_type}.pickle", 'wb') as handle:
        pickle.dump(data_merged, handle)

    import csv
    with open(f"{savepath}/data_{data_type}.txt", 'w') as handle:
        wr = csv.writer(handle)
        wr.writerows(data_merged)

    np.save(f'{savepath}/docs_roberta_{data_type}.npy', b)


if __name__=='__main__':
    # tf.compat.v1.disable_v2_behavior()
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()

    parser = argparse.ArgumentParser(
        description='Given URL of a paper, this script download the PDFs of the paper'
    )

    parser.add_argument('--doc_dir', default="../../datasets/dataset/LongSumm2021/extractive/test", help='link to the folder that contains the documents')
    parser.add_argument('--summ_dir', default="../../datasets/dataset/original/extractive/test/summary", help='link to the folder that contains the summaries')

    # parser.add_argument('--doc_dir', default="../../datasets/dataset/LongSumm2021/abstractive/avail", help='link to the folder that contains the documents')
    # parser.add_argument('--summ_dir', default="../../datasets/dataset/original/abstractive/avail/summary", help='link to the folder that contains the summaries')
    parser.add_argument('--is_extractive', default=True, type=bool) 

    args = parser.parse_args()

    for idx in range(10):

        print(f' ----- {idx} ----- ')

        args.is_extractive = False
        args.doc_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive/avail"
        args.summ_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/avail/summary/"
        main(args, idx)  

        args.is_extractive = False
        args.doc_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive/test"
        args.summ_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/test/summary/"
        main(args, idx)  

        # args.is_extractive = True
        # args.doc_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/avail"
        # args.summ_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/original/extractive/avail/summary"
        # main(args, idx)  

        # args.is_extractive = True
        # args.doc_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/extractive/test"
        # args.summ_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/original/extractive/test/summary"
        # main(args, idx)  