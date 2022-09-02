# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: divide scientific documents into session piece dataset for training


import nltk
import json
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
import pickle, random
from prepare_data import prepare_doc_and_summ_extrative, prepare_doc_and_summ_abstrative
import argparse
import tensorflow as tf
import os 

stop_words = stopwords.words('english')

print("length of stopwords: ", len(stop_words))
# print(stop_words[:20])

window_size = 1024
buffer = 128
decode_max_len = 220
split = 30

print("window_size: ", window_size)
print("buffer: ", buffer)
print("decode_max_len: ", decode_max_len)



def window_score_pegasus(single_window_data: str, gt_emb, tokenizer, model):
    sents = nltk.sent_tokenize(single_window_data)
    batch_sent = tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
    output = model(batch_sent['input_ids'], batch_sent['attention_mask'], decoder_input_ids=batch_sent['input_ids'])
    encode_sent = output.encoder_last_hidden_state
    sent_emb = torch.sum(encode_sent, dim=1)

    cos = nn.CosineSimilarity()

    scores = [torch.mean(cos(sent_emb, g.reshape(1, -1))) for g in gt_emb]
    scores = torch.hstack(scores)
    return scores


def slide_window(raw_data: str, mode='window'):
    """
    split raw data into piece of session

    Args:
        raw_data: str, raw text of a paper
        mode: str, not used
    Returns:
    split_data: list, session data

    """
    spilt_data = []
    words = nltk.word_tokenize(raw_data)
    if len(words) - window_size - buffer < 0:
        spilt_data.append(raw_data)
    else:
        for i in range(0, len(words), window_size):
            spilt_data.append(" ".join(words[max(0, i - buffer):min(i + window_size, len(words))]))

    return spilt_data


def window_score(single_window_data: str, gt_rm_sw: list, metric='recall'):
    """compute all scores of a session with all ground truth """
    if metric == 'recall':
        win_rm_sw = word_token(single_window_data, "article")
        score = [len(set(win_rm_sw) & set(it)) / len(set(it)) for it in gt_rm_sw]

    if metric == 'precision':
        win_rm_sw = word_token(single_window_data, "article")
        score = [len(set(win_rm_sw) & set(it)) / len(set(win_rm_sw)) for it in gt_rm_sw]

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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def save_pickle(d, s, summ_idx, f):
    feat = {"document": d, "summary": s, "summary_index": summ_idx}

    with open(f, 'wb') as handle:
        pickle.dump(feat, handle)

def save_record(d, s, f):
    """save tf record type data"""
    
    with tf.io.TFRecordWriter(f) as writer:
        for idx, doc in enumerate(d):
            summ = s[idx]

            example = tf.train.Example(features=tf.train.Features(feature={
                "document": _bytes_feature(serialize_array(doc)),
                "summary": _bytes_feature(serialize_array(summ)),
            }))

            writer.write(example.SerializeToString())


def session_rank(document, summary, out_file, mode='more'):
    """
    split document into session piece and match each session with its ground truth

    Args:
        document: list, document list
        summary: list, ground truth list
        out_file: str, output file path
        mode: str, match method 'more' means matching as much labels as possible to each session

    Returns:
    tf record type dataset
    """
    split_doc = list(map(slide_window, document))
    print("split_doc length: ", len(split_doc))
    split_summary = [nltk.sent_tokenize(it) for it in summary]

    features, labels = [], []
    summ_idx = []

    for i in tqdm(range(len(split_doc))): # iterating through each document
        
        gt_rm_sw = [word_token(s, "sent") for s in split_summary[i]]
        if mode == 'less':
            scores = np.array([window_score(it, gt_rm_sw, metric="recall") for it in split_doc[i]])

            ind = np.argmax(scores, axis=0)
            summ = [[] for it in split_doc[i]]
            for j in range(len(split_summary[i])):
                summ[ind[j]].append(split_summary[i][j])

            for j in range(len(split_doc[i])):
                if not summ[j]:
                    continue
                features.append(split_doc[i][j])
                labels.append(" ".join(summ[j]))
                summ_idx.append(i)

        if mode == 'more':
            np_s = np.array(split_summary[i])
            scores = np.array([window_score(it, gt_rm_sw, metric="recall") for it in split_doc[i]])
            d1, d2 = scores.shape
            for j in range(d1):
                s_len = 0
                for _ in range(d2):
                    if s_len > decode_max_len:
                        break
                    r = np.argmax(scores[j])
                    s_len += len(word_token(split_summary[i][r], 'no_stop'))
                    scores[j][r] = -1
            mask = (scores < 0)

            for j in range(d1):
                summ = " ".join(np_s[mask[j]])
                features.append(split_doc[i][j])
                labels.append(summ)
                summ_idx.append(i)

    save_record(features, labels, out_file+".tfrecord")
    save_pickle(features, labels, summ_idx, out_file+".pickle")

    print("num examples: ", len(features))
    print("write into %s" % out_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Given URL of a paper, this script download the PDFs of the paper'
    )

    # python3 utils/session_rank2.py --doc_dir="../../datasets/dataset/LongSumm2021/extractive/avail" --summ_dir="../../datasets/dataset/original/extractive/avail/summary/" --is_extractive
    # python3 utils/session_rank2.py --doc_dir="../../datasets/dataset/LongSumm2021/abstractive/avail" --summ_dir="../../datasets/dataset/original/abstractive/avail/summary/"

    # python3 utils/session_rank2.py --doc_dir="/home/caiyang/Desktop/debugging_data/LongSumm-2021/extractive/avail" --summ_dir="../../datasets/dataset/original/extractive/avail/summary/" --is_extractive
    # python3 utils/session_rank2.py --doc_dir="/home/caiyang/Desktop/debugging_data/LongSumm-2021/abstractive/avail" --summ_dir="../../datasets/dataset/original/abstractive/avail/summary/"

    # parser.add_argument('--doc_dir', help='link to the folder that contains the documents')
    # parser.add_argument('--summ_dir', help='link to the folder that contains the summaries')
    # parser.add_argument('--is_extractive', action="store_true", help='whether preprocessing on extractive docs')

    args = parser.parse_args()

    for idx in range(10):

        doc_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/LongSumm2021/abstractive/avail"
        summ_dir = f"../../datasets/dataset_multiple_splits/split_{idx}/original/abstractive/avail/summary/"

        test_doc_path = doc_dir.replace("avail", "test")
        test_summ_path = summ_dir.replace("avail", "test")

        print("test_doc_path: ", test_doc_path)
        print("test_summ_path: ", test_summ_path)

        document, summary = prepare_doc_and_summ_abstrative(doc_dir, summ_dir)

        test_document, test_summary = prepare_doc_and_summ_abstrative(test_doc_path, test_summ_path)

        print(document[0])

        break

        # savepath = doc_dir.rsplit("/", 1)[0]
        
        # if not os.path.exists(f'{savepath}/processed/'):
        #     os.makedirs(f'{savepath}/processed/')

        # param_file = f'{savepath}/processed/param.json'
        
        # with open (param_file, 'w') as f:   
        #     json.dump({'window_size': window_size, 'decode_max_len': decode_max_len, 'buffer': buffer}, f)


        # train_x, train_y = document[split:], summary[split:]
        # val_x, val_y = document[:split], summary[:split]

        # train_file = f'{savepath}/processed/train'
        # val_file = f'{savepath}/processed/eval'

        # session_rank(train_x, train_y, train_file)
        # session_rank(val_x, val_y, val_file)


        # test_file = f'{savepath}/processed/test'
        # print("test_document length: ", len(test_document))
        # session_rank(test_document, test_summary, test_file)

        # pred = []
        # for i in range(len(test_document)):
        #     pred.append({'document':test_document[i], "summary": test_summary[i]})

        # pred_json = f'{savepath}/processed/test.json'
        # with open(pred_json, 'w') as f:
        #     json.dump(pred, f)
