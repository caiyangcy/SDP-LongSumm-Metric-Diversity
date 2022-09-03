# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: ensemble method for summarization


import re
import nltk
from summa import summarizer
from summa import keywords
import json
import numpy as np
from nltk.corpus import stopwords, wordnet
from text_rank import rank_scores, gen_embedding
from tqdm import tqdm

stop_words = stopwords.words('english')


def drop_sent(raw_s: str):
    """drop redundancy"""
    sentences = nltk.sent_tokenize(raw_s)
    d_sentences = []
    [d_sentences.append(it) for it in sentences if it not in d_sentences]

    return " ".join(d_sentences)


def add_summ(text):
    """auto summarize"""
    res = summarizer.summarize(text, ratio=0.8)
    return res


def sent_sim(s, t, mode='jaccard', use_stopwords=True):
    """sentences similarity """
    ws = nltk.word_tokenize(s)
    wt = nltk.word_tokenize(t)
    if use_stopwords:
        ws = [i for i in ws if i.lower() not in stop_words]
        wt = [i for i in wt if i.lower() not in stop_words]

    if mode == 'jaccard':
        ws = set(ws)
        wt = set(wt)
        return len(ws & wt) / len(ws | wt)


def self_clip(raw_str: str, r=0.8):
    """clip same sentences"""
    sents = nltk.sent_tokenize(raw_str)
    sents = np.array(sents)
    l = len(sents)
    mask = np.ones(l, dtype=bool)
    for i in range(l):
        if not mask[i]:
            continue
        for j in range(i+1, l):
            if sent_sim(sents[i],sents[j], mode='jaccard') >= r:
                mask[j] = False

    selected_sents = sents[mask]
    clip_str = " ".join(selected_sents)
    return clip_str


def summary_merge(d1, d2, out_file, r=0.5, method='recall'):
    """ensemble the output of abstractive model and the extractive model"""

    embedding_path = "REPLACE-BY-YOUR-PATH/Longsumm_code/glove.6B.200d.txt"
    emb = gen_embedding(embedding_path)
    res = []
    for k in tqdm(range(len(d1))):
        if method == 'raw':
            m_s = self_clip(" ".join([d1[k], d2[k]]), r)
        if method == 'recall':
            s = self_clip(" ".join([d1[k], d2[k]]), r)
            sents = nltk.sent_tokenize(s)

            scores = rank_scores(sents, emb)

            sents = np.array(sents)
            mask = (scores > 1)
            m_s = " ".join(sents[mask])

        if method == 'text_rank':
            m_s = add_summ(" ".join([d1[k], d2[k]]))

        m_s = m_s.replace("\n", " ")

        res.append(m_s)

    with open(out_file, 'w') as f:
        res_to_write = "\n".join(r.strip() for r in res)
        f.write(res_to_write)


def join_words(s: str):
    with open("../dataset/words.txt", 'r', encoding='utf-8') as f:
        txt = f.read()
    words_list = txt.split('\n')
    sents = nltk.sent_tokenize(s)
    words = []

    def is_word(w):
        return w in words_list
    for sent in sents:
        words.extend(nltk.word_tokenize(sent))
    res = ""
    for i in range(len(words)):
        if i != len(words)-1 and not is_word(words[i]) and not is_word(words[i+1]) and is_word(words[i]+words[i+1]):
            print("merge words: {}, {}".format(words[i], words[i+1]))
            res += " " + words[i] + words[i+1]
            i += 1
        else:
            res += " " + words[i]
    return res


if __name__ == "__main__":

    extractive_path = "REPLACE-BY-YOUR-PATH/leaderboard/baseline/DGCNN/abstractive_trunc/system_20.txt"
    abstractive_path = "REPLACE-BY-YOUR-PATH/leaderboard/baseline/Bigbird-Pegasus/abstractive/system_80000.txt"

    # re-order the two summaries first, based on the reference set

    dgcnn_ref_path = "REPLACE-BY-YOUR-PATH/leaderboard/baseline/DGCNN/abstractive_trunc/reference_20.txt"
    pegasus_ref_path = "REPLACE-BY-YOUR-PATH/leaderboard/baseline/Bigbird-Pegasus/abstractive/reference.txt"

    with open(dgcnn_ref_path) as f:
        ref_1 = f.readlines()
        ref_1 = [line.strip() for line in ref_1 if line.strip() != ""]

    with open(pegasus_ref_path) as f:
        ref_2 = f.readlines()
        ref_2 = [line.strip() for line in ref_2 if line.strip() != ""]

    dgcnn_system_reorder = []
    for idx, ref in enumerate(ref_2):
        try:
            index = ref_1.index(ref)
        except Exception as e:
            print(f"{idx+1} in pegasus summary raise an error")
            print(f"Error: {e}")
            assert False
        dgcnn_system_reorder.append(index)

    with open(extractive_path) as f:
        dgcnn_sys = f.readlines()

    with open(abstractive_path) as f:
        pegasus_sys = f.readlines()

    dgcnn_sys = [dgcnn_sys[i] for i in dgcnn_system_reorder]

    method = "text_rank"
    out_file = f"REPLACE-BY-YOUR-PATH/leaderboard/baseline/LongSumm_merged/abstractive/system_merged_{method}_ext_first.txt"

    summary_merge(dgcnn_sys, pegasus_sys, out_file, method=method)