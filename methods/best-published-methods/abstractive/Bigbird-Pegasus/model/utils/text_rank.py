# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: implement of the text rank algorithm



import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

stop_words = stopwords.words('english')
word_embeddings = {}
emb_dim = 200
steps = 100
damping = 0.85
min_diff = 1e-5


def gen_embedding(path):
    """return word embedding from path file"""
    word_emb = {}
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word_emb[values[0]] = np.asarray(values[1:], dtype='float32')
    return word_emb


def remove_stopwords(s):
    return ' '.join([i for i in s if i not in stop_words])


def page_rank(similarity_matrix):
    """page rank algorithm"""
    cur_vector = np.array([1] * len(similarity_matrix))

    pre = 0
    for epoch in range(steps):
        cur_vector = (1 - damping) + damping * np.matmul(similarity_matrix, cur_vector)
        if abs(pre - sum(cur_vector)) < min_diff:
            break
        else:
            pre = sum(cur_vector)

    return cur_vector


def rank_scores(sentences, word_embeddings):
    """
    text rank algorithm implement by page rank
    Args:
        sentences: list, sentences for rank
        word_embeddings: dict, word embedding default: glove
    Returns:
        sentences_rank: np.array, sentences score

    """

    clean_sentences = pd.Series(sentences).str.replace('[^a-zA-Z]', ' ')

    clean_sentences = [remove_stopwords(s.lower().split()) for s in clean_sentences]

    sentences_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((emb_dim,))) for w in i.split()]) / (len(i.split()) + 1e-2)
        else:
            v = np.zeros((emb_dim,))
        sentences_vectors.append(v)

    similarity_matrix = np.zeros((len(clean_sentences), len(clean_sentences)))

    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(
                    sentences_vectors[i].reshape(1, -1), sentences_vectors[j].reshape(1, -1)
                )

    similarity_matrix = similarity_matrix + similarity_matrix.T - np.diag(similarity_matrix.diagonal())

    norm = np.sum(similarity_matrix, axis=0)
    norm_similarity_matrix = np.divide(similarity_matrix, norm, where=norm != 0)

    sentences_rank = page_rank(norm_similarity_matrix)

    return sentences_rank

