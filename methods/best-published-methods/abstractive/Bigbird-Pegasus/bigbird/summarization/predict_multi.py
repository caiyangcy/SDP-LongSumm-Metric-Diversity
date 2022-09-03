# @Author: yingsenci
# @Time: 2021/03/30
# @Contact: scying@zju.edu.com,
# @Description: predict summary for scientific documents

import os
import sys

import tensorflow.compat.v2 as tf
import numpy as np
import random 
from collections import defaultdict

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


sys.path.append('REPLACE-BY-YOUR-PATH/Longsumm_code/')


from bigbird.core import flags
from bigbird.core import modeling
from bigbird.summarization import run_summarization

# import tensorflow as tf
from tensorflow.python.ops.variable_scope import EagerVariableStore
import tensorflow_text as tft
from tqdm import tqdm
import sys
import json
from rouge import Rouge
import nltk
import re

rouge = Rouge()


FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"):
    flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)
# tf.enable_v2_behavior()


FLAGS.max_encoder_length = 1024
FLAGS.max_decoder_length = 128
FLAGS.vocab_model_file = "REPLACE-BY-YOUR-PATH/Longsumm_code/bigbird/vocab/pegasus.model"
FLAGS.eval_batch_size = 4
FLAGS.substitute_newline = "<n>"


ckpt_path = "REPLACE-BY-YOUR-PATH/Longsumm_code/pretrained/summarization_arxiv_pegasus_model.ckpt-300000"

# pred_in = 'REPLACE-BY-YOUR-PATH/datasets/dataset/LongSumm2021/abstractive/processed/test.json'

tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(FLAGS.vocab_model_file, "rb").read())

num_pred_steps = 5


window_size = 1024
buffer = 128
decode_max_len = 220
split = 30


def slide_window(raw_data: str, mode='window'):
    spilt_data = []
    words = nltk.word_tokenize(raw_data)
    if len(words) - window_size - buffer < 0:
        spilt_data.append(raw_data)
    else:
        for i in range(0, len(words), window_size):
            spilt_data.append(" ".join(words[max(0, i - buffer):min(i + window_size, len(words))]))

    return spilt_data


def rouge_metric(source, target):
    #source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1-f': scores[0]['rouge-1']['f'],
            'rouge-2-f': scores[0]['rouge-2']['f'],
            'rouge-l-f': scores[0]['rouge-l']['f'],
            'rouge-1-r': scores[0]['rouge-1']['r'],
            'rouge-2-r': scores[0]['rouge-2']['r'],
            'rouge-l-r': scores[0]['rouge-l']['r'],
        }
    except ValueError:
        return {
            'rouge-1-f': 0.0,
            'rouge-2-f': 0.0,
            'rouge-l-f': 0.0,
            'rouge-1-r': 0.0,
            'rouge-2-r': 0.0,
            'rouge-l-r': 0.0,
        }


def input_fn(document):

    def _tokenize_example(doc):
        if FLAGS.substitute_newline:
            doc = tf.strings.regex_replace(doc, "\n", FLAGS.substitute_newline)
        doc = tf.strings.regex_replace(doc, r" ([<\[]\S+[>\]])", b"\\1")
        document_ids = tokenizer.tokenize(doc)
        if isinstance(document_ids, tf.RaggedTensor):
            dim = document_ids.shape[0]
            document_ids = document_ids.to_tensor(0, shape=(dim, FLAGS.max_encoder_length))
        else:
            document_ids = document_ids[:, :FLAGS.max_encoder_length]

        return document_ids

    feats = slide_window(document)
    d = _tokenize_example(feats)

    return d


def main(dataset_curr_idx):
    transformer_config = flags.as_dictionary()
    container = EagerVariableStore()
    with container.as_default():
        model = modeling.TransformerModel(transformer_config)


    pred_in = f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{dataset_curr_idx}/LongSumm2021/abstractive/processed/test.json'
    with open(pred_in, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        truth_test = [d['summary'].strip() for d in dataset]

    import pickle
    # pred_in = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{dataset_curr_idx}/LongSumm2021/abstractive/processed/test.pickle"

    pred_in = f'REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{dataset_curr_idx}/Session/processed-1024/abstractive_test.pickle'

    with open(pred_in, 'rb') as f:
        tmp = pickle.load(f)
        docs = list(  tmp['document']  )
        summ = list(  tmp['summary']  )
        summary_index = list( tmp["summary_index"] )

        dataset = [ {"document": docs[idx], "summary": summ[idx]}  for idx in range(len(docs)) ]

        reference_summaries = tmp['correct_summary']

    print('len(dataset: ', len(dataset) )
    summ_idx_to_doc = defaultdict(list)

    for idx_, summ_idx_ in enumerate(summary_index):
        summ_idx_to_doc[summ_idx_].append(docs[idx_])



    @tf.function(experimental_compile=True)
    def fwd_only(features):
        (llh, logits, pred_ids), _ = model(features, target_ids=None, training=False)
        return llh, logits, pred_ids

    ex = input_fn("this is a test")
    with container.as_default():
        tmp = tf.reshape(ex[0], shape=(1,-1))
        llh, logits, pred_ids = fwd_only(tmp)

        pred_sents = tokenizer.detokenize(pred_ids)

        pred_sents = tf.strings.regex_replace(pred_sents, r"([<\[]\S+[>\]])", b" \\1")

        pred_sents = tf.strings.regex_replace(pred_sents, transformer_config["substitute_newline"], "\n")

        pred_summary = " ".join([s.numpy().decode('utf-8') for s in pred_sents])

        print("\npred_sents: ", pred_sents)
        print("\npred_summary: ", pred_summary)

    print('==== build model')

    ckpt_reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
    loaded_weights = []

    for v in tqdm(model.trainable_weights, position=0):
        try:
            val = ckpt_reader.get_tensor(v.name[:-2])
        except:
            val = v.numpy()
        loaded_weights.append(val)
    model.set_weights(loaded_weights)
    print("==== load model weights")

    s1, s2 = [], []

    log_likelihoods = []
    logits = []

    index_to_system = {idx: [] for idx in range(max(summary_index)+1)}
    # print(index_to_system)
    for i, ex in tqdm(enumerate(dataset )):
        # print("### Example %d" % i)
        document, summary = ex['document'], ex['summary']

        document = "What is stopping us from applying meta-learning to new tasks? Where do the tasks come from? Designing task distribution is laborious. We should automatically learn tasks! Unsupervised Learning via Meta-Learning: The idea is to use a distance metric in an out-of-the-box unsupervised embedding space created by BiGAN/ALI or DeepCluster to construct tasks in an unsupervised way. If you cluster points to randomly define classes (e.g. random k-means) you can then sample tasks of 2 or 3 classes and use them to train a model. Where does the extra information come from? The metric space used for k-means asserts specific distances. The intuition why this works is that it is useful model initialization for downstream tasks. This summary was written with the help of Chelsea Finn."
        doc_ids = input_fn(document)
        # print("pred tensor shape: ", doc_ids.shape)

        llh, logit, pred_ids = fwd_only(doc_ids) # logits: output from softmax layer

        log_likelihoods.append(llh.numpy().mean())
        logits.append(logit.numpy().mean())

        del llh, logit
        pred_sents = tokenizer.detokenize(pred_ids)
        del pred_ids
        pred_sents = tf.strings.regex_replace(pred_sents, r"([<\[]\S+[>\]])", b" \\1")
        if transformer_config["substitute_newline"]:
            pred_sents = tf.strings.regex_replace(pred_sents, transformer_config["substitute_newline"], "\n")

        pred_summary = " ".join([s.numpy().decode('utf-8') for s in pred_sents])

        # print("pred_sents: ", pred_sents)
        # print("pred_summary: ", pred_summary)

        index_to_system[ summary_index[i] ].append(pred_summary)


    all_summaries_full = []
    all_summaries_trunc = []
    all_summaries_session_len_even = []

    for idx, session_pred in index_to_system.items():
        pred_summary = " ".join(session_pred)
        pred_summary = pred_summary.replace("\n", " ").replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" ?", "?").replace(" !", "!")
        pred_summary = re.sub(r" +", " ", pred_summary)

        pred_sents = nltk.tokenize.sent_tokenize(pred_summary)
        pred_summary = ""
        for sent in pred_sents:
            if len(pred_summary.split()) + len(sent.split()) <= 600:
                pred_summary += " " + sent
    
        all_summaries_trunc.append(pred_summary)

        s1.append(pred_summary), s2.append(summary)

        all_summaries_full.append( " ".join(pred_sents).strip() )


        section_even_length = np.round( 600/len(session_pred) )
        summ_sec_even_len = ""

        for sess_pred_texts in session_pred:

            pred_sents = nltk.tokenize.sent_tokenize(sess_pred_texts)
            pred_sents = [sent.replace("\n", " ").replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" ?", "?").replace(" !", "!").strip() for sent in pred_sents]
            pred_sents = [re.sub(r" +", " ", sent) for sent in pred_sents]


            section_text = ""
            for sent in pred_sents:
                if len(section_text.split()) + len(sent.split()) <= section_even_length:
                    section_text += " " + sent
                else:
                    break

            summ_sec_even_len += " " + section_text.strip()

        all_summaries_session_len_even.append(summ_sec_even_len.strip())





    res = rouge_metric(s1, s2)
    avg_rouge = "average rouge score: \n"
    for key in sorted(res.keys()):
        avg_rouge += "%s = %.4f\n" % (key, res[key])
    print("avg_rouge: ", avg_rouge)


    if not os.path.exists(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{dataset_curr_idx}/baseline/Session_based/"):
        os.mkdir(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{dataset_curr_idx}/baseline/Session_based/")


    if not os.path.exists(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{dataset_curr_idx}/baseline/Session_based/Bigbird-Pegasus-1024"):
        os.mkdir(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{dataset_curr_idx}/baseline/Session_based/Bigbird-Pegasus-1024")

    base_path = f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{dataset_curr_idx}/baseline/Session_based/Bigbird-Pegasus-1024"
    
    with open(f'{base_path}/system_trunc.txt', 'w') as f:
        pred_test_to_write = "\n".join(all_summaries_trunc)
        f.write(pred_test_to_write)

    with open(f'{base_path}/system_full.txt', 'w') as f:
        pred_test_to_write = "\n".join(all_summaries_full)
        f.write(pred_test_to_write)


    with open(f"{base_path}/system_even.txt", "w") as f:
        pred_to_write = "\n".join( all_summaries_session_len_even )
        f.write(  pred_to_write   )


    with open(f'{base_path}/reference.txt', 'w') as f:
        truth_test_to_write = "\n\n".join(reference_summaries)
        f.write(truth_test_to_write)


if __name__=="__main__":

    for idx in range(1, 10):
        print(f"idx: ----- {idx} -----")
        main(idx)