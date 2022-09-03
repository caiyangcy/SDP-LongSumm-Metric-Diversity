import os.path
import json
import sys
import re
try:
    os.system("pip install rouge-score")
except:
    print("An exception occurred rouge-score")

import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer

from subprocess import call
import csv 


def impose_max_length(summary_text, max_tokens=600):
    #same tokenization as in rouge_score
    #https://github.com/google-research/google-research/blob/26a130831ee903cb97b7d04e71f227bbe24960b2/rouge/tokenize.py
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)

def evaluate_rouge(groudtruth, submission):
    metrics = ['rouge1', 'rouge2', 'rougeL']

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f":[], "rouge1_r":[], "rouge2_f":[], "rouge2_r":[], "rougeL_f":[], "rougeL_r":[]}

    results_avg = {}

    if len(submission) < len(groudtruth):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)

    for idx, ground_truth_summary in enumerate(groudtruth):
        submission_summary = submission[idx]

        submission_summary = impose_max_length(submission_summary)
        ground_truth_summary = impose_max_length(ground_truth_summary)

        scores = scorer.score(ground_truth_summary.strip(), submission_summary.strip())
        for metric in metrics:
            results[metric+"_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)

    return results_avg, results


scorer = BERTScorer(lang="en", rescale_with_baseline=False)
    
def evaluate(test_annotation_file, user_submission_file, e2e_file, spice_file, leaderboard_file, selection):

    with open(test_annotation_file) as f1:
        groudtruth = f1.readlines()
        groudtruth = [line for line in groudtruth if line != '\n']

    with open(user_submission_file) as f2:
        submission = f2.readlines()

    with open(spice_file) as f3:
        spice = f3.readline().strip()

    eval_results, results = evaluate_rouge(groudtruth, submission)

    values =  list( eval_results.values() )
    test_annotation_file = test_annotation_file.rsplit("/", 1)[-1]

    title = list(eval_results.keys())

    P, R, F1 = scorer.score(submission, groudtruth)

    title = title + ["BERTScore-P", "BERTScore-R", "BERTScore-F"] 
    values = values + [P.mean().item(), R.mean().item(), F1.mean().item()] 

    if e2e_file is not None:
        with open(e2e_file, "r") as f:
            metrics = f.readline().strip().split("\t")
            new_metrics = metrics + ["SPICE"] + title
            scores = f.readline().split("\t")
            new_scores = scores + [spice] + [ str( np.round(val, 4) ) for val in values ]


        if os.path.exists(leaderboard_file):
            with open(leaderboard_file, "a") as f:
                writer = csv.writer(f, delimiter=",")

                writer.writerow( [f"{selection}"] + new_scores )

        else:
            with open(leaderboard_file, "a") as f:
                writer = csv.writer(f, delimiter=",")

                writer.writerow( [" "] + new_metrics )

                writer.writerow( [f"{selection}"] + new_scores )

    print("DONE")

if __name__ == "__main__":

    for data_idx in range(10):

        print(f" ----- data_idx: {data_idx} ----- ")


        selection = "DGCNN"
        summ_type = "abstractive"
        subpath = "DGCNN_train_on_ext"

        base_path = f"../leaderboard_splits/split_{data_idx}/methods_leaderboard"
        reference_file = f"{base_path}/{subpath}/reference.txt"

        system_file = f"{base_path}/{subpath}/system.txt" 

        if not os.path.exists(f"{base_path}/{subpath}/eval"):
            os.makedirs(f"{base_path}/{subpath}/eval")

        e2e_save_file = f"{base_path}/{subpath}/eval/{selection}_{summ_type}_eval.txt"
        spice_save_file = f"{base_path}/{subpath}/eval/{selection}_{summ_type}_spice.txt"

        call(f"python3 ../e2e-metrics/measure_scores.py  {reference_file} {system_file} -o {e2e_save_file} ", shell=True)

        call(f"python3 ../coco-caption/pycocoevalcap/spice/spice.py  --reference {reference_file} --system {system_file} --savefile {spice_save_file}", shell=True)

        leaderboard_file = f"{base_path}/single_method.csv"
        evaluate(reference_file, system_file, e2e_save_file, spice_save_file, leaderboard_file, "DGCNN")
