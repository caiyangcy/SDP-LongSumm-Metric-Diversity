from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import argparse
import tempfile
from nltk.tokenize import sent_tokenize

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric 
    """

    def float_convert(self, obj):
        try:
          return float(obj)
        except:
          return np.nan

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, mode='w')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir=os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
          os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx36G', SPICE_JAR, in_file.name,
          # '-cache', cache_dir,
          '-out', out_file.name,
          '-subset',
          '-silent'
        ]
        subprocess.check_call(spice_cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:    
          results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
        max_score = np.max(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
          # Convert none to NaN before saving scores over subcategories
          score_set = {}
          for category,score_tuple in imgId_to_scores[image_id].items():
            score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
          scores.append(score_set)
        return max_score, scores

    def method(self):
        return "SPICE"




if __name__=="__main__":

  spice = Spice()

  parser = argparse.ArgumentParser()

  parser.add_argument('--reference', help='link to the text file containing the reference summary')
  parser.add_argument('--system', help='link to the text file containing the system summary')
  parser.add_argument('--savefile', help='link to the text file saving the result')

  args = parser.parse_args()

  ref_path = args.reference
  sys_path = args.system


  # ref_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/reference_DGCNN.txt"
  # sys_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/system_DGCNN.txt"
  # args.savefile = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/DGCNN_res_sentence_one_by_one.txt"

  # ref_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/reference_SummaFormer.txt"
  # sys_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/system_SummaFormer.txt"
  # args.savefile = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/SummaFormer_res_sentence_one_by_one.txt"


  ref_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/reference_bart-large.txt"
  sys_path = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/system_bart-large.txt"
  args.savefile = "REPLACE-BY-YOUR-PATH/leaderboard_splits/SPICE_compare/Bart_res_sentence_one_by_one.txt"

  with open(ref_path) as f1:
    groudtruth = f1.readlines()
    groudtruth = [line for line in groudtruth if line != '\n']

  with open(sys_path) as f2:
    submission = f2.readlines()
  

  # gts = {k: [summ] for k, summ in enumerate(groudtruth)}
  # res = {k: [summ] for k, summ in enumerate(submission)}

  # gts = {1: ["This is a test"], 2: ["I am happy"] } # groudtruth

  # res = {1: ["Irrelavant text"], 2: ["I am not happy"] } # system

  # submission = [ 'This is the first submission. This is the second sentence from the first submission', 'This is some what irrelevant. This is a cat. This is another cat. This is the first submission.'  ]
  # groudtruth = [ 'This is the second submission. This is the also second sentence from the second submission', 'This is the first submission. This is the second sentence from the first submission',  ]

  scores = []

  for idx, system_generated in enumerate(submission):

    system_sents = sent_tokenize(system_generated)

    print(f"\n---------- SPICE IDX: {idx}: {len(system_sents)} ----------\n ")

    reference_sents = sent_tokenize( groudtruth[idx] )
    
    system_sents = [sent.strip() for sent in system_sents]
    reference_sents = [sent.strip() for sent in reference_sents]


    gts = { i : reference_sents for i in range(len(system_sents)) }
    res = { i : [ sys_sent ] for i, sys_sent in enumerate(system_sents) }

    max_score, _ = spice.compute_score(gts, res)

    scores.append( np.mean(max_score) )

    print(f"\n---------- SPICE IDX: {idx} DONE ----------\n ")


  # print('scores: ', scores)

  scores = np.round(scores, 4)
  scores = list( map( str, scores ) )
  scores = ", ".join(scores)

  with open(args.savefile, "w") as f:
      f.write(f"{scores}")