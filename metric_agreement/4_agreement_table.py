import pickle
import numpy as np
from collections import defaultdict


savepath = "../leaderboard_splits/metric_agreement/stats/spice.pickle"
with open(savepath, "rb") as f:
    spice_top_agreement, spice_bottom_agreement = pickle.load(f)

savepath = "../leaderboard_splits/metric_agreement/stats/rouge.pickle"
with open(savepath, "rb") as f:
    rouge_top_agreement, rouge_bottom_agreement = pickle.load(f)


savepath = "../leaderboard_splits/metric_agreement/stats/bleu.pickle"
with open(savepath, "rb") as f:
    bleu_top_agreement, bleu_bottom_agreement = pickle.load(f)


savepath = "../leaderboard_splits/metric_agreement/stats/bertscore.pickle"
with open(savepath, "rb") as f:
    bertscore_top_agreement, bertscore_bottom_agreement = pickle.load(f)


entry_labels = [ 'dgcnn', 'bigbird', 'summa', 'bertsum', 'bart' ]
metrics = ['ROUGE', 'BLEU', 'BERTSCORE', 'SPICE']

for idx, label in enumerate(entry_labels):

    top_agreement = defaultdict(list)
    bottom_agreement = defaultdict(list)

    for data_partition_idx in range(10):

        model_rouge_top5 = rouge_top_agreement[data_partition_idx][idx]
        model_bleu_top5 = bleu_top_agreement[data_partition_idx][idx]
        model_bertscore_top5 = bertscore_top_agreement[data_partition_idx][idx]
        model_spice_top5 = spice_top_agreement[data_partition_idx][idx]

        tops = [model_rouge_top5, model_bleu_top5, model_bertscore_top5, model_spice_top5]

        for top_i, metric_i_top in enumerate(tops):
            for top_j, metric_j_top in enumerate(tops):

                if top_j <= top_i:
                    continue

                metrc_i, metric_j = metrics[top_i], metrics[top_j]


                top_agreement[(metrc_i, metric_j)] += [ len( set(metric_i_top)&set(metric_j_top) ) ]


        model_rouge_bottom5 = rouge_bottom_agreement[data_partition_idx][idx]
        model_bleu_bottom5 = bleu_bottom_agreement[data_partition_idx][idx]
        model_bertscore_bottom5 = bertscore_bottom_agreement[data_partition_idx][idx]
        model_spice_bottom5 = spice_bottom_agreement[data_partition_idx][idx]

        bottoms = [model_rouge_bottom5, model_bleu_bottom5, model_bertscore_bottom5, model_spice_bottom5]

        for bottom_i, metric_i_btm in enumerate(bottoms):
            for bottom_j, metric_j_btm in enumerate(bottoms):

                if bottom_j <= bottom_i:
                    continue

                metrc_i, metric_j = metrics[bottom_i], metrics[bottom_j]

                bottom_agreement[(metrc_i, metric_j)] += [ len( set(metric_i_btm)&set(metric_j_btm) ) ]


    if label not in ['bart', 'bertsum', 'summa']:
        continue

    for metric_pair in top_agreement.keys():
        agreement = top_agreement[metric_pair]
        disagreement = bottom_agreement[metric_pair]

        print(f"{label}; {metric_pair}; Average agreement: { np.round(np.mean(agreement)/5, 2) } std: {np.round(np.std(agreement)/5, 2)}; Average agreement: { np.round(np.mean(disagreement)/5, 2) } std: {np.round(np.std(disagreement)/5, 2)};")

    print()
