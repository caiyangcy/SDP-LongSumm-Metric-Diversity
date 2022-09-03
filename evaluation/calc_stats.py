from collections import defaultdict
import numpy as np
import pandas as pd
import os
import csv

if __name__ == "__main__":


    stats_board = "../leaderboard_splits/stats/single_method_stats.csv"
    
    model_to_metric = defaultdict(list)
    headers = None

    for data_idx in range(10):

        base_path = f"../leaderboard_splits/split_{data_idx}/methods_leaderboard"

        leaderboard_file = f"{base_path}/single_method.csv"
        df = pd.read_csv(leaderboard_file)

        if headers is None:
            headers = df.columns[1:].tolist()

        for idx, row in df.iterrows():
            model = row[0]
            metrics = row[1:].values    

            model_to_metric[ model ].append( metrics.tolist() )


    model_to_stats = {} 

    for model, metrics in model_to_metric.items():

        metrics_arr = np.array( metrics )

        mean = np.mean(metrics_arr, axis=0)
        std = np.std(metrics_arr, axis=0)

        mean = np.round(mean, 4)
        std = np.round(std, 4)

        model_to_stats[f"{model}-Mean"] = [ str(val) for val in mean]
        model_to_stats[f"{model}-Std"] = [ str(val) for val in std]
    

    if os.path.exists(stats_board):
        with open(stats_board, "a") as f:
            writer = csv.writer(f, delimiter=",")

            for model, stats in model_to_stats.items():
                writer.writerow( [f"{model}"] + stats )

    else:
        with open(stats_board, "a") as f:
            writer = csv.writer(f, delimiter=",")

            writer.writerow( [" "] + headers )

            for model, stats in model_to_stats.items():
                writer.writerow( [f"{model}"] + stats )
