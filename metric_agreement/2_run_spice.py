from subprocess import call

for data_idx in range(10):

    for model_name in ["Bart", "BERTSum", "Bigbird-Pegasus", "DGCNN", "SummaRuNNer"]:

        print(f'------ {data_idx} {model_name} ------')

        savepath = f"../leaderboard_splits/metric_agreement/spice_prep/{model_name}"

        reference_file = f"{savepath}/reference_{data_idx}.txt"
        system_file = f"{savepath}/system_{data_idx}.txt" 

        spice_save_file = f"{savepath}/{model_name}_spice_{data_idx}.txt"

        call(f"python3 ../coco-caption/pycocoevalcap/spice/spice_list.py  --reference {reference_file} --system {system_file} --savefile {spice_save_file}", shell=True)
