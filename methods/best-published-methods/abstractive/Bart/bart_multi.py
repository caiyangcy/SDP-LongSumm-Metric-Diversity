from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tqdm import tqdm 
from rouge_score import rouge_scorer

import argparse
import json
import pickle
import torch 
import os 
import numpy as np
import gc
import random
import nltk


def read_doc_summ(doc_path, summ_path):
    with open(doc_path) as f:
        document = f.readlines()
    with open(summ_path) as f:
        summary = f.readlines()

    document = [doc.strip() for doc in document if doc != "\n"]
    summary = [summ.strip() for summ in summary if summ != "\n"]

    # if "test" in summ_path:
    #     print("test: ", summary)
    return document, summary



def prepare_doc_and_summ_abstrative(doc_dir, summ_dir):
    # doc_dir = args.doc_dir
    # summ_dir = args.summ_dir

    document = []
    summary = []

    missing_docs = [] # note since we did not download all the pdfs then there will be missing documents

    for subdir, dirs, files in os.walk(summ_dir):

        # print("doc_dir: ", doc_dir)

        for file in files:
            
            summ_path = subdir + os.sep + file

            summ_path = summ_path.replace('`', '').replace("'", '')

            # print(summ_path)

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


def tokenize_data(document, summary, tokenizer):

    tokenized_inputs = tokenizer(document, return_tensors='pt', padding=True, truncation=True) #max_length=128, return_tensors='pt', padding=True, truncation=True)
    tokenized_summ   = tokenizer(summary, return_tensors='pt', padding=True, truncation=True) #max_length=32, return_tensors='pt', padding=True, truncation=True)


    return tokenized_inputs, tokenized_summ


def train_and_evaluation(train_data, eval_data, tokenizer, model_name, save_path, data_idx):

    train_on_gpu=torch.cuda.is_available()
    device = torch.device('cuda' if train_on_gpu else 'cpu')
    
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    train_inputs, train_summ = train_data
    eval_inputs, eval_summ = eval_data

    dataset_train = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_summ['input_ids'])
    dataset_val = TensorDataset(eval_inputs['input_ids'], eval_inputs['attention_mask'], eval_summ['input_ids'])

    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=1)
    dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=1)

    epochs = 20

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=len(dataloader_train)*epochs)
    
    verbose = 4

    train_loss_over_epochs = []
    val_loss_over_epochs = []

    best_rouge_l = 0

    for epoch in tqdm(range(1, epochs+1)):
        
        torch.cuda.empty_cache()

        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids'     : batch[0],
                      'attention_mask': batch[1],
                      'labels'        : batch[2],
                     }       

            del batch 
            outputs = model(**inputs)
            del inputs
            loss = outputs[0]
            del outputs
            loss_train_total += loss.item()
            loss.backward()
            del loss
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()


        train_loss_over_epochs.append( loss_train_total )
        gc.collect()

        if epoch % verbose == 0:
            # progress_bar = tqdm(dataloader_validation, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            torch.cuda.empty_cache()

            metrics = ['rouge1', 'rouge2', 'rougeL']

            scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

            results = {"rouge1_f":[], "rouge1_r":[], "rouge2_f":[], "rouge2_r":[], "rougeL_f":[], "rougeL_r":[]}
            results_avg = {}
            
            model.eval()

            loss_val_total = 0

            summaries = []
            targets = []

            for batch in dataloader_validation:

                batch = tuple(b.to(device) for b in batch)
                
                inputs = {'input_ids'     : batch[0],
                          'attention_mask': batch[1],
                          'labels'        : batch[2],
                         }       

                outputs = model(**inputs)
                del inputs
                loss = outputs[0]
                del outputs
                loss_val_total += loss.item()
                del loss

                summary_ids = model.generate(batch[0], num_beams=1, max_length=600, length_penalty=0.8, early_stopping=True)
                summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids] 
                del summary_ids
                summaries += summ 

                targets += [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in batch[2]]

                del batch

            val_loss_over_epochs.append( loss_val_total )


            for idx, summ in enumerate(summaries):
                target_summ = targets[idx]

                scores = scorer.score(target_summ.strip(), summ.strip())
                for metric in metrics:
                    results[metric+"_f"].append(scores[metric].fmeasure)
                    results[metric + "_r"].append(scores[metric].recall)

                for rouge_metric, rouge_scores in results.items():
                    results_avg[rouge_metric] = np.average(rouge_scores)

            progress_bar.write(f"Epoch {epoch+1}: loss_val_total: {loss_val_total}" )

            print(" ---- Evaluation Result ---- ")
            for metric, val in results_avg.items():
                print(f"{metric}: {np.round(val, 4)}")

            if results_avg["rougeL_f"] > best_rouge_l:
                print(" ---- Saving Best Model ---- ")
                best_rouge_l = results_avg["rougeL_f"]

                model_save = model_name.split("/")[-1]
                best_path = f"{save_path}/{model_save}_best_{epoch}_split_{data_idx}.pth"
                torch.save(model.state_dict(), best_path)

            gc.collect()


    del model 

    return best_path, train_loss_over_epochs, val_loss_over_epochs


def test(test_data, original_summ, tokenizer, model_name, best_path, data_idx):

    train_on_gpu=torch.cuda.is_available()
    device = torch.device('cuda' if train_on_gpu else 'cpu')
    
    test_inputs, test_summ = test_data

    dataset_test = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_summ['input_ids'])

    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=1)

    print('---- Final Evaluation ----')

    model = BartForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.load_state_dict(torch.load(best_path))
    model.eval()

    metrics = ['rouge1', 'rouge2', 'rougeL']

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    results = {"rouge1_f":[], "rouge1_r":[], "rouge2_f":[], "rouge2_r":[], "rougeL_f":[], "rougeL_r":[]}
    results_avg = {}


    summaries = []
    # targets = [] # note this line may be a bit repetitive

    for batch in tqdm(dataloader_test):

        batch = tuple(b.to(device) for b in batch)
        
        summary_ids = model.generate(batch[0], num_beams=5, max_length=600, length_penalty=0.8, early_stopping=True)
        summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids] 
        del summary_ids
        summaries += summ

        # targets += [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in batch[2]]

    summaries_trunc = []
    summaries_full = []

    for idx, summ in enumerate(summaries):
        target_summ = original_summ[idx]

        pred_sents = nltk.tokenize.sent_tokenize(summ)
        summ_full = " ".join(pred_sents)
        summaries_full.append(summ_full)

        summ_trunc = ""
        for sent in pred_sents:
            if len(summ_trunc.split()) + len(sent.split()) <= 600:
                summ_trunc += " " + sent
            else:
                break

        # summ_trunc = summ.split(" ")[:600]
        # summ_trunc = " ".join(summ_trunc)

        summaries_trunc.append(summ_trunc.strip())

        scores = scorer.score(target_summ.strip(), summ_trunc.strip())
        for metric in metrics:
            results[metric+"_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)


    print(" ---- Final Test Result ---- ")
    for metric, val in results_avg.items():
        print(f"{metric}: {np.round(val, 4)}")

    model_save = model_name.split("/")[-1]

    if not os.path.exists(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{data_idx}/baseline/Bart/"):
        os.mkdir(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{data_idx}/baseline/Bart/")

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{data_idx}/baseline/Bart/system_{model_save}.txt", "w") as f:
        pred_to_write = "\n".join( summaries_trunc )
        f.write(pred_to_write)

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{data_idx}/baseline/Bart/system_{model_save}_full.txt", "w") as f:
        pred_to_write = "\n".join( summaries_full )
        f.write(pred_to_write)

    with open(f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{data_idx}/baseline/Bart/reference_{model_save}.txt", "w") as f:
        truth_to_write = "\n\n".join( original_summ )
        f.write(truth_to_write)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Given URL of a paper, this script download the PDFs of the paper'
    )

    seed = 42
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = "facebook/bart-large"
    model_save = model_name.split("/")[-1]
    tokenizer = BartTokenizer.from_pretrained(model_name) # facebook/bart-large facebook/bart-large-cnn

    for data_idx in range(10):
        print(f" ------------------------ ")
        print(f" ------ idx: {data_idx} ------ ")
        print(f" ------------------------ ")

        avail_source_file = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{data_idx}/LongSumm2021/abstractive/avail"
        avail_target_file = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{data_idx}/original/abstractive/avail/summary/"

        test_source_file = avail_source_file.replace("avail", "test")
        test_target_file = avail_target_file.replace("avail", "test")

        base_path = "REPLACE-BY-YOUR-PATH/Bart/"

        if os.path.exists(f"{base_path}/multi_tokenized/tokenized_data_{model_save}_split_{data_idx}.pickle"):
            print("---- reading from existing tokenized data ----")
            with open(f"{base_path}/multi_tokenized/tokenized_data_{model_save}_split_{data_idx}.pickle", 'rb') as handle:
                data_tokenized = pickle.load(handle)
                train_data_tokenized, val_data_tokenized, test_data_tokenized = data_tokenized

            test_doc, test_summ = prepare_doc_and_summ_abstrative(test_source_file, test_target_file)

        else:

            print('---- reading data ----')
            avail_doc, avail_summ = prepare_doc_and_summ_abstrative(avail_source_file, avail_target_file)
            test_doc, test_summ = prepare_doc_and_summ_abstrative(test_source_file, test_target_file)

            split = int(0.9*len(avail_doc))

            train_doc, train_summ = avail_doc[:split], avail_summ[:split]
            val_doc, val_summ = avail_doc[split:], avail_summ[split:]

            avail_doc_length = [ len(i.split()) for i in avail_doc ]
            avail_summ_length = [ len(i.split()) for i in avail_summ ]
            test_doc_length = [ len(i.split()) for i in test_doc ]
            test_summ_length = [ len(i.split()) for i in test_summ ]

            print(f"avail_doc length max: {  max(avail_doc_length)  }; min: { min(avail_doc_length) }")
            print(f"avail_summ length max: {  max(avail_summ_length)  }; min: { min(avail_summ_length) }")
            print(f"test_doc length max: {  max(test_doc_length)  }; min: { min(test_doc_length) }")
            print(f"test_summ length max: {  max(test_summ_length)  }; min: { min(test_summ_length) }")


            print('---- tokenizing data ----')
            train_data_tokenized = tokenize_data(train_doc, train_summ, tokenizer, )
            val_data_tokenized = tokenize_data(val_doc, val_summ, tokenizer, )
            test_data_tokenized = tokenize_data(test_doc, test_summ, tokenizer, )

            print('---- saving tokenized results ----')
            with open(f"{base_path}/multi_tokenized/tokenized_data_{model_save}_split_{data_idx}.pickle", 'wb') as handle:
                pickle.dump([train_data_tokenized, val_data_tokenized, test_data_tokenized], handle)

        print('---- starting training ----')
        best_path, train_loss_over_epochs, val_loss_over_epochs = train_and_evaluation(train_data_tokenized, val_data_tokenized, tokenizer, model_name, base_path+"multi_bart_large", data_idx)
        print(f"best_path: {best_path}")
        print('train_loss_over_epochs: ', train_loss_over_epochs)
        print('val_loss_over_epochs: ', val_loss_over_epochs)

        print('---- starting testing ----') 
        

        gc.collect()
        torch.cuda.empty_cache()


        test(test_data_tokenized, test_summ, tokenizer, model_name, best_path, data_idx)