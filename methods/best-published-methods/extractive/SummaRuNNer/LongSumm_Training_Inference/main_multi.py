#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import pandas as pd
from torch.autograd import Variable
# from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
from datetime import datetime
from nltk import sent_tokenize 
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='REPLACE-BY-YOUR-PATH/Summaformers/LongSumm_Training_Inference/fine-tuning-ckpt-multi-train-abs/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='REPLACE-BY-YOUR-PATH/datasets/dataset/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_train.txt')
parser.add_argument('-val_dir',type=str,default='REPLACE-BY-YOUR-PATH/datasets/dataset/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_eval.txt')
parser.add_argument('-embedding',type=str,default='/home/caiyang/Documents/glove-6b/glove.6B.100d.txt')
parser.add_argument('-word2id',type=str,default='word2id_glove_100d.json')
parser.add_argument('-report_every',type=int,default=50)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
parser.add_argument('-retrain',type=bool,default=False)
parser.add_argument('-split',type=int)

# test
parser.add_argument('-load_dir',type=str,default='REPLACE-BY-YOUR-PATH/Summaformers/LongSumm_Training_Inference/fine-tuning-checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='REPLACE-BY-YOUR-PATH/datasets/dataset/SummaFormer/extractive/LongSumm_extractive_withlabels_test.txt')
parser.add_argument('-ref',type=str,default='REPLACE-BY-YOUR-PATH/Summaformers/outputs/extractive/ref')
parser.add_argument('-hyp',type=str,default='REPLACE-BY-YOUR-PATH/Summaformers/outputs/extractive/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-foldername' ,type=str,default='def_folder')
parser.add_argument('-topk',type=int,default=15)
# device
parser.add_argument('-device',type=int, default=0)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 
    
substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]

def replace_subs(text):
    for subs_tuple in substitutions:
        text = text.replace(subs_tuple[0], subs_tuple[1])
    return text

def revert_subs(text):
    for subs_tuple in substitutions:
        text = text.replace(subs_tuple[1], subs_tuple[0])
    return text


def eval(net, vocab, val_dataset, criterion, batch_size):
    net.eval()
    total_loss = 0
    batch_num = 0

    for i in tqdm( range(0, len(val_dataset), batch_size), desc=f"Eval" ):

        batch = val_dataset[ i : i+batch_size ]

        features,targets,_,doc_lens = vocab.make_features(batch, sent_trunc=256, doc_trunc=400)
        del batch 
        features, targets = Variable(features), Variable(targets.float())

        if use_gpu:
            features = features.cuda()

        probs = net(features,doc_lens)

        del features
        gc.collect()
        torch.cuda.empty_cache()

        if use_gpu:
            targets = targets.cuda()

        loss = criterion(probs,targets)
        del probs, targets
        gc.collect()
        torch.cuda.empty_cache()

        total_loss += loss.item()
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        batch_num += 1

    loss = total_loss / batch_num
    net.train()

    return loss

def train():
    logging.info('Loading vocab, train and val dataset. Wait a second,please')
    
    w2vec = pd.read_csv(args.embedding, header=None, sep=' ', quoting=3, encoding="ISO-8859-1")
    embed = torch.Tensor( w2vec.iloc[:, 1:].values )

    with open(args.word2id) as f:
        word2id = json.load(f)

    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = json.load(f)#[:100]

    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = json.load(f)

    val_dataset = utils.Dataset(examples)

    # update args
    print('embed size: ', embed.size() )
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]

    # build model
    if args.retrain:
        if use_gpu:
            checkpoint = torch.load(args.load_dir)
        else:
            checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
        if not use_gpu:
            checkpoint['args'].device = None
        net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
        net.load_state_dict(checkpoint['model'])
    else:
        net = getattr(models, args.model)(args,embed)
    
    if use_gpu:
        net.cuda()

    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net,flush = True)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    batch_size = args.batch_size
    t1 = time() 

    for epoch in range(1, args.epochs+1):
   
        for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch}"):

            batch = train_dataset[ i : i+batch_size ]
            features, targets, _, doc_lens = vocab.make_features(batch, sent_trunc=256, doc_trunc=400)

            # print('make features')

            # print('features: ', features.shape )

            del batch

            gc.collect()
            torch.cuda.empty_cache()

            features, targets = Variable(features), Variable(targets.float())

            if use_gpu:
                features = features.cuda()

            # print('passing net')
            probs = net(features, doc_lens) 
            del features

            gc.collect()
            torch.cuda.empty_cache()


            if use_gpu:
                targets = targets.cuda()

            loss = criterion(probs, targets)

            del probs, targets

            gc.collect()
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward()
            
            del loss
            gc.collect()
            torch.cuda.empty_cache()

            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()

            if args.debug:
                print('Batch ID:%d Loss:%f' %(i,loss.data[0]))
                continue

            if i % args.report_every == 0:

                cur_loss = eval(net, vocab, val_dataset, criterion, batch_size)

                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.save()

                logging.info('\nEpoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f\n' % (epoch,min_loss,cur_loss))

        
        gc.collect()
        torch.cuda.empty_cache()

    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))




def test():
     
    # embed = torch.Tensor(np.load(args.embedding)['embedding'])
    w2vec = pd.read_csv(args.embedding, header=None, sep=' ', quoting=3, encoding="ISO-8859-1")
    embed = torch.Tensor( w2vec.iloc[:, 1:].values )

    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = json.load(f)

    test_dataset = utils.Dataset(examples)

    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None

    net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])

    if use_gpu:
        net.cuda()

    net.eval()
    
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1

    batch_size = args.batch_size

    references_to_write = []
    system_to_write = []

    for i in range(0, len(test_dataset), batch_size):

        batch = test_dataset[ i : i+batch_size ]
        features, _, summaries, doc_lens = vocab.make_features(batch, sent_trunc=10000, doc_trunc=400)

        t1 = time()

        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)

        t2 = time()

        time_cost += t2 - t1
        start = 0

        for doc_id, doc_len in enumerate(doc_lens):

            stop = start + doc_len
            prob = probs[start:stop]

            topk = min(args.topk, doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()

            doc = batch[doc_id]['doc']#.split('\n')[:doc_len]

            doc = replace_subs(doc)

            sents = sent_tokenize(doc)

            sents = [revert_subs(sent) for sent in sents]

            hyp = [sents[index].replace("\n", " ") for index in topk_indices]
            ref = summaries[doc_id]

            references_to_write.append( ref )
            hyp = " ".join(hyp).strip()
            hyp = hyp.encode('utf-8', 'replace').decode()

            system_to_write.append( hyp )

            start = stop
            file_id = file_id + 1


    with open(os.path.join(args.ref, 'reference.txt'), 'w') as f:
        ref_to_write = "\n\n".join(references_to_write)
        f.write(ref_to_write)

    with open(os.path.join(args.hyp, 'system.txt'), 'w') as f:
        sys_to_write = "\n".join(system_to_write)
        f.write(sys_to_write)


    print('Speed: %.2f docs / s' % (doc_num / time_cost))



if __name__=='__main__':


    for idx in range(10):
        print(f" ----- idx: {idx} -----")
        args.batch_size = 2
        args.pos_num = 400

        args.train_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_train.txt"
        args.val_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/extractive/LongSumm_extractive_withlabels_avail_eval.txt"

        args.test_dir = f"REPLACE-BY-YOUR-PATH/datasets/dataset_multiple_splits/split_{idx}/SummaFormer/abstractive/LongSumm_abstractive_withlabels_test.txt"

        args.ref = f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{idx}/baseline/SummaFormer_train_on_abs/"
        args.hyp = f"REPLACE-BY-YOUR-PATH/leaderboard_splits/split_{idx}/baseline/SummaFormer_train_on_abs/"

        if not os.path.exists(args.ref):
            os.mkdir(args.ref)

        args.load_dir = f"REPLACE-BY-YOUR-PATH/Summaformers/LongSumm_Training_Inference/fine-tuning-ckpt-multi-train-abs/RNN_RNN_seed_1_split_{idx}.pt"

        args.split = idx

        print(" ----- TRAINING -----")
        train()

        args.kernel_sizes = "3,4,5"

        gc.collect()
        torch.cuda.empty_cache()

        print(" ----- TESTING -----")
        test()

        args.kernel_sizes = "3,4,5"
