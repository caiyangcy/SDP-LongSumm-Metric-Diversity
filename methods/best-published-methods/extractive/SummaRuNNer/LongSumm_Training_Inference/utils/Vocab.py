from os import replace
import torch
from nltk import sent_tokenize , word_tokenize

class Vocab():
    def __init__(self,embed,word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'

        self.substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]
    
    def __len__(self):
        return len(self.word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def replace_subs(self, text):
        for subs_tuple in self.substitutions:
            text = text.replace(subs_tuple[0], subs_tuple[1])
        return text

    def revert_subs(self, text):
        for subs_tuple in self.substitutions:
            text = text.replace(subs_tuple[1], subs_tuple[0])
        return text
    
    def make_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list,targets,doc_lens = [],[],[]
        # trunc document

        for batch_i in batch:
            doc, labels = batch_i['doc'], batch_i['labels']
            
            if doc == "" : 
                continue
            
            # print(f"doc: {doc}")
            # print("doc type: ", type(doc))

            doc = self.replace_subs(doc)

            sents = sent_tokenize(doc)

            sents = [self.revert_subs(sent) for sent in sents]

            # labels = label.split(split_token)
            assert len(sents) == len(labels), f"sents: {len(sents)}; labels: {len(labels)}"
            
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        
        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)
        
        features = torch.LongTensor(features)
        # print(targets)
        targets = torch.LongTensor(targets)
        summaries = [batch_i["summaries"] for batch_i in batch]

        return features, targets, summaries, doc_lens

    
    def make_features2(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list,doc_lens = [],[]

        for doc in batch:

            if doc == "" : 
                continue
            
            doc = self.replace_subs(doc)

            sents = sent_tokenize(doc)

            sents = [self.revert_subs(sent) for sent in sents]

            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            
            sents_list += sents
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
        
        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)
        
        features = torch.LongTensor(features)

        return features, doc_lens

    def get_input_doc(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list,targets,doc_lens = [],[],[]
        # trunc document

        for batch_i in batch:
            doc, labels = batch_i['doc'], batch_i['labels']
            
            if doc == "" : 
                continue
            
            # print(f"doc: {doc}")
            # print("doc type: ", type(doc))

            doc = self.replace_subs(doc)

            sents = sent_tokenize(doc)

            sents = [self.revert_subs(sent) for sent in sents]

            # labels = label.split(split_token)
            assert len(sents) == len(labels), f"sents: {len(sents)}; labels: {len(labels)}"
            
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            
            batch_sents.append(" ".join(words).strip())
        
        return " ".join(batch_sents).strip()


    def make_predict_features(self, batch, sent_trunc=150, doc_trunc=100, split_token='\n'):
        sents_list, doc_lens = [],[]
        # for doc in batch:
        for batch_i in batch:
            doc = batch_i['doc']
            doc = self.replace_subs(doc)
            sents = sent_tokenize(doc)
            sents = [self.revert_subs(sent) for sent in sents]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            if doc == "" : doc_lens.append(0)
            else : doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)
            
        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)
        return features, doc_lens