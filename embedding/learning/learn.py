

import sys
import pickle
import torch.nn.functional
from torch.utils.data import DataLoader
import os
from gensim.models.word2vec import Word2Vec
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from embedding import word_embedding

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

stride = 1
amino_stride = 1
mer = 5
amino_mer = 4
amino_emb_dim = 100
dna_emb_dim = 50

# device
device = torch.device('cpu')
device2 = 'cpu'

# dataset
data_dir = 'pickle'
dataset = 'choice'
order = 2  # TRUE: 1, FALSE: 2
short = 'FALSE'
label = 'D'
valid_sample = 500
test_sample = 1000
valid_Datanum = 1000

pre = 'TRUE'

bp = 1000

amino_dict = '../all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                 '_' + str(amino_stride) + '_amino_dict.pickle'
amino_preVec = '../all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
               '_' + str(amino_stride) + '_Aword2vec.gensim.model'

os.makedirs('../all_vector', exist_ok=True)

if device2 == 'cpu':
    ## パラメータの設定 ##
    filename = '../data/ans.pickle'
    miss_datafile = '../data/2D_miss.pickle'
    TF_bunpu = '../data/TF_bunpu'

data = [device2, dataset, filename, miss_datafile, TF_bunpu, order, label, valid_sample,
        test_sample, valid_Datanum, bp, data_dir]


dict = [amino_dict, dna_dict, amino_preVec, dna_preVec, amino_mer, mer,
        amino_emb_dim, dna_emb_dim, amino_stride, stride, 0, 0]

# k-merの辞書がない
if not os.path.isfile(amino_dict):
    dict[10] += 1
if not os.path.isfile(dna_dict):
    dict[11] += 1

train_vector = word_embedding(data, dict)
train_vector.train_wordVec()
