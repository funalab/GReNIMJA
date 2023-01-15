import pickle
import pandas as pd
import numpy as np
import os
import torch


bp = 1000
m_bp = 1000
stride = 1
amino_stride = 1
mer = 5
amino_mer = 4
amino_emb_dim = 100
dna_emb_dim = 50
HIDDEN_DIM = 10  # 隠れ層の次元
cnn = 1  # 1:　1×1conv有り
Calc = 0  # 1: add, 0: mean

dna2vec_file = 'edit_dna2vec'

dna_dict = '../all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dna2vec_dict.pickle'
dna_preVec = '../all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dna2vec.pickle'

os.makedirs('../all_vector', exist_ok=True)
df = pd.read_csv(dna2vec_file, sep='\t', names=['acid', 'vec'], header=1)

dna_weights = []
for index in range(len(df)):
    a = df.iat[index, 1].split(' ')
    a = list(map(float, a))
    dna_weights.append(a)
dna_weights = np.array(dna_weights)

with open('a', 'wb') as f:
    pickle.dump(dna_weights, f)

key = list(df['acid'])
# 単語IDの設定
keys, values = [], []
for i, word in enumerate(key):
    keys.append(word)
    values.append(i + 1)
d = dict(zip(keys, values))
with open('a', 'wb') as f:
    pickle.dump(d, f)

