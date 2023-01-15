import pandas as pd
from gensim.models.word2vec import Word2Vec
import pickle
import torch.nn as nn
import torch

######### パラメータ ##########

# filename
#aminoSeq_data = '../1_make_Ans_data/GCF_000001405.39/proteomes/GCF_000001405.39_protein_refseq.faa'
#dnaSeq_data = '../1_make_Ans_data/GCF_000001405.39/genomes/GCF_000001405.39_genomic_refseq.fna'

datafile = '../data/ans.pickle'
miss_datafile = '../data/D_dna_miss.pickle'

bp = 1000
label = 'D'


#df = pd.read_csv(filename, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))

with open(datafile, 'rb') as f:
    df = pickle.load(f)

positive = df[df['confidence'] <= label]
encode = df[df['confidence'] == 'na']
positive = pd.concat([positive, encode])

with open(miss_datafile, 'rb') as f:
    miss_df = pickle.load(f)

df = pd.concat([positive, miss_df], axis=0)
df = df[~df['ta_seq'].str.contains('N')]
df = df[~df['tf_seq'].str.contains('X')]
df.reset_index(drop=True, inplace=True)

df['ta_seq'] = df['ta_seq'].str[-bp:]

A_df = df.drop_duplicates(subset='tf_seq')
D_df = df.drop_duplicates(subset='ta_seq')
print(len(A_df))
print(len(D_df))

# vector
a_k = 6  # アミノ酸のk_merのk
d_k = 6  # DNA配列のk_merのk
#############################


########### 関数 ############

# 配列をn-gramに分ける関数
def k_mer(str, n):
    mer = []
    for i in range(len(str) - n + 1):
        mer.append(str[i:i+n])

    return mer
############################


# 辞書の作成
word_to_id = {}
for index in range(len(A_df)):
    print(index)
    mer = k_mer(A_df.iat[index, 1], a_k)
    for word in mer:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id

with open(str(a_k) + '_amino_dict.pickle', 'wb') as f:
    pickle.dump(word_to_id, f)

word_to_id = {}
for index in range(len(D_df)):
    print(index)
    mer = k_mer(D_df.iat[index, 3], d_k)
    for word in mer:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id

with open(str(d_k) + '_dna_dict.pickle', 'wb') as f:
    pickle.dump(word_to_id, f)

exit()
aminoSeq_data = '../1_make_Ans_data/GCF_000001405.39/proteomes/GCF_000001405.39_protein_refseq.faa'
dnaSeq_data = '../1_make_Ans_data/GCF_000001405.39/genomes/GCF_000001405.39_genomic_refseq.fna'


#　配列ファイルの読み込み
amino_seq = pd.read_csv(aminoSeq_data, sep='\t', names=('tf', 'tf_seq'))
dna_seq = pd.read_csv(dnaSeq_data, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))

# k_merへの変換
# アミノ酸配列
amino_k_mer = []
for index in range(len(amino_seq)):
    mer = k_mer(amino_seq.iat[index,1], a_k)
    amino_k_mer.append(mer)

print(amino_k_mer)
amino_model = Word2Vec(sentences=amino_k_mer, vector_size=100, min_count=1, window=10, epochs=15, sg=1, seed=1)
amino_model.save('new_Aword2vec.gensim.model')

amino_word_model = Word2Vec.load('new_Aword2vec.gensim.model')
keys = []
values = []
for i, word in enumerate(amino_word_model.wv.index_to_key):
    keys.append(word)
    values.append(i)
d = dict(zip(keys, values))
with open ('new_amino_dict.pickle', 'wb') as f:
    pickle.dump(d, f)

exit()

# DNA配列
dna_k_mer = []
for index in range(len(dna_seq)):
    mer = k_mer(dna_seq.iat[index,3], k)
    dna_k_mer.append(mer)
#Word2Vecモデルの学習
#sizeは特徴量の数、min_count以下の登場数の単語を無視、前後window幅の単語との関係を考慮、iter回数分繰り返し計算

dna_model = Word2Vec(sentences=dna_k_mer, vector_size=100, min_count=1, window=10, epochs=15, sg=1, seed=1)
dna_model.save('new_Dword2vec.gensim.model')


'''
with open ('test.pickle', 'wb') as f:
    pickle.dump(dict, f)

with open('test.pickle', 'rb') as f:
    test = pickle.load(f)

print(test)
print(test['C'])
'''
# 単語IDの設定
# アミノ酸配列
amino_word_model = Word2Vec.load('new_Aword2vec.gensim.model')
keys = []
values = []
for i, word in enumerate(amino_word_model.wv.index_to_key):
    keys.append(word)
    values.append(i)
d = dict(zip(keys, values))
with open ('new_amino_dict.pickle', 'wb') as f:
    pickle.dump(d, f)

# DNA配列
dna_word_model = Word2Vec.load('new_Dword2vec.gensim.model')
keys = []
values = []
for i, word in enumerate(dna_word_model.wv.index_to_key):
    keys.append(word)
    values.append(i + 1)
d = dict(zip(keys, values))
with open ('new_dna_dict.pickle', 'wb') as f:
    pickle.dump(d, f)
