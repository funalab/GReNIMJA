import pandas as pd
from gensim.models.word2vec import Word2Vec
import pickle
import torch.nn as nn
import torch


class word_embedding:
    def __init__(self, data, dict):
        self.datafile, self.miss_datafile = data[2], data[3]
        self.bp = data[10]
        self.label = data[6]
        self.data_dir = data[11]
        self.amino_dict, self.dna_dict = dict[0], dict[1]
        self.amino_preVec, self.dna_preVec = dict[2], dict[3]
        self.amino_mer, self.dna_mer = dict[4], dict[5]
        self.amino_dim, self.dna_dim = dict[6], dict[7]
        self.amino_stride, self.dna_stride = dict[8], dict[9]
        self.a_dict, self.d_dict = dict[10], dict[11]

    def train_wordVec(self):
        if self.a_dict == 0 and self.d_dict == 0:
            return

        with open('../result/many_valid/data/many_valid/pickle/test.pickle', 'rb') as f:
            df = pickle.load(f)

        #with open('../data/' + self.data_dir + '/train.pickle', 'rb') as f:
            #df = pickle.load(f)

        #with open(self.datafile, 'rb') as f:
        #    df = pickle.load(f)

        #positive = df[df['confidence'] <= self.label]
        #encode = df[df['confidence'] == 'na']
        #positive = pd.concat([positive, encode])

        #with open(self.miss_datafile, 'rb') as f:
        #    miss_df = pickle.load(f)

        #df = pd.concat([positive, miss_df], axis=0)
        #df = df[~df['ta_seq'].str.contains('N')]
        #df = df[~df['tf_seq'].str.contains('X')]
        #df.reset_index(drop=True, inplace=True)

        #df['ta_seq'] = df['ta_seq'].str[-self.bp:]

        # vector
        a_k = self.amino_mer  # アミノ酸のk_merのk
        d_k = self.dna_mer  # DNA配列のk_merのk

        if self.a_dict != 0:
            # アミノ酸配列
            A_df = df.drop_duplicates(subset='tf_seq')
            A_df.reset_index(drop=True, inplace=True)
            amino_seq = A_df.loc[:, ['tf', 'tf_seq']]

            # k_merへの変換
            amino_k_mer = []
            for index in range(len(amino_seq)):
                mer = self.k_mer(amino_seq.iat[index, 1], a_k)
                amino_k_mer.append(mer)

            # 学習
            amino_model = Word2Vec(sentences=amino_k_mer, vector_size=self.amino_dim, min_count=1, window=10,
                                   epochs=1, sg=1, seed=1)
            amino_model.save(self.amino_preVec)

            # 単語IDの設定
            keys, values = [], []
            for i, word in enumerate(amino_model.wv.index_to_key):
                keys.append(word)
                values.append(i + 1)
            d = dict(zip(keys, values))
            with open(self.amino_dict, 'wb') as f:
                pickle.dump(d, f)

        if self.d_dict != 0:
            # DNA配列
            D_df = df.drop_duplicates(subset='ta_seq')
            D_df.reset_index(drop=True, inplace=True)
            dna_seq = D_df.loc[:, ['target', 'ta_seq']]
            dna_k_mer = []
            for index in range(len(dna_seq)):
                mer = self.k_mer(dna_seq.iat[index, 1], d_k)
                dna_k_mer.append(mer)
            # Word2Vecモデルの学習
            # sizeは特徴量の数、min_count以下の登場数の単語を無視、前後window幅の単語との関係を考慮、iter回数分繰り返し計算
            dna_model = Word2Vec(sentences=dna_k_mer, vector_size=self.dna_dim, min_count=1, window=10,
                                 epochs=15, sg=1, seed=1)
            dna_model.save(self.dna_preVec)

            # 単語IDの設定
            keys, values = [], []
            for i, word in enumerate(dna_model.wv.index_to_key):
                keys.append(word)
                values.append(i + 1)
            d = dict(zip(keys, values))
            with open(self.dna_dict, 'wb') as f:
                pickle.dump(d, f)

    # 配列をn-gramに分ける関数
    def k_mer(self, str, n):
        mer = []
        for i in range(len(str) - n + 1):
            mer.append(str[i:i + n])

        return mer

