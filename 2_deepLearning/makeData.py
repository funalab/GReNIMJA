import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from func import make_group
import random
import os
import torch
import numpy as np

def make_Data(data_para):
    device2 = data_para[0]
    dataset = data_para[1]
    filename = data_para[2]
    miss_datafile = data_para[3]
    TF_bunpu = data_para[4]
    order = data_para[5]
    label = data_para[6]
    valid_sample = data_para[7]
    test_sample = data_para[8]
    valid_Datanum = data_para[9]
    bp = data_para[10]
    data_dir = data_para[11]

    if device2 == 'cpu':  # cpu用のdfを作成
        # 全てのデータを読み込みpickle形式で "longestData.pickle" に保存する。最初に一度実行した。
        df = pd.read_csv(filename, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))
        groups = pd.read_csv(TF_bunpu, header=None, names=['group'])
        df = make_group(df, groups, order)
        #df = pd.concat([df, group], axis=1, names='group')

    else:  # gpu用のdfを作成
        # データのOpen
        with open(filename, 'rb') as f:
            df = pickle.load(f)
        with open(miss_datafile, 'rb') as f:
            miss_df = pickle.load(f)
        groups = pd.read_csv(TF_bunpu, sep=' ', header=None, names=('tf', 'length', 'group'))

        df = make_group(df, groups, order)
        miss_df = make_group(miss_df, groups, order)
        positive = df[df['confidence'] <= label]
        encode = df[df['confidence'] == 'na']
        positive = pd.concat([positive, encode])
        df = pd.concat([positive, miss_df], axis=0)
        df.reset_index(drop=True, inplace=True)

    blen = len(df)
    # 未知の塩基および残基を含むデータを削除
    df = df[~df['ta_seq'].str.contains('N')]
    df = df[~df['tf_seq'].str.contains('X')]
    # target配列を指定した上流塩基数分にする
    df['ta_seq'] = df['ta_seq'].str[-bp:]
    # df['ta_seq'] = df['ta_seq'].str[:50]
    blen = len(df)
    alen = len(df)
    print(f'before size: {blen}, ->->->  after size: {alen}')

    if dataset == 'dchoice':
        random.seed(0)
        target = list(set(df['target'].tolist()))
        target = random.sample(target, test_sample)
        for i, ta in enumerate(target):
            tmp = df[df['target'] == ta]
            df = df[df['target'] != ta]
            if i == 0:
                test = tmp
            else:
                test = pd.concat([test, tmp], axis=0)

        target = list(set(df['target'].tolist()))
        target = random.sample(target, valid_sample)
        for i, ta in enumerate(target):
            tmp = df[df['target'] == ta]
            df = df[df['target'] != ta]
            if i == 0:
                valid = tmp
            else:
                valid = pd.concat([valid, tmp], axis=0)

        train = df

    if dataset == 'achoice':
        random.seed(0)
        tf = list(set(df['tf'].tolist()))
        tf = random.sample(tf, test_sample)
        for i, ta in enumerate(tf):
            tmp = df[df['tf'] == ta]
            df = df[df['tf'] != ta]
            if i == 0:
                test = tmp
            else:
                test = pd.concat([test, tmp], axis=0)

        tf = list(set(df['tf'].tolist()))
        tf = random.sample(tf, valid_sample)
        for i, ta in enumerate(tf):
            tmp = df[df['tf'] == ta]
            df = df[df['tf'] != ta]
            if i == 0:
                valid = tmp
            else:
                valid = pd.concat([valid, tmp], axis=0)

        train = df

    if dataset == 'Normal':
        testNum = 316298
        #testNum = int(len(df) * 0.2)
        #Num = int(testNum / 2)
        posi = df[df['label'] == 1]
        nega = df[df['label'] == 0]
        # true_nega = df[df['confidence'] == 'M']
        # false_nega = df[df['confidence'] == 'T']

        ptrain, ptest = train_test_split(posi, test_size=int(testNum / 2), shuffle=True, random_state=123)
        ptrain, pvalid = train_test_split(ptrain, test_size=int(valid_Datanum / 2), shuffle=True, random_state=123)
        ntrain, ntest = train_test_split(nega, test_size=int(testNum / 2), shuffle=True, random_state=123)
        ntrain, nvalid = train_test_split(ntrain, test_size=int(valid_Datanum / 2), shuffle=True, random_state=123)
        # ntrain = pd.concat([ntrain, false_nega], axis=0)

        train = pd.concat([ptrain, ntrain], axis=0)
        valid = pd.concat([pvalid, nvalid], axis=0)
        test = pd.concat([ptest, ntest], axis=0)

    if dataset == 'unbalance':
        posi = df[df['label'] == 1]
        nega = df[df['label'] == 0]

        ptrain, pvalid_test = train_test_split(posi, test_size=int(testNum / 2), shuffle=True, random_state=123)
        pvalid, ptest = train_test_split(pvalid_test, test_size=int(valid_Datanum / 2), shuffle=True,
                                         random_state=123)
        ntrain, nvalid_test = train_test_split(nega, test_size=int(testNum / 2), shuffle=True, random_state=123)
        nvalid, ntest = train_test_split(nvalid_test, test_size=int(valid_Datanum / 2), shuffle=True,
                                         random_state=123)

        train = pd.concat([ptrain, ntrain], axis=0)
        valid = pd.concat([pvalid, nvalid], axis=0)
        test = pd.concat([ptest, ntest], axis=0)

    train = train.sample(frac=1, random_state=123)
    valid = valid.sample(frac=1, random_state=123)
    test = test.sample(frac=1, random_state=123)

    train = train.sort_values('group', ascending=False)
    valid = valid.sort_values('group', ascending=False)
    test = test.sort_values('group', ascending=False)

    trainlist = train.drop(['tf_seq', 'ta_seq'], axis=1)
    validlist = valid.drop(['tf_seq', 'ta_seq'], axis=1)
    testlist = test.drop(['tf_seq', 'ta_seq'], axis=1)
    os.makedirs('../data/dataset', exist_ok=True)
    trainlist.to_csv('../data/dataset/train.txt', sep='\t', index=False)
    validlist.to_csv('../data/dataset/valid.txt', sep='\t', index=False)
    testlist.to_csv('../data/dataset/test.txt', sep='\t', index=False)

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    os.makedirs('../data/pickle', exist_ok=True)
    with open('../data/' + data_dir + '/train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('../data/' + data_dir + '/valid.pickle', 'wb') as f:
        pickle.dump(valid, f)
    with open('../data/' + data_dir + '/test.pickle', 'wb') as f:
        pickle.dump(test, f)



def cross_validation(K, data_dir, dataset):
    with open('../data/' + data_dir + '/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('../data/' + data_dir + '/valid.pickle', 'rb') as f:
        valid = pickle.load(f)

    train = pd.concat([train, valid], axis=0)
    train.reset_index(drop=True, inplace=True)
    true_train = train
    trains, valids = [], []

    if dataset == 'Normal':
        posi = true_train[true_train['label'] == 1]
        nega = true_train[true_train['label'] == 0]
        Num = int(len(true_train) / K)
        for i in range(K):
            ptrain, pvalid = train_test_split(posi, test_size=int(Num / 2), shuffle=True, random_state=i)
            ntrain, nvalid = train_test_split(nega, test_size=int(Num / 2), shuffle=True, random_state=i)
            train = pd.concat([ptrain, ntrain], axis=0)
            valid = pd.concat([pvalid, nvalid], axis=0)

            train = train.sample(frac=1, random_state=123)
            valid = valid.sample(frac=1, random_state=123)
            train = train.sort_values('group', ascending=False)
            valid = valid.sort_values('group', ascending=False)
            train.reset_index(drop=True, inplace=True)
            valid.reset_index(drop=True, inplace=True)
            trains.append(train)
            valids.append(valid)
            print(trains[i])
            print(valids[i])

    else:
        if dataset == 'dchoice':
            target_list = sorted(list(set(train['target'].tolist())))
        elif dataset == 'achoice':
            target_list = sorted(list(set(train['tf'].tolist())))
        target_num = int(len(target_list) / K)
        target_nums = []

        #  validデータに含まれるDNA配列の種類数をtarget_numsに格納する
        for i in range(K):
            target_nums.append(target_num)
        i = 0
        while sum(target_nums) < len(target_list):
            target_nums[i] += 1
            i += 1

        random.seed(0)
        random.shuffle(target_list)

        l = 0
        for i in range(len(target_nums)):
            train = true_train
            for j, ta in enumerate(target_list[l:l + target_nums[i]]):
                if dataset == 'dchoice':
                    tmp = train[train['target'] == ta]
                    train = train[train['target'] != ta]
                elif dataset == 'achoice':
                    tmp = train[train['tf'] == ta]
                    train = train[train['tf'] != ta]
                if j == 0:
                    valids.append(tmp)
                else:
                    valids[i] = pd.concat([valids[i], tmp], axis=0)
            l += target_nums[i]
            train = train.sample(frac=1, random_state=123)
            valids[i] = valids[i].sample(frac=1, random_state=123)
            train = train.sort_values('group', ascending=False)
            valids[i] = valids[i].sort_values('group', ascending=False)
            train.reset_index(drop=True, inplace=True)
            valids[i].reset_index(drop=True, inplace=True)
            trains.append(train)

    os.makedirs('../data/' + data_dir + '/' + str(K) + 'CV', exist_ok=True)
    for i in range(K):
        with open('../data/' + data_dir + '/' + str(K) + 'CV/train' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(trains[i], f)
        with open('../data/' + data_dir + '/' + str(K) + 'CV/valid' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(valids[i], f)



class convert:  # アミノ酸配列とDNA配列をone-hotベクトルに変換する
    def __init__(self, embedding, species):
        self.embedding = embedding
        self.species = species
        self.dim = 0
        self.NCPstart = 0
        self.DPCPstart = 0
        self.dict = {}

        self.amino = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                      'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        if embedding in [0, 1, 2, 3]:
            self.dim += 4
            self.NCPstart += 4
            self.DPCPstart += 4
            self.One_hot = {'A': [1, 0, 0, 0],
                            'C': [0, 1, 0, 0],
                            'G': [0, 0, 1, 0],
                            'T': [0, 0, 0, 1]}
        if embedding in [1, 2, 4, 6]:
            self.NCP = {'A': [1, 1, 1],
                        'T': [0, 1, 0],
                        'G': [1, 0, 0],
                        'C': [0, 0, 1]}
            self.dim += 3
            self.DPCPstart += 3
        if embedding in [1, 3, 5, 6]:
            self.DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332,
                           0.5856582729115565,
                           0.5476708282666789],
                    'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
                           0.76847598772376],
                    'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182,
                           0.5249586459219764,
                           0.45903777008667923],
                    'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978,
                           0.7888705476333944,
                           0.7467063799220581],
                    'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257,
                           0.3059118434042811,
                           0.32686549630327577],
                    'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
                           0.5476708282666789],
                    'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657,
                           0.45898067049412195,
                           0.3501900760908136],
                    'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978,
                           0.49856742124957026,
                           0.6891727614587756],
                    'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978,
                           0.49856742124957026,
                           0.6891727614587756],
                    'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978,
                           0.7888705476333944,
                           0.7467063799220581],
                    'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315,
                           0.4246720956706261,
                           0.6083143907016332],
                    'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165,
                           0.6780274730118172,
                           0.8400043540595654],
                    'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657,
                           0.45898067049412195,
                           0.3501900760908136],
                    'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182,
                           0.5249586459219764,
                           0.45903777008667923],
                    'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134,
                           0.5170412957708868,
                           0.32686549630327577],
                    'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315,
                           0.4246720956706261,
                           0.6083143907016332]}
            self.dim += 6

    def amino_convert_vector(self, aseq):
        for index in range(len(aseq)):
            amino_seq = list(aseq.iat[index, 1])  # iat:列番号と行番号で位置を指定してその要素を取得
            amino_vector = torch.from_numpy(self.amino_convert_one_hot(self.amino, amino_seq).astype(np.float32))
            self.dict[aseq.iat[index, 0]] = amino_vector
        with open('../data/pickle/dict/' + str(self.species) + '_' + str(self.embedding) + 'Aonehot.pickle', 'wb') as f:
            pickle.dump(self.dict, f)
        '''
        for k in self.dict.keys():
            print(k)
            exit()
        '''

    def dna_convert_vector(self, dseq):
        for index in range(len(dseq)):
            dna_seq = dseq.iat[index, 3]
            dna_vector = np.zeros((len(dna_seq), self.dim), dtype=np.float32)
            for pos in range(len(list(dna_seq))):
                if self.embedding in [0, 1, 2, 3]:
                    dna_vector[pos, 0:4] += np.asarray(np.float32(self.One_hot[dna_seq[pos]]))
                if self.embedding in [1, 2, 4, 6]:
                        dna_vector[pos, self.NCPstart:self.NCPstart+3] += np.asarray(np.float32(self.NCP[dna_seq[pos]]))
                if self.embedding in [1, 3, 5, 6]:
                    if pos != 0 and pos != len(list(dna_seq)) - 1:
                        dna_vector[pos, self.DPCPstart:self.DPCPstart+6] += np.asarray(np.float32(self.DPCP[dna_seq[pos - 1:pos + 1]])) / 2
                        dna_vector[pos, self.DPCPstart:self.DPCPstart+6] += np.asarray(np.float32(self.DPCP[dna_seq[pos:pos + 2]])) / 2
                    elif pos == 0:
                        dna_vector[pos, self.DPCPstart:self.DPCPstart+6] += np.asarray(np.float32(self.DPCP[dna_seq[pos:pos + 2]]))
                    else:
                        dna_vector[pos, self.DPCPstart:self.DPCPstart+6] += np.asarray(np.float32(self.DPCP[dna_seq[pos - 1:pos + 1]]))
                        dna_vector = torch.from_numpy(dna_vector)
            self.dict[dseq.iat[index, 2]] = dna_vector
        with open('../data/pickle/dict/' + str(self.species) + '_' + str(self.embedding) + 'Donehot.pickle', 'wb') as f:
            pickle.dump(self.dict, f)

    def amino_convert_one_hot(self, amino, amino_seq):  # アミノ酸配列をone-hotに変換する
        aminos = list(amino_seq)
        leng = len(aminos)
        one_hot = np.zeros((leng, 20), dtype=np.float32)
        corpus = np.array([amino[w] for w in aminos])
        for idx_0, amino_id in enumerate(corpus):
            one_hot[idx_0, amino_id] = 1
        return one_hot


def make_oneHot(filename, species):
    os.makedirs('../data/pickle/dict', exist_ok=True)
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    # amino
    aseq = df.drop_duplicates(subset='tf')
    aseq = aseq[~aseq['tf_seq'].str.contains('X')]
    aseq.reset_index(drop=True, inplace=True)
    vector = convert(0, species)
    vector.amino_convert_vector(aseq)

    # dna
    dseq = df.drop_duplicates(subset='target')
    dseq = dseq[~dseq['ta_seq'].str.contains('N')]
    dseq.reset_index(drop=True, inplace=True)
    for embedding in range(7):
        vector = convert(embedding, species)
        vector.dna_convert_vector(dseq)
