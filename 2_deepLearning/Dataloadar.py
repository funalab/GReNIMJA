from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pickle
from torch.utils.data import DataLoader


def pad_collate(batch):
    (xx, yy, zz) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, zz_pad

def only_pad_collate(batch):
    (xx) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad

class NewsDataset(Dataset):
    def __init__(self, X, y, z):  # datasetの構成要素を指定
        self.X = X
        self.y = y
        self.z = z

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, xid):  # dataset[idx]で返す値を指定
        return [self.X[xid], self.y[xid], self.z[xid]]

class only_NewsDataset(Dataset):
    def __init__(self, X):  # datasetの構成要素を指定
        self.X = X

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, xid):  # dataset[idx]で返す値を指定
        return [self.X[xid]]

def data_to_Num(data, amino_dict, dna_dict):
    aName, dName = data['tf'], data['target']
    Aname_toNum = {}
    Dname_toNum = {}
    aseq = []
    dseq = []
    for i, k in enumerate(amino_dict):
        Aname_toNum[k] = i
    for i in range(len(aName)):
        aseq.append(Aname_toNum[aName[i]])

    for i, k in enumerate(dna_dict):
        Dname_toNum[k] = i
    for i in range(len(dName)):
        dseq.append(Dname_toNum[dName[i]])

    return aseq, dseq

def data_load(CV_para, short, bp, data_dir, K_i, batch_size, amino_dict, dna_dict):
    CV = CV_para[0]
    K = CV_para[1]
    train_batchsize = batch_size[0]
    valid_batchsize = batch_size[1]

    if CV == 'TRUE':
        with open('../data/' + data_dir + '/' + str(K) + 'CV/train' + str(K_i) + '.pickle', 'rb') as f:
            train = pickle.load(f)
        with open('../data/' + data_dir + '/' + str(K) + 'CV/valid' + str(K_i) + '.pickle', 'rb') as f:
            valid = pickle.load(f)

    else:
        with open('../data/' + data_dir + '/train.pickle', 'rb') as f:
            train = pickle.load(f)
        with open('../data/' + data_dir + '/valid.pickle', 'rb') as f:
            valid = pickle.load(f)

    with open('../data/pickle/test.pickle', 'rb') as f:
        test = pickle.load(f)

    if short == 'TRUE':
        train['ta_seq'] = train['ta_seq'].str[-bp:]
        valid['ta_seq'] = valid['ta_seq'].str[-bp:]
        test['ta_seq'] = test['ta_seq'].str[-bp:]

        #train = train[train['group'] < 20]
        #valid = valid[valid['group'] < 20]
        #test = test[test['group'] < 20]


        train['ta_seq'] = train['ta_seq'].str[-bp:]
        valid['ta_seq'] = valid['ta_seq'].str[-bp:]
        test['ta_seq'] = test['ta_seq'].str[-bp:]

    # 1epochあたりの繰り返し数
    train_iter_per_epoch = max(int(len(train) / train_batchsize), 1)
    valid_iter_per_epoch = max(int(len(valid) / valid_batchsize), 1)
    iter = [train_iter_per_epoch, valid_iter_per_epoch]

    if not isinstance(amino_dict, dict):
        train_aseq, train_dseq, train_ans = train['tf_seq'], train['ta_seq'], train['label']
        valid_aseq, valid_dseq, valid_ans = valid['tf_seq'], valid['ta_seq'], valid['label']
    else:
        train_aseq, train_dseq = data_to_Num(train, amino_dict, dna_dict)
        train_ans = train['label']
        valid_aseq, valid_dseq = data_to_Num(valid, amino_dict, dna_dict)
        valid_ans = valid['label']

    dataset_train = NewsDataset(train_aseq, train_dseq, train_ans)
    dataset_valid = NewsDataset(valid_aseq, valid_dseq, valid_ans)

    train_dataset = DataLoader(dataset=dataset_train, batch_size=train_batchsize, drop_last=True)
    valid_dataset = DataLoader(dataset=dataset_valid, batch_size=valid_batchsize, drop_last=True)

    return train_dataset, valid_dataset, iter


def test_dataLoad(short, bp, data_dir, batch_size, species, amino_dict, dna_dict):
    test_batchsize = batch_size[2]

    if species == 'mouse':
        with open('../data/pickle/Dmouse_test.pickle', 'rb') as f:
            test = pickle.load(f)

    else:
        with open('../data/pickle/test.pickle', 'rb') as f:
            test = pickle.load(f)
            

    if short == 'TRUE':
        test['ta_seq'] = test['ta_seq'].str[-bp:]

    test_iter = max(int(len(test) / test_batchsize), 1)

    if not isinstance(amino_dict, dict):
        test_aseq, test_dseq, test_ans = test['tf_seq'], test['ta_seq'], test['label']
    else:
        test_aseq, test_dseq = data_to_Num(test, amino_dict, dna_dict)
        test_ans = test['label']

    dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
    test_dataset = DataLoader(dataset=dataset_test, batch_size=test_batchsize, drop_last=True)

    return test, test_dataset, test_iter
