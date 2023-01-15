import sys
import pickle
from func import convert, NewsDataset, Embedding
from layers import myModel
import torch.nn.functional
from torch.utils.data import DataLoader
from test import test_model
import os
from gensim.models.word2vec import Word2Vec
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

traincheck = 'TRUE'
testcheck = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# device
device = torch.device('cuda')
device2 = 'cuda'

# dataset
old_data = 'TRUE'
dataset = 'choice'
test_batchsize = 1

# model
rnn = 'gru'  # lstm or gru
att = 'NULL'  # att or coatt or NULL
embedding = 'embedding'
#embedding = 'one_hot'
learn = 'TRUE'
pre = 'FALSE'
emb_num = ''
view = 'FALSE'
check = './checkpoint/checkpoint3.pt'

# Attention
DA = 10  # AttentionをNeural Networkで計算する際の重み行列のサイズ
R = 1  # Attentionは1層

args = sys.argv

bp = 1000
stride = 1
amino_stride = 1
mer = 6
amino_mer = 6
amino_emb_dim = 100
dna_emb_dim = 100
HIDDEN_DIM = 50  # 隠れ層の次元
cnn = 1  # 1:　1×1conv有り
Calc = 0  # 1: add, 0: mean

if len(args) > 1:
    bp = args[1]
    stride = args[2]
    amino_stride = args[3]
    mer = args[4]
    amino_mer = args[5]
    amino_emb_dim = args[6]
    dna_emb_dim = args[7]
    HIDDEN_DIM = args[8]
    cnn = args[9]
    Calc = args[10]

# filename
if emb_num == 'all':
    amino_dict = str(dir) + '../embedding/all_dict/' + str(amino_mer) + '_amino_dict.pickle'
    dna_dict = str(dir) + '../embedding/all_dict/' + str(mer) + '_dna_dict.pickle'
    amino_dict = str(dir) + '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                 '_' + str(amino_stride) + '_amino_dict.pickle'
    dna_dict = str(dir) + '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_dna_dict.pickle'
else:
    amino_dict = str(dir) + '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                 '_' + str(amino_stride) + '_amino_dict.pickle'
    dna_dict = str(dir) + '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_dna_dict.pickle'
amino_preVec = str(dir) + '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
               '_' + str(amino_stride) + '_Aword2vec.gensim.model'
dna_preVec = str(dir) + '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dword2vec.gensim.model'

if device2 == 'cpu':
    ## パラメータの設定 ##
    batch_size = 50
    ####################
else:
    batch_size = 250
    ####################

para = [embedding, pre, att, learn, cnn, Calc, rnn]

## データの分割 ##
with open('../data/pickle/train.pickle', 'rb') as f:
    train = pickle.load(f)
with open('../data/pickle/valid.pickle', 'rb') as f:
    valid = pickle.load(f)
with open('../data/pickle/test.pickle', 'rb') as f:
    test = pickle.load(f)

train_size, test_size = len(train), len(test)

train_aseq, test_aseq = train['tf_seq'], test['tf_seq']
train_dseq, test_dseq = train['ta_seq'], test['ta_seq']
train_ans, test_ans = train['label'], test['label']

dataset_train = NewsDataset(train_aseq, train_dseq, train_ans)
dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)

train_dataset = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True)
test_dataset = DataLoader(dataset=dataset_test, batch_size=test_batchsize, drop_last=True)

#################################

# 値の初期化
amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim = 0, 0, 0, 0, 0

## モデルの定義 ##
if embedding == 'one_hot':
    amino = {'M': 0, 'L': 1, 'K': 2, 'F': 3, 'A': 4, 'Q': 5, 'E': 6, 'T': 7, 'P': 8, 'V': 9, 'N': 10, 'G': 11, 'S': 12,
             'R': 13, 'I': 14, 'D': 15, 'Y': 16, 'C': 17, 'H': 18, 'W': 19}
    dna = {'G': 0, 'C': 1, 'T': 2, 'A': 3}
    ans = {'0': 0, '1': 1}
    vector = convert(amino, dna, ans)
    # アミノ酸配列の種類数
    A_emb_dim = 20
    # 塩基配列の種類数
    D_emb_dim = 4

if embedding == 'embedding':
    with open(amino_dict, 'rb') as f:
        amino_dict = pickle.load(f)
    with open(dna_dict, 'rb') as f:
        dna_dict = pickle.load(f)
    if pre != 'TRUE':
        A_emb_dim = amino_emb_dim
        D_emb_dim = dna_emb_dim
        Avocab_size = len(amino_dict)
        Dvocab_size = len(dna_dict)
    else:
        amino_word = Word2Vec.load('Aword2vec.gensim.model')
        amino_weights = amino_word.syn1neg
        A_emb_dim = amino_weights.shape[1]
        Avocab_size = amino_weights.shape[0] + 1
        amino_weights = torch.from_numpy(amino_weights)
        zero = torch.zeros([1, A_emb_dim])
        amino_weights = torch.cat((zero, amino_weights), dim=0)
        dna_word = Word2Vec.load('Dword2vec.gensim.model')
        dna_weights = dna_word.syn1neg
        D_emb_dim = dna_weights.shape[1]
        Dvocab_size = dna_weights.shape[0] + 1
        zero = torch.zeros([1, D_emb_dim])
        dna_weights = torch.from_numpy(dna_weights)
        dna_weights = torch.cat((zero, dna_weights), dim=0)

    vector = Embedding(amino_dict, dna_dict, stride, mer, amino_stride, amino_mer)
    # 配列の次元数
    emb_dim = [A_emb_dim, D_emb_dim]

# 1epochあたりの繰り返し数
train_iter_per_epoch = max(int(train_size / batch_size), 1)
test_iter_per_epoch = max(int(test_size / test_batchsize), 1)


# モデル宣言
model = myModel(HIDDEN_DIM, batch_size, DA, R, amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim, para)

print('model sengen')
# 損失関数はクロスエントロピー誤差を使う(シグモイド関数にはこれがオススメらしい)
lossFn = torch.nn.functional.binary_cross_entropy

# 最適化の手法はAdamを使う(適当)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# training
if device != 'cpu':
    model.to(device)

checkpoint = torch.load(check, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"] + 1

if traincheck == 'TRUE':
    os.makedirs('result/train_Details', exist_ok=True)
    TPs, FPs, FNs, TNs, score = test_model(vector, train_dataset, batch_size, model, train_iter_per_epoch, device,
                                           embedding)
    test_TPs = train.iloc[TPs, [0, 1, 2, 3, 4, 5, 6]]
    test_TPs.to_csv('result/train_Details/TPs', sep='\t', index=False)
    test_FPs = train.iloc[FPs, [0, 1, 2, 3, 4, 5, 6]]
    test_FPs.to_csv('result/train_Details/FPs', sep='\t', index=False)
    test_FNs = train.iloc[FNs, [0, 1, 2, 3, 4, 5, 6]]
    test_FNs.to_csv('result/train_Details/FNs', sep='\t', index=False)
    test_TNs = train.iloc[TNs, [0, 1, 2, 3, 4, 5, 6]]
    test_TNs.to_csv('result/train_Details/TNs', sep='\t', index=False)

    test["score"] = score
    testlist = train.drop(['tf_seq', 'ta_seq'], axis=1)
    testlist.to_csv('result/train_Details/trainscore.txt', sep='\t', index=False)
    y_true = list(test['label'])
    y_score = list(test['score'])

    # AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    print(thresholds)
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('result/train_Details/sklearn_roc_curve.pdf')

    print(roc_auc_score(y_true, y_score))

if testcheck == 'TRUE':
    os.makedirs('result/Details', exist_ok=True)
    TPs, FPs, FNs, TNs, score = test_model(vector, test_dataset, test_batchsize, model, test_iter_per_epoch, device,
                                           embedding)
    test_TPs = test.iloc[TPs, [0, 1, 2, 3, 4, 5, 6]]
    test_TPs.to_csv('result/Details/TPs', sep='\t', index=False)
    test_FPs = test.iloc[FPs, [0, 1, 2, 3, 4, 5, 6]]
    test_FPs.to_csv('result/Details/FPs', sep='\t', index=False)
    test_FNs = test.iloc[FNs, [0, 1, 2, 3, 4, 5, 6]]
    test_FNs.to_csv('result/Details/FNs', sep='\t', index=False)
    test_TNs = test.iloc[TNs, [0, 1, 2, 3, 4, 5, 6]]
    test_TNs.to_csv('result/Details/TNs', sep='\t', index=False)

    test["score"] = score
    testlist = test.drop(['tf_seq', 'ta_seq'], axis=1)
    testlist.to_csv('result/train_Details/testscore.txt', sep='\t', index=False)
    y_true = list(test['label'])
    y_score = list(test['score'])

    # AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    print(thresholds)
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('result/train_Details/sklearn_roc_curve.pdf')

    print(roc_auc_score(y_true, y_score))



