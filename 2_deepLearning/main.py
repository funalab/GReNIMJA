# CNNだけどatt=Falseの組み合わせはできない

import sys
import pickle
from func import convert, NewsDataset, Embedding, initialize
from layers import myModel
import torch.nn.functional
from train import train_model
from torch.utils.data import DataLoader
from test import test_model
import os
from gensim.models.word2vec import Word2Vec
import torch.nn as nn
from makeData import make_Data, cross_validation, make_oneHot
import matplotlib.pyplot as plt
from Dataloadar import data_load, test_dataLoad
from embedding import word_embedding
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# device
device = torch.device('cpu')
device2 = 'cpu'

# initial state
check = './checkpoint/init_checkpoint.pt'

# species
species = 'human'

# dataset
data_dir = 'pickle'
dataset = 'choice'
order = 2  # TRUE: 1, FALSE: 2
short = 'TRUE'
bp = 1000
m_bp = 1000
label = 'D'
valid_sample = 500
test_sample = 1000
CV = 'FALSE'
CV_k = 1
test_batchsize = 1
valid_Datanum = 62655

# model
embedding = ['one_hot']
if embedding[0] == 'one_hot':
    a_embedding = 0  # 0: onehot
    d_embedding = 1  # 0: onehot, 1: all, 2: onehot+NCP, 3:onehot+DPCP, 4: NCP, 5: DPCP, 6: NCP+DPCP
    embedding = [a_embedding, d_embedding]
    amino_emb_dim = 0
    dna_emb_dim = 0
    if d_embedding in [0, 1, 2, 3]:
        amino_emb_dim += 20
        dna_emb_dim += 4

    if d_embedding in [1, 2, 4, 6]:
        dna_emb_dim += 3

    if d_embedding in [1, 3, 5, 6]:
        dna_emb_dim += 6


if embedding[0] == 'embedding':
    stride = 1
    amino_stride = 1
    mer = 5
    amino_mer = 4
    amino_emb_dim = 100
    dna_emb_dim = 50
    emb_num = 'all'

learn = 'TRUE'
pre = 'TRUE'

kernel_size, kernel_stride, a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, cnn1_1, \
conv_num, rnn2d_dim, cat_rnn, max_pool_len, output_channel, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D = initialize()


CNN = True
if CNN:
    a_kernel_Xlen = 26
    a_kernel_Ylen = amino_emb_dim
    d_kernel_Xlen = 26
    d_kernel_Ylen = dna_emb_dim
    a_kernel_size = (a_kernel_Xlen, a_kernel_Ylen)
    d_kernel_size = (d_kernel_Xlen, d_kernel_Ylen)
    kernel_size = [a_kernel_size, d_kernel_size]
    a_kernel_stride = (4, 1)
    d_kernel_stride = (4, 1)
    kernel_stride = [a_kernel_stride, d_kernel_stride]
    a_output_channel = 160
    d_output_channel = 160
    output_channel = [a_output_channel, d_output_channel]
    a_max_pool_len = 13
    d_max_pool_len = 13
    max_pool_len = [a_max_pool_len, d_max_pool_len]
    a_CNN_D = 0
    d_CNN_D = 0
    a_hiddendim = a_output_channel
    d_hiddendim = d_output_channel

biLSTM = False
if biLSTM:
    a_rnn = 'lstm'  # lstm or gru
    d_rnn = 'lstm'  # lstm or gru
    a_hiddendim = 50
    d_hiddendim = 50
    a_biLSTM_D = 0
    d_biLSTM_D = 0

# fusion=0: co_hiddendim=d_hiddendim*2, fuison=1: co_hiddendim=a_hiddendim*2,
# fusion='NULL': a_hiddendim=d_hiddendim, co_hiddendim = 2*a_hiddendim
fusion = 2 # 4:1linear(aminoを寄せる) or 1:1linear(dnaを寄せる) or 2:2linear or 3:3linear or False
if fusion:
    if fusion == 4:
        if CNN:
            co_hiddendim = d_output_channel
        if biLSTM:
            co_hiddendim = d_hiddendim * 2
    if fusion == 1:
        if CNN:
            co_hiddendim = a_output_channel
        if biLSTM:
            co_hiddendim = a_hiddendim * 2
    if fusion in [2, 3]:
        co_hiddendim = 100
    if fusion in [3]:
        co_hiddendim2 = 50

att = '2DLSTM'  # att or mean or add or 2DLSTM or RNN or Flase
if att == 'att':
    # Attention
    DA = 50  # AttentionをNeural Networkで計算する際の重み行列のサイズ
    R = 1  # Attentionは1層

if att == 'mean' or att == 'add':
    cnn1_1 = 0  # 1:　1×1conv有り
    conv_num = 10  # 1×1 convolutionのchannel数

if att == '2DLSTM' or att == 'LSTM':
    rnn2d_dim = 200
    cat_rnn = 'lstm'  # lstm or gru
    LSTM2D_D = 0
rnn = [a_rnn, d_rnn, cat_rnn]

epoch_num = 75
args = sys.argv

# NNの2層目の中間層の次元
linear2_inputDim = 50

# Dropout
Dense_D1 = 0
Dense_D2 = 0
D_p = [a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, Dense_D1, Dense_D2]

view = 'FALSE'
sigopt = 'FALSE'

if len(args) > 1:
    bp = args[1]
    stride = args[2]
    amino_stride = args[3]
    mer = args[4]
    amino_mer = args[5]
    amino_emb_dim = args[6]
    dna_emb_dim = args[7]
    a_hiddendim = args[8]
    d_hiddendim = args[9]
    cnn = args[10]
    Calc = args[11]
    conv_num = args[12]


write_list = ['bp:', str(bp), 'amino_dim:', str(amino_emb_dim), 'dna_dim', str(dna_emb_dim), 'a_hiddenDim:',
              str(a_hiddendim), 'd_hiddenDim', str(d_hiddendim), 'conv:', str(cnn1_1), 'convNum:', str(conv_num)]
f = open('Log.txt', 'a')
f.write('\t'.join(write_list))
f.write('\n')
f.close()


if embedding[0] == 'embedding':
    # filename
    if emb_num == 'all':
        amino_dict = '../embedding/all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                     '_' + str(amino_stride) + '_amino_dict.pickle'
        dna_dict = '../embedding/all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(
            stride) + '_Dna2vec_dict.pickle'
        amino_preVec = '../embedding/all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                       '_' + str(amino_stride) + '_Aword2vec.gensim.model'
        dna_preVec = '../embedding/all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(
            stride) + '_Dna2vec.pickle'

    else:
        amino_dict = '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                     '_' + str(amino_stride) + '_amino_dict.pickle'
        dna_dict = '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_dna_dict.pickle'
        amino_preVec = '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                       '_' + str(amino_stride) + '_Aword2vec.gensim.model'
        dna_preVec = '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dword2vec.gensim.model'

if device2 == 'cpu':
    ## パラメータの設定 ##
    filename = '../data/ans.pickle'
    miss_datafile = '../data/cpu_2D_miss.pickle'
    TF_bunpu = '../TF_group'
    TF_bunpu = '../TF_cpubunpu'
    train_batchsize = 2
    valid_batchsize = 1
    ####################
else:
    filename = '../data/ans.pickle'
    miss_datafile = '../data/2D_miss.pickle'
    TF_bunpu = '../data/TF_bunpu'
    train_batchsize = 512
    valid_batchsize = 512
    ####################

os.makedirs('./result', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)

m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM]
param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p, kernel_size, kernel_stride, output_channel, max_pool_len, DA, R]
data = [device2, dataset, filename, miss_datafile, TF_bunpu, order, label, valid_sample,
        test_sample, valid_Datanum, m_bp, data_dir]
CV_para = [CV, CV_k]
batch_size = [train_batchsize, valid_batchsize, test_batchsize]

## データファイルの作成 ##
# そもそもデータが全く用意されていない
if not os.path.isfile('../data/' + str(data_dir) + '/test.pickle'):
    make_Data(data)
# train, valid, testは用意されているが交差検証用のデータが用意されていない
if CV == 'TRUE' and not os.path.exists('../data/' + str(data_dir) + '/' + str(CV_k) + 'CV'):
    cross_validation(CV_k, data_dir)

if embedding[0] == 'embeddhing':
    dict = [amino_dict, dna_dict, amino_preVec, dna_preVec, amino_mer, mer,
            amino_emb_dim, dna_emb_dim, amino_stride, stride, 0, 0]
    # k-merの辞書がない
    if not os.path.isfile(amino_dict):
        dict[10] += 1
    if not os.path.isfile(dna_dict):
        dict[11] += 1

    os.makedirs('../embedding', exist_ok=True)
    train_vector = word_embedding(data, dict)
    train_vector.train_wordVec()
    
#################################

# 値の初期化
amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim = 0, 0, 0, 0, 0

## モデルの定義 ##
if len(embedding) > 1:
    if embedding[0] in [0]:
        amino = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        if not os.path.isfile(
                '../data/' + str(data_dir) + '/dict/' + str(species) + '_' + str(embedding[0]) + 'Aonehot.pickle'):
            make_oneHot(filename, species)
        with open('../data/pickle/dict/' + str(species) + '_' + str(embedding[0]) + 'Aonehot.pickle', 'rb') as f:
            amino_dict = pickle.load(f)

    if embedding[1] in [0, 1, 2, 3, 4, 5, 6]:
        dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        ans = {'0': 0, '1': 1}
        if not os.path.isfile(
                '../data/' + str(data_dir) + '/dict/' + str(species) + '_' + str(embedding[1]) + 'Donehot.pickle'):
            make_oneHot(filename, species)
        with open('../data/pickle/dict/' + str(species) + '_' + str(embedding[1]) + 'Donehot.pickle', 'rb') as f:
            dna_dict = pickle.load(f)

        vector = convert(amino_dict, dna_dict)
        # アミノ酸配列の種類数
        A_emb_dim = amino_emb_dim
        # 塩基配列の種類数
        D_emb_dim = dna_emb_dim

if embedding[0] == 'embedding':
    with open(amino_dict, 'rb') as f:
        amino_dict = pickle.load(f)
    with open(dna_dict, 'rb') as f:
        dna_dict = pickle.load(f)

    if pre != 'TRUE':
        A_emb_dim = amino_emb_dim
        D_emb_dim = dna_emb_dim
        Avocab_size = len(amino_dict) + 1
        Dvocab_size = len(dna_dict) + 1
    else:
        amino_word = Word2Vec.load(amino_preVec)
        amino_weights = amino_word.syn1neg
        A_emb_dim = amino_weights.shape[1]
        Avocab_size = amino_weights.shape[0] + 1
        amino_weights = torch.from_numpy(amino_weights)
        zero = torch.zeros([1, A_emb_dim])
        amino_weights = torch.cat((zero, amino_weights), dim=0)
        if emb_num == 'all':
            with open(dna_preVec, 'rb') as f:
                dna_weights = pickle.load(f)
        else:
            dna_word = Word2Vec.load(dna_preVec)
            dna_weights = dna_word.syn1neg
        D_emb_dim = dna_weights.shape[1]
        Dvocab_size = dna_weights.shape[0] + 1
        zero = torch.zeros([1, D_emb_dim])
        dna_weights = torch.from_numpy(dna_weights).to(torch.float32)
        dna_weights = torch.cat((zero, dna_weights), dim=0)

    vector = Embedding(amino_dict, dna_dict, stride, mer, amino_stride, amino_mer)
# 配列の次元数
emb_dim = [A_emb_dim, D_emb_dim]

# モデル宣言
model = myModel(m_para, batch_size, DA, R, amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim, param)

print('model sengen')
# 損失関数はクロスエントロピー誤差を使う(シグモイド関数にはこれがオススメらしい)
lossFn = torch.nn.functional.binary_cross_entropy
# 最適化の手法はAdamを使う(適当)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
torch.save({'epoch': 0, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint/init_checkpoint.pt')


# training
model.to(device)

if CV != 'TRUE':  # 交差検証を行わない場合、パラメータの変え忘れに対応
    CV_k = 1

cv_trainLoss, cv_trainAC, cv_validLoss, cv_validAC = [], [], [], []
#for i in [3]:
for i in range(CV_k):
    checkpoint = torch.load(check, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1

    train_dataset, valid_dataset, iter = data_load(CV_para, short, bp, data_dir, i, batch_size, amino_dict, dna_dict)

    os.makedirs('./checkpoint/' + str(i) + 'checkpoint', exist_ok=True)
    losses, training_accuracies, valid_losses, valid_accuracies = train_model(vector, train_dataset, valid_dataset,
                                                                          batch_size, model, lossFn, optimizer,
                                                                          epoch_num, iter, device, embedding,
                                                                          epoch, sigopt, i)
    cv_trainLoss.append(losses)
    cv_trainAC.append(training_accuracies)
    cv_validLoss.append(valid_losses)
    cv_validAC.append(valid_accuracies)
    print(cv_validLoss)

if sigopt != 'TRUE':
    # cross validationした中でvalidationのaccuracyが最大の時を取得する
    cv_validAC = np.array(cv_validAC)
    idx = np.unravel_index(np.argmax(cv_validAC), cv_validAC.shape)
    c = idx[0]  # 何回目のcrossか
    max_index = idx[1]  # 何エポック目か

    #max_value = max(valid_accuracies)
    #min_index = valid_accuracies.index(max_value)
    #min_value = min(valid_losses)
    #min_index = valid_losses.index(min_value)
    #print(min_index + 1)
    check = './checkpoint/' + str(c) + 'checkpoint/checkpoint' + str(max_index + 1) + '.pt'
    checkpoint = torch.load(check, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
    os.makedirs('./result/Details', exist_ok=True)

    test, test_dataset, test_iter = test_dataLoad(short, bp, data_dir, batch_size, amino_dict, dna_dict)
    # save_data(epoch_num, losses, training_accuracies, valid_losses, valid_accuracies)
    TPs, FPs, FNs, TNs = test_model(vector, test_dataset, test_batchsize, model, test_iter, device, embedding, test)

    test_TPs = test.iloc[TPs, [0, 1, 2, 3, 4, 5, 6]]
    test_TPs.to_csv('result/Details/TPs', sep='\t', index=False)
    test_FPs = test.iloc[FPs, [0, 1, 2, 3, 4, 5, 6]]
    test_FPs.to_csv('result/Details/FPs', sep='\t', index=False)
    test_FNs = test.iloc[FNs, [0, 1, 2, 3, 4, 5, 6]]
    test_FNs.to_csv('result/Details/FNs', sep='\t', index=False)
    test_TNs = test.iloc[TNs, [0, 1, 2, 3, 4, 5, 6]]
    test_TNs.to_csv('result/Details/TNs', sep='\t', index=False)

else:
    valid_losses = sorted(valid_losses)
    print(-1 * valid_losses[0])



