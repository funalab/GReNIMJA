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
from makeData import make_Data, cross_validation
import matplotlib.pyplot as plt
from Dataloadar import data_load, test_dataLoad
from embedding import word_embedding
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device
device = torch.device('cuda')
device2 = 'cuda'

# initial state
check = './checkpoint/init_checkpoint.pt'

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
CV = 'TRUE'
CV_k = 5
test_batchsize = 1
valid_Datanum = 10000  # 使わない

# model
embedding = 'embedding'
stride = 1
amino_stride = 1
mer = 5
amino_mer = 4
amino_emb_dim = 100
dna_emb_dim = 50
#embedding = 'one_hot'
learn = 'TRUE'
pre = 'TRUE'
emb_num = 'all'

kernel_size, kernel_stride, a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, cnn1_1, \
conv_num, rnn2d_dim, cat_rnn, max_pool_len, output_channel, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D = initialize()


CNN = True
if CNN:
    a_kernel_Xlen = 30
    a_kernel_Ylen = amino_emb_dim
    d_kernel_Xlen = 18
    d_kernel_Ylen = dna_emb_dim
    a_kernel_size = (a_kernel_Xlen, a_kernel_Ylen)
    d_kernel_size = (d_kernel_Xlen, d_kernel_Ylen)
    kernel_size = [a_kernel_size, d_kernel_size]
    a_kernel_stride = (4, 1)
    d_kernel_stride = (4, 1)
    kernel_stride = [a_kernel_stride, d_kernel_stride]
    a_output_channel = 300
    d_output_channel = 260
    output_channel = [a_output_channel, d_output_channel]
    a_max_pool_len = 15
    d_max_pool_len = 15
    max_pool_len = [a_max_pool_len, d_max_pool_len]
    a_CNN_D = 0.458280
    d_CNN_D = 0.074344
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
        co_hiddendim = 190
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
    rnn2d_dim = 190
    cat_rnn = 'lstm'  # lstm or gru
    LSTM2D_D = 0.136456
rnn = [a_rnn, d_rnn, cat_rnn]

epoch_num = 100
args = sys.argv

# NNの2層目の中間層の次元
linear2_inputDim = 140

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


write_list = ['bp:', str(bp), 'dna_stride:', str(stride), 'amino_stride:', str(amino_stride), 'dna_mer:', str(mer),
              'amino_mer:', str(amino_mer), 'amino_dim:', str(amino_emb_dim), 'dna_dim', str(dna_emb_dim), 'a_hiddenDim:',
              str(a_hiddendim), 'd_hiddenDim', str(d_hiddendim), 'conv:', str(cnn1_1), 'convNum:', str(conv_num)]
f = open('Log.txt', 'a')
f.write('\t'.join(write_list))
f.write('\n')
f.close()

# filename
if emb_num == 'all':
    amino_dict = '../embedding/all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                 '_' + str(amino_stride) + '_amino_dict.pickle'
    dna_dict = '../embedding/all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dna2vec_dict.pickle'
    amino_preVec = '../embedding/all_vector/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                   '_' + str(amino_stride) + '_Aword2vec.gensim.model'
    dna_preVec = '../embedding/all_vector/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dna2vec.pickle'

else:
    amino_dict = '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                 '_' + str(amino_stride) + '_amino_dict.pickle'
    dna_dict = '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_dna_dict.pickle'
    amino_preVec = '../embedding/' + str(amino_mer) + '_' + str(amino_emb_dim) + \
                   '_' + str(amino_stride) + '_Aword2vec.gensim.model'
    dna_preVec = '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(stride) + '_Dword2vec.gensim.model'

if device2 == 'cpu':
    ## パラメータの設定 ##
    filename = '../data/cpu_ans.pickle'
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
    train_batchsize = 1024
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
dict = [amino_dict, dna_dict, amino_preVec, dna_preVec, amino_mer, mer,
        amino_emb_dim, dna_emb_dim, amino_stride, stride, 0, 0]

## データファイルの作成 ##
# そもそもデータが全く用意されていない
if not os.path.isfile('../data/' + str(data_dir) + '/test.pickle'):
    make_Data(data)
# train, valid, testは用意されているが交差検証用のデータが用意されていない
if CV == 'TRUE' and not os.path.exists('../data/' + str(data_dir) + '/' + str(CV_k) + 'CV'):
    cross_validation(CV_k, data_dir)

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

for i in range(0, 1):
    print(i)
    checkpoint = torch.load(check, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1

    train_dataset, valid_dataset, iter = data_load(CV_para, short, bp, data_dir, i, batch_size)

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

    test, test_dataset, test_iter = test_dataLoad(short, bp, data_dir, batch_size)
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



