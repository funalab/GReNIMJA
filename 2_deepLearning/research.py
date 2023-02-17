import pickle
from func import convert, Embedding, initialize
from layers import myModel
from torch.utils.data import DataLoader
from view_test import test_model
import torch
import os
from gensim.models.word2vec import Word2Vec
from func import NewsDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, \
    f1_score, accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn.functional
from func import NewsDataset, ans_one_hot, attention_view, pad_collate
from torch.utils.data import DataLoader
from test import Calculation
import numpy as np

def main():
    check_dir = '/Users/okubo/m1okubo/result/kekka/2_random_mean/checkpoint/0checkpoint/'
    dataNum = 10
    dir = r'../result/data/random/'

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # device
    device = torch.device('cpu')

    # どのモデルを読み込むか
    check = 'checkpoint47.pt'

    CNN = False
    biLSTM = True
    att = 'mean'  # att or mean or add or 2DLSTM or RNN or Flase or SPP or MBT
    fusion = 2  # 4:1linear(aminoを寄せる) or 1:1linear(dnaを寄せる) or 2:2linear or 3:3linear or False


    # dataset
    dataset = 'achoice'
    bp = 1000

    # model
    embedding = 'embedding'
    stride = 1
    amino_stride = 1
    mer = 5
    amino_mer = 4
    amino_emb_dim = 100
    dna_emb_dim = 50
    # embedding = 'one_hot'
    learn = 'TRUE'
    pre = 'TRUE'
    emb_num = 'all'
    batch_size = 1

    a_kernel_Xlen, d_kernel_Xlen, a_kernel_strideX, d_kernel_strideX, a_output_channel, d_output_channel, \
    a_max_pool_len, d_max_pool_len, a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, \
    cnn1_1, conv_num, rnn2d_dim, cat_rnn, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, cos, output_num = initialize()

    if CNN:
        a_kernel_Xlen = 26
        d_kernel_Xlen = 26
        a_kernel_strideX = 4
        d_kernel_strideX = 4
        a_output_channel = 160
        d_output_channel = 160
        a_max_pool_len = 13
        d_max_pool_len = 13
        a_CNN_D = 0
        d_CNN_D = 0
        # a_hiddendim = a_output_channel
        # d_hiddendim = d_output_channel

    a_kernel_Ylen = amino_emb_dim
    d_kernel_Ylen = dna_emb_dim
    a_kernel_size = (a_kernel_Xlen, a_kernel_Ylen)
    d_kernel_size = (d_kernel_Xlen, d_kernel_Ylen)
    kernel_size = [a_kernel_size, d_kernel_size]
    a_kernel_stride = (a_kernel_strideX, 1)
    d_kernel_stride = (d_kernel_strideX, 1)
    kernel_stride = [a_kernel_stride, d_kernel_stride]
    output_channel = [a_output_channel, d_output_channel]
    max_pool_len = [a_max_pool_len, d_max_pool_len]

    if biLSTM:
        a_rnn = 'lstm'  # lstm or gru
        d_rnn = 'lstm'  # lstm or gru
        a_hiddendim = 50
        d_hiddendim = 50
        a_biLSTM_D = 0
        d_biLSTM_D = 0

    # fusion=0: co_hiddendim=d_hiddendim*2, fuison=1: co_hiddendim=a_hiddendim*2,
    # fusion='NULL': a_hiddendim=d_hiddendim, co_hiddendim = 2*a_hiddendim
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

    if att == 'mean' or att == 'add' or att == '2DLSTM' or att == 'RNN':
        cos = 1  # 0: 内積 1:cos類似度
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
    if att == 'SPP':
        output_num = [12, 8, 6, 3, 1]

    rnn = [a_rnn, d_rnn, cat_rnn]

    # NNの2層目の中間層の次元
    linear2_inputDim = 50

    # Dropout
    Dense_D1 = 0
    Dense_D2 = 0
    D_p = [a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, Dense_D1, Dense_D2]

    # Attentionを可視化するか
    view = 'TRUE'
    map = 'FALSE'

    m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM, cos]
    param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p,
             kernel_size,
             kernel_stride, output_channel, max_pool_len, DA, R, output_num]

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

    # どれがTPなのかなどを出力
    os.makedirs(check_dir + 'Details', exist_ok=True)
    os.makedirs(check_dir + 'html', exist_ok=True)
    os.makedirs(check_dir + 'Details/AttentionMap', exist_ok=True)

    with open(dir + 'test.pickle', 'rb') as f:
        test = pickle.load(f)

    # balanceDNA.txtに羅列されたTFのみでテストを行う
    if dataset == 'choice':
        f = open(dir + '../data/new_gene/dataset/balanceDNA.txt', 'r')
        data = f.readlines()
        f.close()

        for i, ta in enumerate(data):
            tmp = test[test['target'] == ta.rstrip('\n')]
            test = test[test['target'] != ta.rstrip('\n')]
            if i == 0:
                balance_test = tmp
            else:
                balance_test = pd.concat([balance_test, tmp], axis=0)
        test = balance_test

    # test = test[test['target'] == 'RNF7']
    #test = test[test['tf'] == 'ZIC3']
    # TLK1

    #test = test[test['group'] == 1]
    test = test[test['target'] == 'SCRIB']
    #test = test[test['group'] > 5]
    #test = test[test['group'] < 10]
    test['ta_seq'] = test['ta_seq'].str[-bp:]

    dataNum = 'NULL'
    if dataNum != 'NULL':
        not_use, test = train_test_split(test, test_size=dataNum, shuffle=True, random_state=40, stratify=test['label'])

    test_size = len(test)
    test.reset_index(drop=True, inplace=True)
    print(test)

    test_tf, test_ta, test_aseq, test_dseq, test_ans \
        = test['tf'], test['target'], test['tf_seq'], test['ta_seq'], test['label']

    dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
    test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True)

    ## モデルの定義 ##
    amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim = 0, 0, 0, 0, 0
    if embedding == 'one_hot':
        # アミノ酸配列の種類数
        a_emb_dim = 20
        # 塩基配列の種類数
        d_emb_dim = 4
        amino = {'M': 0, 'L': 1, 'K': 2, 'F': 3, 'A': 4, 'Q': 5, 'E': 6, 'T': 7, 'P': 8, 'V': 9, 'N': 10, 'G': 11,
                 'S': 12,
                 'R': 13, 'I': 14, 'D': 15, 'Y': 16, 'C': 17, 'H': 18, 'W': 19}
        dna = {'G': 0, 'C': 1, 'T': 2, 'A': 3}
        ans = {'0': 0, '1': 1}
        vector = convert(amino, dna, ans)

    if embedding == 'embedding':
        with open(amino_dict, 'rb') as f:
            amino_dict = pickle.load(f)
        with open(dna_dict, 'rb') as f:
            dna_dict = pickle.load(f)
        if pre != 'TRUE':
            a_emb_dim = amino_emb_dim
            d_emb_dim = dna_emb_dim
            Avocab_size = len(amino_dict) + 1
            Dvocab_size = len(dna_dict) + 1
        else:
            amino_word = Word2Vec.load(amino_preVec)
            amino_weights = amino_word.syn1neg
            a_emb_dim = amino_weights.shape[1]
            Avocab_size = amino_weights.shape[0] + 1
            amino_weights = torch.from_numpy(amino_weights)
            zero = torch.zeros([1, a_emb_dim])
            amino_weights = torch.cat((zero, amino_weights), dim=0)
            if emb_num == 'all':
                with open(dna_preVec, 'rb') as f:
                    dna_weights = pickle.load(f)
            else:
                dna_word = Word2Vec.load(dna_preVec)
                dna_weights = dna_word.syn1neg
            d_emb_dim = dna_weights.shape[1]
            Dvocab_size = dna_weights.shape[0] + 1
            zero = torch.zeros([1, d_emb_dim])
            dna_weights = torch.from_numpy(dna_weights).to(torch.float32)
            dna_weights = torch.cat((zero, dna_weights), dim=0)

        vector = Embedding(amino_dict, dna_dict, stride, mer, amino_stride, amino_mer)
    emb_dim = [a_emb_dim, d_emb_dim]

    # 1epochあたりの繰り返し数
    test_iter_per_epoch = max(int(test_size / batch_size), 1)
    test_num = test_iter_per_epoch * batch_size

    # モデル宣言
    model = myModel(m_para, batch_size, DA, R, amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim, param)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    # training
    if device != 'cpu':
        model.to(device)

    if check != 'NULL':
        checkpoint = torch.load(check_dir + check, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        model.load_state_dict(torch.load(check_dir + "model_cpu.pth"))

    print(model.state_dict().keys())

    import seaborn as sns
    omomi = np.array(model.state_dict()['linear1.weight'])
    plt.figure()
    sns.heatmap(omomi)
    plt.savefig(check_dir + 'fc1_heatmap_ndarray.pdf')

    TPs, FPs, FNs, TNs, score = test_model(vector, test_dataset, batch_size, model, test_num, device, test_tf, test_ta,
                                           view, embedding, check_dir, att, stride, mer, amino_stride, amino_mer, map,
                                           epoch)

    test["score"] = score
    print(test)
    testlist = test.drop(['tf_seq', 'ta_seq'], axis=1)
    print(testlist)
    testlist.to_csv(check_dir + 'Details/testscore.txt', sep='\t', index=False)

    if view == "FALSE":
        test_TPs = test.iloc[TPs, [0, 1, 2, 3, 4, 5, 6]]
        test_TPs.to_csv(check_dir + 'Details/TPs', sep='\t', index=False)
        test_FPs = test.iloc[FPs, [0, 1, 2, 3, 4, 5, 6]]
        test_FPs.to_csv(check_dir + './Details/FPs', sep='\t', index=False)
        test_FNs = test.iloc[FNs, [0, 1, 2, 3, 4, 5, 6]]
        test_FNs.to_csv(check_dir + './Details/FNs', sep='\t', index=False)
        test_TNs = test.iloc[TNs, [0, 1, 2, 3, 4, 5, 6]]
        test_TNs.to_csv(check_dir + './Details/TNs', sep='\t', index=False)


if __name__ == '__main__':
    main()