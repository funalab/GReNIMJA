import optuna
import torch.nn.functional
import pandas as pd
import sqlite3
import numpy as np
import pickle
from func import convert, Embedding, initialize
from layers import myModel
import torch.nn.functional
from train import train_model
import os
from gensim.models.word2vec import Word2Vec
import torch.nn as nn
from makeData import make_Data, cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from Dataloadar import data_load, test_dataLoad
from embedding import word_embedding



import torch
import torch.nn.functional
from func import NewsDataset, ans_one_hot, pad_collate
from torch.utils.data import DataLoader




def objective(trial):
    CNN = True
    biLSTM = False
    fusion = 2
    att = '2DLSTM'

    a_kernel_Xlen, d_kernel_strideX, \
    a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, \
    cnn1_1, conv_num, rnn2d_dim, cat_rnn, max_pool_len, output_channel, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D = initialize()
    output_num = 0

    if CNN:
        a_kernel_Xlen = trial.suggest_int('a_kernelLen', 10, 30, 2)
        d_kernel_Xlen = trial.suggest_int('d_kernelLen', 10, 30, 2)
        a_kernel_strideX = trial.suggest_int('a_stride', 4, 6, 1)
        d_kernel_strideX = trial.suggest_int('d_stride', 4, 6, 1)
        a_output_channel = trial.suggest_int('a_channel', 80, 320, 20)
        d_output_channel = trial.suggest_int('d_channel', 80, 320, 20)
        a_max_pool_len = trial.suggest_int('a_maxPool', 12, 16, 1)
        d_max_pool_len = trial.suggest_int('a_maxPool', 12, 16, 1)
        a_CNN_D = trial.suggest_uniform('a_CNN_Drate', 0.0, 0.5)
        d_CNN_D = trial.suggest_uniform('d_CNN_Drate', 0.0, 0.5)

    if biLSTM:
        a_rnn = trial.suggest_categorical('a_rnn', ['lstm', 'gru'])
        d_rnn = trial.suggest_categorical('d_rnn', ['lstm', 'gru'])
        a_hiddendim = trial.suggest_int('a_hiddendim', 30, 100, 10)
        d_hiddendim = trial.suggest_int('d_hiddendim', 30, 100, 10)
        #a_biLSTM_D = trial.suggest_uniform('a_biLSTM_Drate', 0.0, 0.8)
        #d_biLSTM_D = trial.suggest_uniform('d_biLSTM_Drate', 0.0, 0.8)

    # fusion=0: co_hiddendim=d_hiddendim*2, fuison=1: co_hiddendim=a_hiddendim*2,
    # fusion='NULL': a_hiddendim=d_hiddendim, co_hiddendim = 2*a_hiddendim
    if fusion:
        if fusion in [2, 3]:
            co_hiddendim = trial.suggest_int('co_hiddendim', 30, 200, 10)
        if fusion in [3]:
            co_hiddendim2 = trial.suggest_int('co_hiddendim2', 30, 100, 10)

    if att == 'mean' or att == 'add' or att == '2DLSTM' or att == 'RNN':
        cos = 1

    if att == 'att':
        # Attention
        DA = trial.suggest_int('DA', 10, 100, 10)

    if att == '2DLSTM' or att == 'LSTM':
        rnn2d_dim = trial.suggest_int('rnn2d_dim', 50, 300, 10)
        LSTM2D_D = trial.suggest_uniform('LSTM2D_Drate', 0.0, 0.5)

    # NNの2層目の中間層の次元
    linear2_inputDim = trial.suggest_int('linear2_inputDim', 30, 200, 10)

    # Dropout
    Dense_D1 = 0.0
    Dense_D2 = 0.0

    # Int parameter
    bp = 1000

    #amino_mer = trial.suggest_int('A_mer', 3, 6)
    #mer = trial.suggest_int('D_mer', 3, 6)
    #amino_emb_dim = trial.suggest_int('A_dim', 50, 200)
    #dna_emb_dim = trial.suggest_int('D_dim', 50, 200)
    #A_stride = trial.suggest_int('A_stride', 1, 6)
    #D_stride = trial.suggest_int('D_stride', 1, 6)

    # Uniform parameter
    #dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # Discrete-uniform parameter
    #drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)

    ##############################################

    # device
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
    m_bp = 1000
    label = 'D'
    valid_sample = 500
    test_sample = 1000
    CV = 'TRUE'
    CV_k = 1
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
    learn = 'TRUE'
    pre = 'TRUE'
    emb_num = 'all'

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
        a_biLSTM_D = 0
        d_biLSTM_D = 0

    if fusion == 0:
        if biLSTM:
            co_hiddendim = d_hiddendim * 2
        elif CNN:
            co_hiddendim = d_output_channel
    if fusion == 1:
        if biLSTM:
            co_hiddendim = a_hiddendim * 2
        elif CNN:
            co_hiddendim = a_output_channel

    if att == 'att':
        # Attention
        R = 1  # Attentionは1層

    if att == 'SPP':
        output_num = [48, 24, 12, 6, 1]

    if att == 'mean' or att == 'add':
        cnn1_1 = 0  # 1:　1×1conv有り
        conv_num = 10  # 1×1 convolutionのchannel数

    if att == '2DLSTM' or att == 'LSTM':
        cat_rnn = 'lstm'  # lstm or gru
    rnn = [a_rnn, d_rnn, cat_rnn]

    epoch_num = 20
    D_p = [a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, Dense_D1, Dense_D2]

    sigopt = 'TRUE'



    write_list = ['bp:', str(bp), 'a_kernelX:', str(a_kernel_Xlen), 'd_kernelX:', str(d_kernel_Xlen), 'a_K_stride:',
                  str(a_kernel_strideX), 'd_K_stride:', str(d_kernel_strideX),
                  'a_channel:', str(a_output_channel), 'd_channel:', str(d_output_channel), 'a_Pool',
                  str(a_max_pool_len), 'd_Pool', str(d_max_pool_len), 'a_CNN_D:', str(a_CNN_D), 'd_CNN_D:',
                  str(d_CNN_D), 'a_rnn:', str(a_rnn), 'd_rnn:', str(d_rnn), 'a_hiddendim', str(a_hiddendim),
                  'd_hiddenDim', str(d_hiddendim), 'co_hiddendim', str(co_hiddendim), 'a_biLSTM_D:', str(a_biLSTM_D),
                  'b_biLSTM_D:', str(d_biLSTM_D),
                  'rnn2d_dim', str(rnn2d_dim), 'LSTM2D_D:', str(LSTM2D_D), 'liniear2:', str(linear2_inputDim),
                  'Dense_D1:', str(Dense_D1), 'Dense_D2:', str(Dense_D2), 'cos:', str(cos)]
    f = open('Log.txt', 'a')
    f.write('\t'.join(write_list))
    f.write('\n')
    f.close()

    print(write_list)

    # filename
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
    os.makedirs('./pickle', exist_ok=True)
    os.makedirs('./dataset', exist_ok=True)
    os.makedirs('./checkpoint', exist_ok=True)

    m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM, cos]
    param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p,
             kernel_size, kernel_stride, output_channel, max_pool_len, DA, R, output_num]
    data = [device2, dataset, filename, miss_datafile, TF_bunpu, order, label, valid_sample,
            test_sample, valid_Datanum, m_bp, data_dir]
    CV_para = [CV, CV_k]
    batch_size = [train_batchsize, valid_batchsize, test_batchsize]
    dict = [amino_dict, dna_dict, amino_preVec, dna_preVec, amino_mer, mer,
            amino_emb_dim, dna_emb_dim, amino_stride, stride, 0, 0]

    ## データファイルの作成 ##
    # そもそもデータが全く用意されていない
    if not os.path.isfile('../data/' + str(data_dir) + '/train.pickle'):
        make_Data(data)
    # train, valid, testは用意されているが交差検証用のデータが用意されていない
    if CV == 'TRUE' and not os.path.exists('../data/' + str(data_dir) + '/' + str(CV_k) + 'CV'):
        cross_validation(CV_k, data_dir)

    # k-merの辞書がない
    if not os.path.isfile(amino_dict):
        dict[10] += 1
    if not os.path.isfile(dna_dict):
        dict[11] += 1

    train_vector = word_embedding(data, dict)
    train_vector.train_wordVec()
    #################################

    # 値の初期化
    amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim = 0, 0, 0, 0, 0

    ## モデルの定義 ##
    if embedding == 'one_hot':
        amino = {'M': 0, 'L': 1, 'K': 2, 'F': 3, 'A': 4, 'Q': 5, 'E': 6, 'T': 7, 'P': 8, 'V': 9, 'N': 10, 'G': 11,
                 'S': 12,
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
    lossFn = nn.functional.binary_cross_entropy

    optimizer = 'Adam'
    # 最適化手法
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                     amsgrad=False)

    if optimizer == 'MomentumSGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    if optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0,
                                        initial_accumulator_value=0, eps=1e-10)

    torch.save({'epoch': 0, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint/init_checkpoint.pt')

    # training
    if device != 'cpu':
        model.to(device)

    if CV != 'TRUE':  # 交差検証を行わない場合、パラメータの変え忘れに対応
        CV_k = 1

    cv_trainLoss, cv_trainAC, cv_validLoss, cv_validAC = [], [], [], []
    for i in range(CV_k):
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

    try:
        max_value = valid_losses[24]
    except:
        max_value = valid_losses[-1]
    '''
    max_index = valid_accuracies.index(max_value)
    # min_value = min(valid_losses)
    # min_index = valid_losses.index(min_value)
    print(max_index + 1)
    check = './checkpoint/0checkpoint/checkpoint' + str(max_index + 1) + '.pt'
    checkpoint = torch.load(check, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1

    test, test_dataset, test_iter = test_dataLoad(short, bp, data_dir, batch_size)
    test_accuracy = test_model(vector, test_dataset, test_batchsize, model, test_iter, device, embedding)
    '''

    return 1 * max_value

def test_model(vector, test_datasets, batch_size, model, test_num, device, embedding):
    with torch.no_grad():
        test_TRUE = 0
        num = 0
        for aseqs, dseqs, ansss in test_datasets:
            ansss = ans_one_hot(ansss)
            amino_hoge, dna_hoge = vector.convert_vector(aseqs, dseqs)
            dataset_test = NewsDataset(amino_hoge, dna_hoge, ansss)
            test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True, collate_fn=pad_collate)
            for aseq, dseq, ans in test_dataset:
                anss = []
                ans = ans.numpy()
                for index in range(len(ans)):
                    if ans[index][0] == 1:
                        anss.append(0)
                    else:
                        anss.append(1)
                ans = torch.tensor(anss)

                if device != 'cpu':
                    aseq = aseq.to(device)
                    dseq = dseq.to(device)
                    ans = ans.to(device)

                if embedding == 'embedding':
                    aseq = model.amino_embed(aseq)
                    dseq = model.dna_embed(dseq)

                aminos, amino_att = model.amino_forward(aseq)
                dnas, dna_att = model.dna_forward(dseq)
                batch_sigmoid, _, _, _ = model.Integration(aminos, dnas)
                batch_sigmoid = torch.reshape(batch_sigmoid, (-1, 2))
                _, pre = torch.max(batch_sigmoid, 1)

                n = torch.add(pre, ans)
                # 正解した個数
                TRUE = len(n[torch.where(n == 0)]) + len(n[torch.where(n == 2)])
                test_TRUE += TRUE
                num += 1

        test_accuracy = test_TRUE / test_num
        print("test_accuracy", test_accuracy)

        write_list = [str(test_accuracy)]
        f = open('result/testresult.txt', 'a')
        f.write('\t'.join(write_list))
        f.write('\n')
        f.close()

        return test_accuracy


def main():
    # 定義
    study_name = 'example-study'
    study = optuna.study.create_study(study_name=study_name,
                                      storage='sqlite:///./optuna_study.db',
                                      load_if_exists=True)
    # 最適化の実行
    study.optimize(objective, n_trials=40)

    # 最適化結果の確認
    dbname = "optuna_study.db"
    with sqlite3.connect(dbname) as conn:
        df = pd.read_sql("SELECT * FROM trial_params;", conn)

    values = [each.value for each in study.trials]
    values = list(filter(None, values))
    trial_id = values.index(max(values))
    best_values = [np.min(values[:k + 1]) for k in range(len(values))]

    print(values)  # valueの遷移の過程
    print(best_values)  # 最適valueの過程
    print(df[df['trial_id'] == trial_id])  # 最適だった時のパラメータの表示



if __name__ == '__main__':
    main()
