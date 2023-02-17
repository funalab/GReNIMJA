import torch
import torch.nn.functional
from func import ans_one_hot, pad_collate, initialize
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from func import convert, NewsDataset, Embedding
from layers import myModel
import torch.nn.functional
from torch.utils.data import DataLoader
import os
from gensim.models.word2vec import Word2Vec
from Dataloadar import test_dataLoad


class Calculation:
    def __init__(self, df, pres, answers, score):
        #self.test = df
        self.test = df[:len(pres)]
        self.pres = pres
        #self.answers = answers
        self.answers = answers[:len(pres)]
        self.score = score

    def get_auroc(self, preds, obs):  # AUROCを計算する（figは作成しない）
        fpr, tpr, thresholds = roc_curve(obs, preds, drop_intermediate=False)
        auroc = auc(fpr, tpr)
        return auroc

    def get_aupr(self, preds, obs):  # AUPRを計算する（figは作成しない）
        precision, recall, thresholds = precision_recall_curve(obs, preds)
        aupr = auc(recall, precision)
        return aupr

    def AUROC(self):  # ROC曲線を作成しAUROCを求める
        fpr, tpr, thresholds = roc_curve(self.answers, self.score)
        fig = plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label='roc curve (AUC = %0.4f)' % auc(fpr, tpr))
        plt.legend()
        plt.xlabel('false positive rate(FPR)')
        plt.ylabel('true positive rate(TPR)')
        plt.grid()
        fig.savefig('./result/roc_curve.pdf')
        return auc(fpr, tpr)

    def AUPR(self):  # PR曲線を作成しAUPRを求める
        precision, recall, thresholds = precision_recall_curve(self.answers, self.score)
        fig = plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, label='precision_recall_curve (AUC = %0.4f)' % auc(recall, precision))
        plt.legend()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()
        fig.savefig('./result/pr_curve.pdf')
        return auc(recall, precision)

    def each_TF(self):
        self.test['pred'] = self.pres
        self.test['score'] = self.score
        TF_list = self.test.drop_duplicates(subset='tf')['tf']
        auroc_list = []
        aupr_list = []
        acc_list = []
        for tf in TF_list:
            tmp = self.test[self.test['tf'] == tf]
            pred_df, ans_df, score_df = tmp['pred'], tmp['label'], tmp['score']
            auroc = round(self.get_auroc(score_df, ans_df), 5)
            aupr = round(self.get_aupr(score_df, ans_df), 5)
            acc = round(accuracy_score(pred_df, ans_df), 5)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            acc_list.append(acc)
        print("Averaged AUROC:", np.nanmean(auroc_list))
        print("Averaged AUPR:", np.nanmean(aupr_list))
        print("Averaged ACC:", np.nanmean(acc_list))
        TF_list = pd.DataFrame(TF_list)
        TF_list['acc'], TF_list['AUROC'], TF_list['AUPR'] = acc_list, auroc_list, aupr_list
        TF_list.to_csv('./result/each_TFresult.txt', sep='\t', index=False)
        return np.nanmean(acc_list), np.nanmean(auroc_list), np.nanmean(aupr_list)

    def calc(self):
        unq = np.array([x + 2 * y for x, y in zip(self.pres, self.answers)])
        cm = confusion_matrix(self.pres, self.answers)
        TN, FP, FN, TP = cm.flatten()
        TPs = np.array(np.where(unq == 3)).tolist()[0]
        FPs = np.array(np.where(unq == 1)).tolist()[0]
        TNs = np.array(np.where(unq == 0)).tolist()[0]
        FNs = np.array(np.where(unq == 2)).tolist()[0]
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F_measure = 2 * Recall * Precision / (Recall + Precision)
        # 全体のACC, AUROC, AUPRを計算する
        test_accuracy = accuracy_score(self.pres, self.answers)
        auroc = self.AUROC()
        aupr = self.AUPR()

        # 転写因子ごとのACC, AUROC, AUPRを計算する
        ave_acc, ave_auroc, ave_aupr = self.each_TF()

        print("test_accuracy", test_accuracy)
        print("TP", TP, "\t", "FP", FP, "\t", "FN", FN, "\t", "TN", TN, "\t", "Recall", Recall, "\t", "Precision",
              Precision, "\t", "F_measure", F_measure)

        write_list = ['Acc', str(test_accuracy), 'TP', str(TP), 'FP', str(FP), 'FN', str(FN), 'TN', str(TN),
                      'Recall', str(Recall), 'Precision', str(Precision), 'F_measure', str(F_measure),
                      'AUROC', str(auroc), 'AUPR', str(aupr), 'ave_acc', str(ave_acc), 'ave_roc', str(ave_auroc),
                      'ave_pr', str(ave_aupr)]

        f = open('result/testresult.txt', 'a')
        f.write('\t'.join(write_list))
        f.write('\n')
        f.close()
        print('\t'.join(write_list))

        self.test["score"] = self.score
        testlist = self.test.drop(['tf_seq', 'ta_seq'], axis=1)
        testlist.to_csv('./result/testscore.txt', sep='\t', index=False)

        return TPs, FPs, FNs, TNs


def test_model(vector, test_datasets, batch_size, model, test_num, device, embedding, test):
    with torch.no_grad():
        test_TRUE = 0
        num = 0
        score = []
        pres = torch.empty(1)
        pres = pres.to(device)
        answers = np.array(test['label'].tolist())

        for aseqs, dseqs, ansss in test_datasets:
            ansss = ans_one_hot(ansss)
            amino_hoge, dna_hoge = vector.convert_vector(aseqs, dseqs)
            dataset_test = NewsDataset(amino_hoge, dna_hoge, ansss)
            test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True,
                                      collate_fn=pad_collate)
            for aseq, dseq, ans in test_dataset:
                anss = []
                ans = ans.numpy()
                for index in range(len(ans)):
                    if ans[index][0] == 1:
                        anss.append(0)
                    else:
                        anss.append(1)
                ans = torch.tensor(anss)

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
                pres = torch.cat((pres, pre), 0)
                # score.extend(batch_sigmoid[:, 1])

                for i in range(len(pre)):
                    score.append(batch_sigmoid[i, 1].item())
                n = torch.add(pre, ans)
                # 正解した個数
                TRUE = len(n[torch.where(n == 0)]) + len(n[torch.where(n == 2)])
                test_TRUE += TRUE
                num += 1
        pres = pres[1:]
        pres = pres.to('cpu')
        pres = np.array(pres)
        calc_acc = Calculation(test, pres, answers, score)
        TPs, FPs, FNs, TNs = calc_acc.calc()

        return TPs, FPs, FNs, TNs


def main():
    check_dir = './checkpoint/1checkpoint/'
    # どのモデルを読み込むか
    check = 'checkpoint96.pt'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # device
    device = torch.device('cuda')
    device2 = 'cuda'

    species = 'human'

    # dataset
    data_dir = 'pickle'
    short = 'TRUE'
    bp = 1000
    test_batchsize = 1
    choice = False

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

    biLSTM = False
    if biLSTM:
        a_rnn = 'lstm'  # lstm or gru
        d_rnn = 'lstm'  # lstm or gru
        a_hiddendim = 50
        d_hiddendim = 50
        a_biLSTM_D = 0.5
        d_biLSTM_D = 0.5

    # fusion=0: co_hiddendim=d_hiddendim*2, fuison=1: co_hiddendim=a_hiddendim*2,
    # fusion='NULL': a_hiddendim=d_hiddendim, co_hiddendim = 2*a_hiddendim
    fusion = 2  # 0:1linear(aminoを寄せる) or 1:1linear(dnaを寄せる) or 2:2linear or 3:3linear or False
    if fusion in [0, 1, 2, 3]:
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

    # NNの2層目の中間層の次元
    linear2_inputDim = 140

    # Dropout
    Dense_D1 = 0
    Dense_D2 = 0
    D_p = [a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, Dense_D1, Dense_D2]

    view = 'FALSE'  # Attention mapを可視化するか

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
        train_batchsize = 200
        valid_batchsize = 1
        ####################
    else:
        train_batchsize = 512
        valid_batchsize = 128
        ####################

    os.makedirs('./result', exist_ok=True)
    os.makedirs('./checkpoint', exist_ok=True)

    m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM]
    param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p,
             kernel_size, kernel_stride, output_channel, max_pool_len, DA, R]
    batch_size = [train_batchsize, valid_batchsize, test_batchsize]

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
    # 損失関数はクロスエントロピー誤差を使う(シグモイド関数にはこれがオススメらしい)
    lossFn = torch.nn.functional.binary_cross_entropy
    # 最適化の手法はAdamを使う(適当)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    checkpoint = torch.load(check_dir + check, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # test
    model.to(device)

    test, test_dataset, test_iter = test_dataLoad(short, bp, data_dir, batch_size, species)

    # balanceDNA.txtに羅列されたTFのみでテストを行う
    if choice:
        f = open(check_dir + 'balanceDNA.txt', 'r')
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

    TPs, FPs, FNs, TNs = test_model(vector, test_dataset, test_batchsize, model, test_iter, device, embedding, test)

    os.makedirs('./result/Details', exist_ok=True)
    test_TPs = test.iloc[TPs, [0, 1, 2, 3, 4, 5, 6]]
    test_TPs.to_csv('result/Details/TPs', sep='\t', index=False)
    test_FPs = test.iloc[FPs, [0, 1, 2, 3, 4, 5, 6]]
    test_FPs.to_csv('result/Details/FPs', sep='\t', index=False)
    test_FNs = test.iloc[FNs, [0, 1, 2, 3, 4, 5, 6]]
    test_FNs.to_csv('result/Details/FNs', sep='\t', index=False)
    test_TNs = test.iloc[TNs, [0, 1, 2, 3, 4, 5, 6]]
    test_TNs.to_csv('result/Details/TNs', sep='\t', index=False)


if __name__ == '__main__':
    main()
