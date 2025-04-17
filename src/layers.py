import torch
import torch.nn as nn
import torch.nn.functional as F
from func import min_max, z_score, z_score2, min_max3
import pandas as pd
from lstm2d import LSTM2d
#from MDRNN._layers.mdlstm import MDLSTM
import numpy as np

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

class myModel(nn.Module):

    def __init__(self, m_para, batch_size, da, r, amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim, param):
        super(myModel, self).__init__()

        '''
        m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM]
        param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p,
                 kernel_size, kernel_stride, output_channel, max_pool_len, DA, R]
        '''

        if m_para[0] == 'embedding':
            self.A_embeddings = nn.Embedding(Avocab_size, emb_dim[0], padding_idx=0)
            self.D_embeddings = nn.Embedding(Dvocab_size, emb_dim[1], padding_idx=0)
            self.pre = m_para[1]
            if self.pre == 'TRUE':
                self.A_embeddings.weight = nn.Parameter(amino_weights)
                self.D_embeddings.weight = nn.Parameter(dna_weights)
                if m_para[3] == 'FALSE':
                    self.A_embeddings.weight.requires_grad = False
                    self.D_embeddings.weight.requires_grad = False


        self.A_vocab_size = Avocab_size
        self.D_vocab_size = Dvocab_size
        self.emb_dim = emb_dim
        self.rnn = m_para[5]
        self.att = m_para[2]
        self.fusion = m_para[6]
        self.conv = m_para[4]
        self.convNum = param[6]
        self.device = m_para[7]
        self.CNN = m_para[8]
        self.biLSTM = m_para[9]
        self.D_p = param[7]

        if self.CNN:
            self.a_kernel_size = param[8][0]
            self.a_kernel_stride = param[9][0]
            self.d_kernel_size = param[8][1]
            self.d_kernel_stride = param[9][1]
            self.a_output_channel = param[10][0]
            self.d_output_channel = param[10][1]
            self.a_max_pool_len = param[11][0]
            self.d_max_pool_len = param[11][1]
            self.a_conv = nn.Conv2d(in_channels=1, out_channels=self.a_output_channel, kernel_size=self.a_kernel_size,
                                    stride=self.a_kernel_stride, padding=(0, 0))
            self.d_conv = nn.Conv2d(in_channels=1, out_channels=self.d_output_channel, kernel_size=self.d_kernel_size,
                                    stride=self.d_kernel_stride, padding=(0, 0))
            self.a_maxPlooing = nn.MaxPool2d(kernel_size=(self.a_max_pool_len, 1), stride=(self.a_max_pool_len, 1))
            self.d_maxPlooing = nn.MaxPool2d(kernel_size=(self.d_max_pool_len, 1), stride=(self.d_max_pool_len, 1))

            self.a_hiddendim = int(self.a_output_channel / 2)
            self.d_hiddendim = int(self.d_output_channel / 2)
            self.co_hiddendim = self.a_output_channel
            self.linear1_input = self.a_output_channel
            self.a_CNN_Dropout = nn.Dropout(self.D_p[0])
            self.d_CNN_Dropout = nn.Dropout(self.D_p[1])

        if self.biLSTM:
            self.a_hiddendim = param[0]
            self.d_hiddendim = param[1]
            if self.CNN:
                self.a_rnn_inputSize = self.a_output_channel
                self.d_rnn_inputSize = self.d_output_channel
            else:
                self.a_rnn_inputSize = self.emb_dim[0]
                self.d_rnn_inputSize = self.emb_dim[1]
            if self.rnn[0] == 'lstm':
                # LSTMの層(双方向でバッチサイズが0次元目)
                self.amino_rnn = nn.LSTM(input_size=self.a_rnn_inputSize, hidden_size=self.a_hiddendim,
                                         batch_first=True, bidirectional=True, dropout=self.D_p[2])
            elif self.rnn[0] == 'gru':
                self.amino_rnn = nn.GRU(input_size=self.a_rnn_inputSize, hidden_size=self.a_hiddendim,
                                        batch_first=True, bidirectional=True, dropout=self.D_p[3])
            if self.rnn[1] == 'lstm':
                # LSTMの層(双方向でバッチサイズが0次元目)
                self.dna_rnn = nn.LSTM(input_size=self.d_rnn_inputSize, hidden_size=self.d_hiddendim,
                                       batch_first=True, bidirectional=True, dropout=self.D_p[2])
            elif self.rnn[1] == 'gru':
                self.dna_rnn = nn.GRU(input_size=self.d_rnn_inputSize, hidden_size=self.d_hiddendim,
                                      batch_first=True, bidirectional=True, dropout=self.D_p[2])

        self.linear1_input = (self.a_hiddendim + self.d_hiddendim) * 2

        if self.att:
            if self.att == 'att':
                # amino_Attention
                self.amino_att = nn.Sequential(
                    # Bidirectionalなので各隠れ層のベクトルの次元は２倍のサイズ。
                    nn.Linear(self.a_hiddendim * 2, da),
                    nn.Tanh(),
                    nn.Linear(da, r)
                )
                # dna_Attention
                self.dna_att = nn.Sequential(
                    # Bidirectionalなので各隠れ層のベクトルの次元は２倍のサイズ。
                    nn.Linear(self.d_hiddendim * 2, da),
                    nn.Tanh(),
                    nn.Linear(da, r)
                )
            else:
                if self.conv == 1:
                    # 畳み込み
                    self.conv1_1_1 = nn.Conv2d(in_channels=1, out_channels=self.convNum, kernel_size=(1, 1),
                                               stride=(1, 1), padding=(0, 0))
                    self.conv1_1_2 = nn.Conv2d(in_channels=self.convNum, out_channels=1, kernel_size=(1, 1),
                                               stride=(1, 1), padding=(0, 0))

                if self.fusion:
                    self.co_hiddendim = param[2]
                    if self.fusion in [4, 2, 3]:
                        self.A_fusion = nn.Conv2d(in_channels=self.a_hiddendim * 2, out_channels=self.co_hiddendim,
                                                  kernel_size=(1, 1), stride=(1, 1))
                        #self.Aco_linear = nn.Linear(in_features=self.a_hiddendim * 2, out_features=self.co_hiddendim)
                    if self.fusion in [1, 2, 3]:
                        self.D_fusion = nn.Conv2d(in_channels=self.d_hiddendim * 2, out_channels=self.co_hiddendim,
                                                  kernel_size=(1, 1), stride=(1, 1))
                        #self.Dco_linear = nn.Linear(in_features=self.d_hiddendim * 2, out_features=self.co_hiddendim)
                    if self.fusion in [4, 1, 2]:
                        self.linear1_input = self.co_hiddendim * 2
                    if self.fusion == 3:
                        self.co_hiddendim2 = param[3]
                        self.co_fusion = nn.Conv2d(in_channels=self.co_hiddendim, out_channels=self.co_hiddendim2,
                                                   kernel_size=(1, 1), stride=(1, 1))
                        #self.co_linear = nn.Linear(in_features=self.co_hiddendim, out_features=self.co_hiddendim2)

                        self.linear1_input = self.co_hiddendim2 * 2


                if self.att == '2DLSTM':
                    self.lstm2d_dim = param[5]
                    self.lstm2D = LSTM2d(future_dim=self.co_hiddendim,
                                         state_dim_2d=self.lstm2d_dim,
                                         device=self.device)
                    self.LSTM2D_D = nn.Dropout(self.D_p[4])
                    self.linear1_input = self.lstm2d_dim

        # バッチサイズ
        self.batchsize = batch_size
        # Dropout
        self.Dense_D1 = torch.nn.Dropout(p=self.D_p[5])
        self.Dense_D2 = torch.nn.Dropout(p=self.D_p[6])
        # 2層の全結合層(数はノリ)
        self.linear2_input = param[4]
        self.linear1 = nn.Linear(in_features=self.linear1_input,
                                 out_features=self.linear2_input)
        nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.linear2_input, out_features=2)
        nn.init.kaiming_uniform_(self.linear2.weight)
        # シグモイド関数
        self.sigmoid = nn.Sigmoid()
        # ReLu関数
        self.ReLu = nn.ReLU()

    def amino_embed(self, aseq):
        amino_vector = self.A_embeddings(aseq)
        return amino_vector

    def dna_embed(self, dseq):
        dna_vector = self.D_embeddings(dseq)
        return dna_vector

    # アミノ酸配列の順伝播処理
    def amino_forward(self, amino_vec):
        if self.CNN:
            amino_vec = torch.reshape(amino_vec, [-1, 1, amino_vec.size()[1], amino_vec.size()[2]])
            amino_vec = self.a_conv(amino_vec)
            amino_vec = self.ReLu(amino_vec)
            amino_vec = self.a_maxPlooing(amino_vec)
            amino_vec = self.a_CNN_Dropout(amino_vec)
            amino_vec = amino_vec[:, :, :, 0]
            amino_att, amino_features = torch.transpose(amino_vec, 1, 2), 0
            amino_vec = amino_att
        if self.biLSTM:
            amino_att, amino_features = self.amino_rnn(amino_vec)
        if self.att:
            if self.att == 'att':
                amino_attw = self.amino_att(amino_att)  # amino_attw = torch.Size([100, 1111, 1])
                amino_attw = F.softmax(amino_attw,
                                       dim=1)  # seqのうちどこに注目するかを確率で表すのでdim=1  amino_attw = torch.Size([100, 1111, 1])
                amino_rnn_out = (torch.mul(amino_att, amino_attw)).sum(
                    dim=1)  # (b, s, h) -> (b, h)  amino_features = torch.Size([100, 100])
                return amino_rnn_out, amino_attw
            else:
                return amino_att, amino_features
        else:
            # # many to oneのタスクを解きたいため、第二戻り値だけ使う。双方向なので最初の隠れ層と最後の隠れ層を連結する
            if self.rnn[0] == 'gru':
                amino_rnn_out = torch.cat([amino_features[0], amino_features[1]], dim=1)
            elif self.rnn[0] == 'lstm':
                amino_rnn_out = torch.cat([amino_features[0][0], amino_features[0][1]], dim=1)
            return amino_rnn_out, 0

    # DNA配列の順伝播処理
    def dna_forward(self, dna_vec):
        if self.CNN:
            dna_vec = torch.reshape(dna_vec, [-1, 1, dna_vec.size()[1], dna_vec.size()[2]])
            dna_vec = self.d_conv(dna_vec)
            max_index = torch.argmax(dna_vec, dim=2)
            max_index = torch.reshape(max_index, [-1, max_index.size()[1]])

            dna_vec = self.ReLu(dna_vec)
            dna_vec = self.d_maxPlooing(dna_vec)
            dna_vec = self.d_CNN_Dropout(dna_vec)
            dna_vec = dna_vec[:, :, :, 0]
            dna_att, dna_features = torch.transpose(dna_vec, 1, 2), 0
            dna_vec = dna_att
        if self.biLSTM:
            dna_att, dna_features = self.dna_rnn(dna_vec)
        if self.att:
            if self.att == 'att':
                dna_attw = self.dna_att(dna_att)
                dna_attw = F.softmax(dna_attw, dim=1)
                dna_rnn_out = torch.mul(dna_att, dna_attw).sum(dim=1)  # (b, s, h) -> (b, h)
                return dna_rnn_out, dna_attw
            else:
                return dna_att, dna_features
        else:
            # many to oneのタスクを解きたいため、第二戻り値だけ使う。
            if self.rnn[1] == 'gru':
                dna_rnn_out = torch.cat([dna_features[0], dna_features[1]], dim=1)
            elif self.rnn[1] == 'lstm':
                dna_rnn_out = torch.cat([dna_features[0][0], dna_features[0][1]], dim=1)
            return dna_rnn_out, 0

    # アミノ酸配列とDNA配列の特徴量を統合し、再びRNNに入力する(相互作用の学習)
    def Integration(self, amino, dna, save_path):
        sigmoid, amino_att, dna_attmap, dna_att = 0, 0, 0, 0
        if self.att == 'att' or self.att == False:
            feartures = torch.cat([amino, dna], dim=1)  # feartures.size() = torch.Size([batch_size, 2 * hidden_dim])

        else:
            # fusionを行う
            if self.fusion:
                if self.fusion in [4, 2, 3]:
                    amino = torch.transpose(amino, 1, 2)
                    amino = amino.unsqueeze(3)
                    amino = self.A_fusion(amino)
                    amino = amino[:, :, :, 0]
                    amino = torch.transpose(amino, 1, 2)


                if self.fusion in [1, 2, 3]:
                    dna = torch.transpose(dna, 1, 2)
                    dna = dna.unsqueeze(3)
                    dna = self.D_fusion(dna)
                    dna = dna[:, :, :, 0]
                    dna = torch.transpose(dna, 1, 2)

            a_size = amino.size()[1]
            d_size = dna.size()[1]
            feartures = torch.cat([amino, dna],
                                  dim=1)  # feartures.size() = torch.Size([batch_size, 2 * hidden_dim])


            if self.att == '2DLSTM':
                feartures = self.lstm2D.forward(dna, amino)
                feartures = self.LSTM2D_D(feartures)
                
                f = feartures.to('cpu')
                lines = []
                for r in f:
                    lines.append(str(r))
                f = open(f'{save_path}/after2DLSM_test_feature.txt', 'a')
                f.write('\n'.join(lines))
                f.write('\n')
                f.close()
                

            else:  # mean or add or RNN
                # アミノ酸の出力とDNAの出力との内積を求める(cos類似度の分子)
                dna_t = torch.transpose(dna, 1, 2)
                att = torch.matmul(amino, dna_t)

                # cos類似度のノルムを求める(cos類似度の分母)
                amino_norm = torch.norm(amino, dim=2)
                dna_norm = torch.norm(dna, dim=2)
                amino_norm = torch.repeat_interleave(amino_norm, d_size)
                amino_norm = torch.reshape(amino_norm, (-1, a_size, d_size))
                dna_norm = torch.repeat_interleave(dna_norm, a_size)
                dna_norm = torch.reshape(dna_norm, (-1, d_size, a_size))
                # cos類似度を求める
                # att = att / amino_norm / torch.transpose(dna_norm, 1, 2)
                att = torch.div(att, torch.div(amino_norm, torch.transpose(dna_norm, 1, 2)))

                '''
                dna2 = att[0].detach().numpy()
                dna2 = pd.DataFrame(dna2)
                dna2.to_csv('~/b_attention', sep='\t', header=True, index=False)
                '''

                if self.conv == 1:  # 1×1 conv
                    att = torch.reshape(att, (-1, 1, a_size, d_size))
                    att = self.conv1_1_1(att)
                    att = self.conv1_1_2(att)
                    att = torch.reshape(att, (-1, a_size, d_size))

                '''
                dna2 = att[0].detach().numpy()
                dna2 = pd.DataFrame(dna2)
                dna2.to_csv('~/a_attention', sep='\t', header=True, index=False)
                exit()
                '''

                if self.att == 'add':  # add
                    amino_att = F.softmax(att, dim=1)
                    amino_att = torch.cumsum(amino_att, 2)
                    amino_att = amino_att[:, :, d_size - 1]

                    dna_attmap = F.softmax(att, dim=2)
                    dna_att = torch.cumsum(dna_attmap, 1)
                    dna_att = dna_att[:, a_size - 1, :]

                elif self.att == 'mean':  # mean
                    amino_att = torch.mean(att, dim=2)
                    dna_att = torch.mean(att, dim=1)

                '''
                dna2 = att[0].detach().numpy()
                dna2 = pd.DataFrame(dna2)
                dna2.to_csv('~/a_attention', sep='\t', header=True, index=False)
                exit()
                '''
                amino_att = F.softmax(amino_att, dim=1)
                amino_att = torch.reshape(amino_att, (-1, a_size, 1))
                dna_att = F.softmax(dna_att, dim=1)
                dna_att = torch.reshape(dna_att, (-1, d_size, 1))
                amino = (torch.mul(amino, amino_att)).sum(dim=1)
                dna = (torch.mul(dna, dna_att)).sum(dim=1)
                feartures = torch.cat([amino, dna],
                                      dim=1)  # feartures.size() = torch.Size([batch_size, 2 * hidden_dim])


        sigmoid_input = self.linear1(feartures)
        sigmoid_input = self.ReLu(sigmoid_input)
        sigmoid_input = self.Dense_D1(sigmoid_input)
        sigmoid_input = self.linear2(sigmoid_input)
        sigmoid_input = self.Dense_D2(sigmoid_input)

        # sigmoid_input = self.linear3(sigmoid_input)
        # sigmoid_input = self.dropout2(sigmoid_input)
        sigmoid = F.softmax(sigmoid_input, dim=1)
        #sigmoid = self.sigmoid(sigmoid_input)
        return sigmoid, amino_att, dna_attmap, dna_att
