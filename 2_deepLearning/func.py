import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
#import seaborn as sns
#torch.manual_seed(0)
import pickle
import re
import math
import csv


def initialize():
    kernel_size = False
    kernel_stride = False
    max_pool_len = False
    output_channel = False

    a_rnn = False  # lstm or gru
    d_rnn = False  # lstm or gru
    a_hiddendim = False
    d_hiddendim = False

    # fusion=0: co_hiddendim=d_hiddendim*2, fuison=1: co_hiddendim=a_hiddendim*2,
    # fusion='NULL': a_hiddendim=d_hiddendim, co_hiddendim = 2*a_hiddendim
    co_hiddendim = False
    co_hiddendim2 = False

    # Attention
    DA = False  # AttentionをNeural Networkで計算する際の重み行列のサイズ
    R = False  # Attentionは1層

    cnn = False  # 1:　1×1conv有り
    conv_num = False  # 1×1 convolutionのchannel数

    rnn2d_dim = False
    cat_rnn = False  # lstm or gru

    a_CNN_D = False
    d_CNN_D = False
    a_biLSTM_D = False
    d_biLSTM_D = False
    LSTM2D_D = False
    return kernel_size, kernel_stride, a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, cnn, \
           conv_num, rnn2d_dim, cat_rnn, max_pool_len, output_channel, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D

def make_group(df, groups, order_num):
    TF = df['tf']
    uniTF = TF[~TF.duplicated()]

    for i in range(len(uniTF)):
        x = uniTF.iat[i]
        group = groups[groups['tf'] == x].iat[0, order_num]
        TF_num = len(TF[TF == x])
        g = np.repeat(group, TF_num)
        if i == 0:
            group_list = g
        else:
            group_list = np.concatenate([group_list, g], 0)
    group_list = pd.Series(group_list)
    df['group'] = group_list
    return df

def preprocess(amino, dna):  # アミノ酸配列とDNA配列の辞書を作る
    dna_to_id = {}
    amino_to_id = {}
    dnas = list(dna)
    aminos = list(amino)
    for dna in dnas:
        if dna not in dna_to_id:
            new_id = len(dna_to_id)
            dna_to_id[dna] = new_id

    for amino in aminos:
        if amino not in amino_to_id:
            new_id = len(amino_to_id)
            amino_to_id[amino] = new_id

    return dna_to_id, amino_to_id



class convert:  # アミノ酸配列とDNA配列をone-hotベクトルに変換する
    def __init__(self, amino_dict, dna_dict):
        self.Num_toAname = {}
        self.Num_toDname = {}

        self.a_dict = amino_dict
        self.d_dict = dna_dict

        for i, k in enumerate(self.a_dict):
            self.Num_toAname[i] = k
        for i, k in enumerate(self.d_dict):
            self.Num_toDname[i] = k

    def convert_vector(self, aseq, dseq):
        amino_hoge, dna_hoge = [], []
        for index in range(len(aseq)):
            dna_seq = dseq[index]
            print(dna_seq)
            exit()
            dna_vector = self.d_dict[self.Num_toDname[dna_seq.item()]]
            amino_seq = aseq[index]
            amino_vector = self.a_dict[self.Num_toAname[amino_seq.item()]]
            #amino_vector = torch.from_numpy(self.amino_convert_one_hot(self.amino, amino_seq).astype(np.float32))  # torch.Size([665, 20])
            amino_hoge.append(amino_vector)
            dna_hoge.append(dna_vector)
        return amino_hoge, dna_hoge


    def amino_convert_vector(self, aseq):
        amino_hoge = []
        for index in range(len(aseq)):
            amino_seq = list(aseq[index])  # iat:列番号と行番号で位置を指定してその要素を取得
            amino_vector = torch.from_numpy(self.amino_convert_one_hot(self.amino, amino_seq).astype(np.float32))  # torch.Size([665, 20])
            amino_hoge.append(amino_vector)
        return amino_hoge

    def dna_convert_vector(self, dseq):
        dna_hoge = []
        for index in range(len(dseq)):
            dna_seq = dseq[index]
            dna_vector = np.zeros((len(dna_seq), self.dim), dtype=np.float32)
            for pos in range(len(list(dna_seq))):
                dna_vector[pos, 0:4] += np.asarray(np.float32(self.One_hot[dna_seq[pos]]))
                dna_vector[pos, 4:7] += np.asarray(np.float32(self.NCP[dna_seq[pos]]))
                if pos != 0 and pos != len(list(dna_seq)) - 1:
                    dna_vector[pos, 7:13] += np.asarray(np.float32(self.DPCP[dna_seq[pos-1:pos+1]])) / 2
                    dna_vector[pos, 7:13] += np.asarray(np.float32(self.DPCP[dna_seq[pos:pos + 2]])) / 2
                elif pos == 0:
                    dna_vector[pos, 7:13] += np.asarray(np.float32(self.DPCP[dna_seq[pos:pos + 2]]))
                else:
                    dna_vector[pos, 7:13] += np.asarray(np.float32(self.DPCP[dna_seq[pos-1:pos + 1]]))
                    dna_vector = torch.from_numpy(dna_vector)
            dna_hoge.append(dna_vector)
        return dna_hoge

    def dna_convert_one_hot(self, dna, dna_seq):  # DNA配列をone-hotに変換する
        dnas = list(dna_seq)
        leng = len(dnas)
        one_hot = np.zeros((leng, 4), dtype=np.float32)
        corpus = np.array([dna[w] for w in dnas])
        for idx_0, dna_id in enumerate(corpus):
            one_hot[idx_0, dna_id] = 1

        return one_hot

    def amino_convert_one_hot(self, amino, amino_seq):  # アミノ酸配列をone-hotに変換する
        aminos = list(amino_seq)
        leng = len(aminos)
        one_hot = np.zeros((leng, 20), dtype=np.float32)
        corpus = np.array([amino[w] for w in aminos])
        for idx_0, amino_id in enumerate(corpus):
            one_hot[idx_0, amino_id] = 1
        return one_hot




def ans_one_hot(tensor):  # labelをワンホットベクトルに変換する
    one_hot = np.zeros((tensor.size()[0], 2), dtype=np.int32)
    for index in range(tensor.size()[0]):
        if tensor[index] == 0:
            one_hot[index][0] = 1
        else:
            one_hot[index][1] = 1
    one_hot = torch.from_numpy(one_hot.astype(np.int32)).clone()
    return one_hot

class Embedding():
    def __init__(self, amino_dict, dna_dict, stride, mer, amino_stride, amino_mer):
        self.amino_dict = amino_dict
        self.dna_dict = dna_dict
        self.stride = stride
        self.mer = mer
        self.amino_stride = amino_stride
        self.a_mer = amino_mer
        self.num = 0

    def convert_vector(self, aseq, dseq):
        amino_hoge, dna_hoge = [], []
        for index in range(len(aseq)):
            amino_seq = aseq[index]  # iat:列番号と行番号で位置を指定してその要素を取得
            dna_seq = dseq[index]
            amino_mer = self.k_mer(amino_seq, self.a_mer, self.amino_stride)
            amino_vector = torch.tensor(self.amino_convert_one_hot(amino_mer))

            dna_mer = self.k_mer(dna_seq, self.mer, self.stride)
            dna_vector = torch.tensor(self.dna_convert_one_hot(dna_mer))
            amino_hoge.append(amino_vector)
            dna_hoge.append(dna_vector)
        return amino_hoge, dna_hoge

    def amino_convert_vector(self, aseq, max_lens):
        amino_hoge = []
        amino_hoge_fill = []
        for index in range(len(aseq)):
            amino_seq = aseq[index]  # iat:列番号と行番号で位置を指定してその要素を取得
            amino_mer = self.k_mer(amino_seq, self.a_mer, self.amino_stride)
            amino_vector = self.amino_convert_one_hot(amino_mer)
            amino_hoge.append(amino_vector)

            amino_seq = amino_seq.zfill(max_lens)
            amino_mer = self.k_mer(amino_seq, self.a_mer, self.amino_stride)
            amino_vector = self.amino_convert_one_hot(amino_mer)
            amino_hoge_fill.append(amino_vector)
        return amino_hoge, amino_hoge_fill

    def dna_convert_vector(self, dseq):
        dna_hoge = []
        for index in range(len(dseq)):
            dna_seq = dseq[index]
            dna_mer = self.k_mer(dna_seq, self.mer, self.stride)
            dna_vector = self.dna_convert_one_hot(dna_mer)
            dna_hoge.append(dna_vector)
        return dna_hoge

    def amino_convert_one_hot(self, amino_mer):
        amino_mers = []
        for mer in amino_mer:
            try:
                amino_mers.append(self.amino_dict[mer])
                
            except:
                amino_mers.append(0)
        return amino_mers

    def dna_convert_one_hot(self, dna_mer):
        dna_mers = []
        for mer in dna_mer:
            try:
                dna_mers.append(self.dna_dict[mer])
            except:
                dna_mers.append(0)
        return dna_mers

    def k_mer(self, str, n, s):
        mer = []
        for i in range(0, len(str) - n + 1, s):
            mer.append(str[i:i + n])
        return mer

def pad_collate(batch):
    (xx, yy, zz) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, zz_pad

class NewsDataset(Dataset):
    def __init__(self, X, y, z):  # datasetの構成要素を指定
        self.X = X
        self.y = y
        self.z = z

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, xid):  # dataset[idx]で返す値を指定
        return [self.X[xid], self.y[xid], self.z[xid]]

def save_data(epoch_num, losses, training_accuracies, valid_losses, valid_accuracies):
    fig = plt.figure()
    plt.plot(np.arange(epoch_num), losses)
    plt.xlabel("epoch")  # x軸ラベル
    plt.ylabel("train_loss")  # y軸ラベル
    plt.title("Cross Entropy Error of train", fontsize=20)  # グラフタイトル
    fig.savefig("result/train_loss.pdf")

    # 交差エントロピー誤差の推移
    fig = plt.figure()
    plt.plot(np.arange(epoch_num), valid_losses)
    plt.xlabel("epoch")  # x軸ラベル
    plt.ylabel("valid_loss")  # y軸ラベル
    plt.title("Cross Entropy Error of valid", fontsize=20)  # グラフタイトル
    fig.savefig("result/valid_loss.pdf")

    epoch = list(range(1, len(losses) + 1, 1))
    train = pd.DataFrame({'epoch': epoch, 'loss': losses, 'accuracy': training_accuracies})
    valid = pd.DataFrame({'epoch': epoch, 'loss': valid_losses, 'accuracy': valid_accuracies})
    train.to_csv('result/train_loss_acc.txt', sep='\t', index=False)
    valid.to_csv('result/valid_loss_acc.txt', sep='\t', index=False)


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html2(sentence, attns, embedding):
    html = ""
    if embedding == 'one_hot':
        attns = attns[:len(sentence)]  # 0埋めした分の重みを削除
        mean = attns.mean()
        for i in range(len(sentence) - 6 + 1):
            attns[i] = max(0, attns[i] - mean)

        for word, attn in zip(sentence, attns):
            html += ' ' + highlight(
                word,
                attn
            )
        return html

    else:
        cat_attn = torch.zeros([len(sentence)])
        attns = attns[:len(sentence) - 6 + 1]  # 0埋めした分の重みを削除

        for i in range(len(sentence) - 6 + 1):
            if i == 0:
                for j in range(6):
                    cat_attn[j] = attns[i]
            else:
                for j in range(6 - 1):
                    cat_attn[i + j] += attns[i]
                if i != len(sentence) - 6 + 1:
                    cat_attn[i + 5] = attns[i]
        maxvalue = cat_attn.max()
        for i in range(len(sentence)):
            cat_attn[i] = max(0, cat_attn[i] / maxvalue)
        for word, attn in zip(sentence, cat_attn):
            html += ' ' + highlight(
                word,
                attn
            )
        return html

def min_max2(x):  # メモリの消費はおそらく少ないけど# めっちゃ時間かかるやつ
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
            maxv = torch.max(x[i][j])
            minv = torch.min(x[i][j])
            x_v = (x[i][j] - minv) / (maxv - minv)
            if i == 0 and j == 0:
                x_new = x_v
            else:
                x_new = torch.cat((x_new, x_v), 0)
    x_new = torch.reshape(x_new, (x.size()[0], x.size()[1], -1))
    return x_new

def min_max(x, dim):
    xmax = torch.max(x, dim=dim)[0]
    xmin = torch.min(x, dim=dim)[0]
    xmax = xmax.repeat(1, 1, x.size()[dim])
    xmin = xmin.repeat(1, 1, x.size()[dim])
    xmax = torch.reshape(xmax, (x.size()[0], x.size()[1], -1))
    xmin = torch.reshape(xmin, (x.size()[0], x.size()[1], -1))
    if dim == 2:
        xmax = torch.transpose(xmax, 1, 2)
        xmin = torch.transpose(xmin, 1, 2)
        x = torch.transpose(x, 1, 2)
    x_v = (x - xmin) / (xmax - xmin)
    if dim == 2:
        x_v = torch.transpose(x_v, 1, 2)
    return x_v

def z_score(x, dim):
    xmean = torch.mean(x, dim=dim)
    xstd = torch.std(x, dim=dim)
    xmean = xmean.repeat(1, 1, x.size()[dim])
    xstd = xstd.repeat(1, 1, x.size()[dim])
    xmean = torch.reshape(xmean, (x.size()[0], x.size()[1], -1))
    xstd = torch.reshape(xstd, (x.size()[0], x.size()[1], -1))
    if dim == 2:
        xmean = torch.transpose(xmean, 1, 2)
        xstd = torch.transpose(xstd, 1, 2)
        x = torch.transpose(x, 1, 2)
    x_v = (x - xmean) / xstd
    if dim == 2:
        x_v = torch.transpose(x_v, 1, 2)
    return x_v


def z_score2(x):
    x = torch.transpose(x, 1, 2)
    xmean = torch.mean(x, dim=1)
    xstd = torch.std(x, dim=1)
    xmean = xmean.repeat(1, 1, x.size()[1])
    xstd = xstd.repeat(1, 1, x.size()[1])
    xmean = torch.reshape(xmean, (x.size()[0], x.size()[1], -1))
    xstd = torch.reshape(xstd, (x.size()[0], x.size()[1], -1))
    x_v = (x - xmean) / xstd
    return x_v

def min_max3(x):
    x = torch.transpose(x, 1, 2)
    xmax = torch.max(x, dim=1)[0]
    xmin = torch.min(x, dim=1)[0]
    xmax = xmax.repeat(1, 1, x.size()[1])
    xmin = xmin.repeat(1, 1, x.size()[1])
    xmax = torch.reshape(xmax, (x.size()[0], x.size()[1], -1))
    xmin = torch.reshape(xmin, (x.size()[0], x.size()[1], -1))
    x_v = (x - xmin) / (xmax - xmin)
    return x_v


def mk_html(sentence, attns, embedding, stride, mer):
    html = ""
    if embedding == 'one_hot':
        attns = attns[:len(sentence)]  # 0埋めした分の重みを削除
        mean = attns.mean()
        for i in range(len(sentence) - 6 + 1):
            attns[i] = max(0, attns[i] - mean)

        for word, attn in zip(sentence, attns):
            html += ' ' + highlight(
                word,
                attn
            )
        return html

    else:
        length = int((len(sentence) - mer) / stride + 1)
        cat_attn = torch.zeros([len(sentence)])
        attns = attns[:len(sentence)]  # 0埋めした分の重みを削除
        for i in range(length):
            for j in range(mer):
                cat_attn[i * stride + j] += attns[i]
        maxvalue = cat_attn.max()
        minvalue = cat_attn.min()
        for i in range(length):
            cat_attn[i] = (cat_attn[i] - minvalue) / (maxvalue - minvalue)
        for word, attn in zip(sentence, cat_attn):
            html += ' ' + highlight(
                word,
                attn
            )
        return html

def attention_view(dir, test_tf, test_ta, num, batch_size, x, i, ans, aseqs, dseqs, pre, amino_att, dna_att, embedding,
                   amino_stride, stride, amino_mer, mer, map, vector, dna_attmap, N):
    id2ans = {'0': 'negative', '1': 'positive'}
    with open(dir + "html/" + x + "_amino.html", "a") as file:
        file.write(test_tf[num * batch_size + i] + ' ->-> ' + test_ta[num * batch_size + i] + '\t【正解】' + id2ans[
            str(ans[i].item())] + '\t【予測】' + id2ans[str(pre[i].item())] + '<br><br>')
        file.write(mk_html(list(aseqs[i]), amino_att.data[i, :, 0], embedding, amino_stride, amino_mer) + '<br><br>')
    with open(dir + "html/" + x + "_dna.html", "a") as file:
        file.write(test_tf[num * batch_size + i] + ' ->-> ' + test_ta[num * batch_size + i] + '\t【正解】' + id2ans[
            str(ans[i].item())] + '\t【予測】' + id2ans[str(pre[i].item())] + '<br><br>')
        file.write(mk_html(list(dseqs[i]), dna_att.data[i, :, 0], embedding, stride, mer) + '<br><br>')

    if map == 'TRUE':
        att = dna_attmap[i].numpy()
        att = pd.DataFrame(att)
        amer = vector.k_mer(aseqs[i], amino_mer, amino_stride)
        dmer = vector.k_mer(dseqs[i], mer, stride)
        att.index = amer
        att.columns = dmer
        fig = plt.figure()
        #sns.heatmap(att)
        fig.savefig(dir + "Details/AttentionMap/" + x + "_" + str(N) + "attention.pdf")
        plt.close('all')


class Occullusion_Embedding():  # stride=1のみに対応
    def __init__(self, amino_dict, dna_dict, stride, mer, amino_stride, amino_mer, NULL_length, NULL_stride):
        self.amino_dict = amino_dict
        self.dna_dict = dna_dict
        self.stride = stride
        self.mer = mer
        self.amino_stride = amino_stride
        self.a_mer = amino_mer
        self.NULL_length = NULL_length
        self.NULL_stride = NULL_stride
        self.a_finish = 0

    def convert_vector(self, aseq, dseq):
        amino_hoge, dna_hoge = [], []
        for index in range(len(aseq)):
            amino_seq = aseq[index]  # iat:列番号と行番号で位置を指定してその要素を取得
            dna_seq = dseq[index]
            amino_mer = self.k_mer(amino_seq, self.a_mer, self.amino_stride)
            amino_vector = torch.tensor(self.amino_convert_one_hot(amino_mer))
            amino_hoge.append(amino_vector)
            dna_mer = self.k_mer(dna_seq, self.mer, self.stride)
            dna_vector = torch.tensor(self.dna_convert_one_hot(dna_mer))
            dna_hoge.append(dna_vector)
            self.a_finish = math.ceil(len(amino_vector) / self.NULL_stride)
            for i in range(math.ceil(len(amino_vector) / self.NULL_stride)):
                init = i * self.NULL_stride
                finish = init + self.NULL_length - self.a_mer + 1
                tmp = list(range(init, finish))
                tmp_amino_vector = torch.tensor(self.NULL_amino_convert_one_hot(amino_mer, tmp))
                amino_hoge.append(tmp_amino_vector)
                dna_hoge.append(dna_vector)
            for i in range(math.ceil(len(dna_vector) / self.NULL_stride)):
                init = i * self.NULL_stride
                finish = init + self.NULL_length - self.mer + 1
                tmp = list(range(init, finish))
                tmp_dna_vector = torch.tensor(self.NULL_dna_convert_one_hot(dna_mer, tmp))
                dna_hoge.append(tmp_dna_vector)
                amino_hoge.append(amino_vector)
        return amino_hoge, dna_hoge

    def amino_convert_one_hot(self, amino_mer):
        amino_mers = []
        for mer in amino_mer:
            try:
                amino_mers.append(self.amino_dict[mer])

            except:
                amino_mers.append(0)
        return amino_mers

    def dna_convert_one_hot(self, dna_mer):
        dna_mers = []
        for mer in dna_mer:
            try:
                dna_mers.append(self.dna_dict[mer])
            except:
                dna_mers.append(0)
        return dna_mers

    def NULL_amino_convert_one_hot(self, amino_mer, tmp):
        amino_mers = []
        for i, mer in enumerate(amino_mer):
            if i in tmp:
                amino_mers.append(0)
            else:
                try:
                    amino_mers.append(self.amino_dict[mer])

                except:
                    amino_mers.append(0)
        return amino_mers

    def NULL_dna_convert_one_hot(self, dna_mer, tmp):
        dna_mers = []
        for i, mer in enumerate(dna_mer):
            if i in tmp:
                dna_mers.append(0)
            else:
                try:
                    dna_mers.append(self.dna_dict[mer])
                except:
                    dna_mers.append(0)
        return dna_mers

    def k_mer(self, str, n, s):
        mer = []
        for i in range(0, len(str) - n + 1, s):
            mer.append(str[i:i + n])
        return mer

    def each_sub(self, val, b, mul_num):
        return mul_num * (val - b)

    def Occulusion_html(self, score, seq, ans):
        basis = score[0]
        mul_num = -1
        if ans.item() == 0:
            mul_num = 1
        score = [self.each_sub(val, basis, mul_num) for val in score]  # scoreが大きいほど正解に寄与している
        length = int((len(list(seq)) - self.NULL_length) / self.NULL_stride + 1)
        cat_attn = torch.zeros([len(list(seq))])
        num = 0
        for i in range(length):
            for j in range(self.NULL_length):
                cat_attn[i * self.NULL_stride + j] += score[i + 1]
            num += 1

        tmp = len(list(seq)) - 1 - (num * self.NULL_stride + self.NULL_length - 1)
        num += 1
        for j in range(tmp):
            cat_attn[num * self.NULL_stride + j] += score[num + 1]

        cat_attn2 = cat_attn.clone()
        maxvalue = cat_attn.max()
        minvalue = cat_attn.min()
        for i in range(len(list(seq))):
            cat_attn2[i] = (cat_attn[i] - minvalue) / (maxvalue - minvalue)
        html = ""
        for word, attn in zip(list(seq), cat_attn2):
            html += ' ' + highlight(
                word,
                attn
            )
        return html, score

    def view_Occulusion(self, score, seq, dseq, test_tf, test_ta, ans, dir, pre):
        id2ans = {'0': 'negative', '1': 'positive'}
        with open(dir + "html/" + test_tf[0] + ".html", "a") as file:
            file.write(test_tf[0] + ' ->-> ' + test_ta[0] + '\t【正解】' + id2ans[
                str(ans[0].item())] + '\t【予測】' + id2ans[str(pre[0].item())] + '<br><br>')
            # file.write(self.Occulusion_html(score[:self.a_finish + 1], seq, ans) + '<br><br>')
            html, a_cat_attn = self.Occulusion_html(score[:self.a_finish + 1], seq, ans[0])
            file.write(html + '<br><br>')
            del score[1: self.a_finish + 1]
            html, d_cat_attn = self.Occulusion_html(score, dseq, ans[0])
            file.write(html + '<br><br>')
        return a_cat_attn, d_cat_attn


def Occulusion_html(sentence, attns, stride, mer):
    html = ""
    length = int((len(sentence) - mer) / stride + 1)
    cat_attn = torch.zeros([len(sentence)])
    attns = attns[:len(sentence)]  # 0埋めした分の重みを削除
    for i in range(length):
        for j in range(mer):
            cat_attn[i * stride + j] += attns[i]
    maxvalue = cat_attn.max()
    minvalue = cat_attn.min()
    for i in range(length):
        cat_attn[i] = (cat_attn[i] - minvalue) / (maxvalue - minvalue)
    for word, attn in zip(sentence, cat_attn):
        html += ' ' + highlight(
            word,
            attn
        )
    return html

def view_motif(name, seq, tf, check_dir, match):
    cat_attn = torch.zeros([len(seq)])
    #print([m.span() for m in match])
    for s in match:
        init = s.span()[0]
        finish = s.span()[1]
        for i in range(init, finish):
            cat_attn[i] += 1
    html = ""
    for word, attn in zip(list(seq), cat_attn):
        html += ' ' + highlight(
            word,
            attn
        )

    with open(check_dir + "html/" + tf + ".html", "a") as file:
        file.write(tf + ' ->-> ' + name + '<br><br>')
        file.write(html + '<br><br>')

def choose_target(motif, check_dir, tf, test, train_or_test):
    if train_or_test == 0:
        score = pd.read_csv(check_dir + '../../result/trainscore.txt', sep='\t')
    else:
        score = pd.read_csv(check_dir + '../../result/testscore.txt', sep='\t')
    score = score[score['tf'] == tf]
    score = score[score['label'] == 1]
    score = score[score['pred'] == 1.0]
    score = score.sort_values('score', ascending=False)
    score.reset_index(drop=True, inplace=True)
    num = 10
    names = []
    seqs = []
    matches = []

    '''
    for t in ['TBC1D2', 'PINX1', 'GRPR', 'MTERF2', 'EIF6', 'A2M', 'RNU4ATAC7P', 'FAM120C', 'RN7SKP270', 'INTS4P2']:
        if t == 'TBC1D2':
            positive = score[score['target'] == t]
        else:
            tmp1 = score[score['target'] == t]
            positive = pd.concat([positive, tmp1])
    score = positive.copy()
    score.reset_index(drop=True, inplace=True)
    print(score)
    '''
    for i in range(len(score)):
        target = score['target'][i]
        tmp = test[test['target'] == target]
        tmp.reset_index(drop=True, inplace=True)
        seq = tmp['ta_seq'][0]
        name = tmp['target'][0]
        for j in range(len(motif)):
            if len(names) >= num:
                return names, seqs, matches
            res = re.match(motif[j], seq)
            if res:
                print("マッチしました。")
                print(name)
                print(motif[j])
                print(score[score['target'] == name]['score'])
                match = re.finditer(motif[j], seq)
                #view_motif(name, seq, tf, check_dir, match)
                names.append(name)
                seqs.append(seq)
                matches.append(match)
                #return name
    print("マッチしませんでした。")
    return names, seqs, matches


def make_filter_pwm(filter_fasta, meme_out, tf='all', nts='DNA'):
    print('filter_fasta: ' + str(filter_fasta))
    ''' Make a PWM for this filter from its top hits '''
    if nts == 'DNA':
        nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # embd
        pwm_counts = []
        nsites = 0  # pseudocounts
        for line in open(filter_fasta):
            if line[0] != '>':
                seq = line.rstrip()
                nsites += 1
                if len(pwm_counts) == 0:
                    # initialize with the length
                    for i in range(len(seq)):
                        pwm_counts.append(np.array([0.0] * 4))

                # count
                for i in range(len(seq)):
                    try:
                        pwm_counts[i][nts[seq[i]]] += 1
                    except KeyError:
                        pwm_counts[i] += np.array([0.25] * 4)
        # normalize
        pwm_freqs = []
        for i in range(len(pwm_counts)):
            pwm_freqs.append([pwm_counts[i][j] / float(nsites) for j in range(4)])

        if nsites > 10:
            # add to the meme motif file
            ''' Print a filter to the growing MEME file

                Attrs:
                    meme_out : open file
                    f (int) : filter index #
                    filter_pwm (array) : filter PWM array
                    nsites (int) : number of filter sites
                '''
            print('MOTIF %s (%s)' % (tf, tf), file=meme_out)
            print('letter-probability matrix: alength= 4 w= %d nsites= %d' % (len(pwm_freqs), nsites),
                  file=meme_out)

            for i in range(len(pwm_freqs)):
                print('%.4f %.4f %.4f %.4f' % tuple(pwm_freqs[i]), file=meme_out)

            print('', file=meme_out)
    else:
        nts = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
               'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        # embd
        pwm_counts = []
        nsites = 0  # pseudocounts
        for line in open(filter_fasta):
            if line[0] != '>':
                seq = line.rstrip()
                nsites += 1
                if len(pwm_counts) == 0:
                    # initialize with the length
                    for i in range(len(seq)):
                        pwm_counts.append(np.array([0.0] * 20))

                # count
                for i in range(len(seq)):
                    try:
                        pwm_counts[i][nts[seq[i]]] += 1
                    except KeyError:
                        pwm_counts[i] += np.array([0.05] * 4)
        # normalize
        pwm_freqs = []
        for i in range(len(pwm_counts)):
            pwm_freqs.append([pwm_counts[i][j] / float(nsites) for j in range(20)])

        if nsites > 10:
            # add to the meme motif file
            ''' Print a filter to the growing MEME file

                Attrs:
                    meme_out : open file
                    f (int) : filter index #
                    filter_pwm (array) : filter PWM array
                    nsites (int) : number of filter sites
                '''
            print('MOTIF %s (%s)' % (tf, tf), file=meme_out)
            print('letter-probability matrix: alength= 20 w= %d nsites= %d' % (len(pwm_freqs), nsites),
                  file=meme_out)

            for i in range(len(pwm_freqs)):
                print('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f '
                      '%.4f' % tuple(pwm_freqs[i]), file=meme_out)

            print('', file=meme_out)



def meme_intro(meme_file, seqs):
    print(meme_file)
    ''' Open MEME motif format file and print intro

    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs

    Returns:
        mem_out : open MEME file
    '''
    nts = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    nt_counts = [1] * 4
    # count
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i] / nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    print('MEME version 4', file=meme_out)
    print('', file=meme_out)
    # embd
    print('ALPHABET= ACGT', file=meme_out)

    print('', file=meme_out)
    print('Background letter frequencies:', file=meme_out)
    # embd
    print('A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs), file=meme_out)
    print('', file=meme_out)

    return meme_out

def amino_meme_intro(meme_file):
    nts = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
           'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    nt_freqs = [0.05] * 20

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    print('MEME version 4', file=meme_out)
    print('', file=meme_out)
    # embd
    print( 'ALPHABET= ACDEFGHIKLMNPQRSTVWY', file=meme_out)

    print('', file=meme_out)
    print('Background letter frequencies:', file=meme_out)
    # embd
    print('A %.4f C %.4f D %.4f E %.4f F %.4f G %.4f H %.4f I %.4f K %.4f L %.4f M %.4f N %.4f P %.4f '
          'Q %.4f R %.4f S %.4f T %.4f V %.4f W %.4f Y %.4f' % tuple(nt_freqs), file=meme_out)
    print('', file=meme_out)

    return meme_out

def check_tomtom(tomtomfile, memefile):
    tom_result = pd.read_csv(tomtomfile, sep='\t')[:-3]

    meme_data = []
    with open(memefile, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 0:
                if 'MOTIF' in row[0].split():
                    meme_data.append(row[0].split()[1])
                    meme_data.append(row[0].split()[2].replace('(', '').replace(')', ''))

    tf_list = tom_result.drop_duplicates(subset='Query_ID')['Query_ID']
    tf_list.reset_index(drop=True, inplace=True)
    for tf in tf_list:
        print(tf)
        try:
            q_motif = meme_data[meme_data.index(tf) - 1]
            tmp = tom_result[tom_result['Query_ID'] == tf]['Target_ID']
            for motif in tmp:
                if q_motif in motif:
                    print(tf + ' ' + q_motif)

        except:
            print(tf + ' is not exist.')

def check_prosite(dir):
    filename = dir + '/tomtom/ELME_tomtom.tsv'
    tomtom_result = pd.read_csv(filename, sep='\t')[:-3]
    Query_IDs = tomtom_result.drop_duplicates(subset='Query_ID')
    Target_IDs = Query_IDs.drop_duplicates(subset='Target_ID')['Target_ID']
    targets = []
    num = []

    for target in Target_IDs:
        targets.append(target)
        num.append(len(Query_IDs[Query_IDs['Target_ID'] == target]))
    data = []
    data.append(targets)
    data.append(num)
    print(data)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(dir + '/tomtom/EL_hit.txt', sep='\t', index=False, header=False)


