import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
#import seaborn as sns
#torch.manual_seed(0)


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
    def __init__(self, amino, dna, ans):
        self.amino = amino
        self.dna = dna
        self.ans = ans

    def convert_vector(self, aseq, dseq):
        amino_hoge, dna_hoge = [], []
        for index in range(len(aseq)):
            amino_seq = aseq[index]  # iat:列番号と行番号で位置を指定してその要素を取得
            dna_seq = dseq[index]
            #ans_seq = df.iat[index, 4]
            amino_vector = torch.from_numpy(self.amino_convert_one_hot(self.amino, amino_seq).astype(np.float32))  # torch.Size([665, 20])
            amino_hoge.append(amino_vector)
            dna_vector = torch.from_numpy(self.dna_convert_one_hot(self.dna, dna_seq).astype(np.float32))
            dna_hoge.append(dna_vector)
        return amino_hoge, dna_hoge

    def dna_convert_one_hot(self, dna, dna_seq):  # DNA配列をone-hotに変換する
        dnas = list(dna_seq)
        leng = len(dnas)
        one_hot = np.zeros((leng, 4), dtype=np.int32)
        corpus = np.array([dna[w] for w in dnas])
        for idx_0, dna_id in enumerate(corpus):
            one_hot[idx_0, dna_id] = 1

        return one_hot

    def amino_convert_one_hot(self, amino, amino_seq):  # アミノ酸配列をone-hotに変換する
        aminos = list(amino_seq)
        leng = len(aminos)
        one_hot = np.zeros((leng, 20), dtype=np.int32)
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

