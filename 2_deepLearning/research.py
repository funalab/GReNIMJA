import pickle
from func import convert, Embedding, initialize, Occullusion_Embedding, choose_target, view_motif, make_filter_pwm, \
    meme_intro, check_tomtom, check_prosite, amino_meme_intro
from layers import myModel
from torch.utils.data import DataLoader
from view_test import test_model
import torch
import os
from gensim.models.word2vec import Word2Vec
from Dataloadar import only_NewsDataset, only_pad_collate
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
from extract_motifs import get_motif, make_meme
import subprocess



def main():
    check_dir = '/Users/okubo/m1okubo/result/kekka/honban_2DLSTM_2CV/checkpoint/1checkpoint/'
    #check_dir = '/Users/okubo/m1okubo/result/kekka/5CV_2DLSTM/checkpoint/3checkpoint/'
    motif_detect = True
    dataNum = 10
    dir = r'../result/data/cpu_random/'
    filename = 'cpu_train1.pickle'
    Occulusion = True
    species = 'human'
    train_or_test = 0  # train:0, test:1

    if Occulusion:
        tf = 'NPAS2'
        motif = ['ACAATG', '[TA]TTGT']  # SOX2
        #motif = ['GGATTA']  # OTX2
        #motif = ['T[CGTA]A[CGTA]T[CGTA]A']  #JUND
        #motif = ['GTAAA[CT]A', 'T[GA]TT[TG][AG]C']  #FOXA1
        #motif = ['[AG]GGAA[AG]']  # SPI1
        NULL_length = 10  # 何塩基単位でNULLにするか
        NULL_stride = 10  # NULL_lengthと同じ場合重なりはなし

    # Attentionを可視化するか
    view = 'FALSE'  # Occulusionとの併用は不可
    map = 'FALSE'

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # device
    device = torch.device('cpu')

    # どのモデルを読み込むか
    check = 'checkpoint96.pt'

    CNN = True
    biLSTM = False
    att = '2DLSTM'  # att or mean or add or 2DLSTM or RNN or Flase or SPP or MBT
    fusion = 2  # 4:1linear(aminoを寄せる) or 1:1linear(dnaを寄せる) or 2:2linear or 3:3linear or False


    # dataset
    dataset = 'achoice'
    bp = 1000

    # model
    embedding = ['embedding']
    if embedding[0] == 'one_hot':
        a_embedding = 0  # onehot
        d_embedding = 0  # 0: onehot, 1: all, 2: onehot+NCP, 3:onehot+DPCP, 4: NCP, 5: DPCP, 6: NCP+DPCP
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
    emb_num = 'all'
    batch_size = 1

    kernel_size, kernel_stride, a_rnn, d_rnn, a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, DA, R, \
    cnn1_1, conv_num, rnn2d_dim, cat_rnn, a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, cos, output_num = initialize()


    if CNN:
        a_kernel_Xlen = 30
        d_kernel_Xlen = 18
        a_kernel_strideX = 4
        d_kernel_strideX = 4
        a_output_channel = 300
        d_output_channel = 260
        a_max_pool_len = 15
        d_max_pool_len = 15
        a_CNN_D = 0.458280
        d_CNN_D = 0.074344
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
            co_hiddendim = 190
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
        rnn2d_dim = 190
        cat_rnn = 'lstm'  # lstm or gru
        LSTM2D_D = 0.136456
    if att == 'SPP':
        output_num = [12, 8, 6, 3, 1]

    rnn = [a_rnn, d_rnn, cat_rnn]

    # NNの2層目の中間層の次元
    linear2_inputDim = 140

    # Dropout
    Dense_D1 = 0
    Dense_D2 = 0
    D_p = [a_CNN_D, d_CNN_D, a_biLSTM_D, d_biLSTM_D, LSTM2D_D, Dense_D1, Dense_D2]


    m_para = [embedding, pre, att, learn, cnn1_1, rnn, fusion, device, CNN, biLSTM, cos]
    param = [a_hiddendim, d_hiddendim, co_hiddendim, co_hiddendim2, linear2_inputDim, rnn2d_dim, conv_num, D_p,
             kernel_size,
             kernel_stride, output_channel, max_pool_len, DA, R, output_num]

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
            dna_preVec = '../embedding/' + str(mer) + '_' + str(dna_emb_dim) + '_' + str(
                stride) + '_Dword2vec.gensim.model'

    # どれがTPなのかなどを出力
    os.makedirs(check_dir + 'Details', exist_ok=True)
    os.makedirs(check_dir + 'html', exist_ok=True)
    os.makedirs(check_dir + 'Details/AttentionMap', exist_ok=True)

    with open(dir + filename, 'rb') as f:
        test = pickle.load(f)
        test['ta_seq'] = test['ta_seq'].str[-bp:]

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
    #test = test[test['target'] == 'SCRIB']
    #test = test[test['tf'] == 'BCL11A']
    #test2 = test[test['tf'] == 'SIN3A']
    #test = pd.concat([test1, test2])
    #print(test)
    #test = test[test['group'] > 5]
    #test = test[test['group'] < 10]

    dataNum = 'NULL'
    if dataNum != 'NULL':
        not_use, test = train_test_split(test, test_size=dataNum, shuffle=True, random_state=40, stratify=test['label'])

    ## モデルの定義 ##
    amino_weights, dna_weights, Avocab_size, Dvocab_size, emb_dim = 0, 0, 0, 0, 0
    if len(embedding) > 1:
        if embedding[0] in [0]:
            with open('../data/pickle/dict/' + str(species) + '_' + str(embedding[0]) + 'Aonehot.pickle', 'rb') as f:
                amino_dict = pickle.load(f)
        if embedding[1] in [0, 1, 2, 3, 4, 5, 6]:
            with open('../data/pickle/dict/' + str(species) + '_' + str(embedding[1]) + 'Donehot.pickle', 'rb') as f:
                dna_dict = pickle.load(f)

            vector = convert(amino_dict, dna_dict)
            # アミノ酸配列の種類数
            a_emb_dim = amino_emb_dim
            # 塩基配列の種類数
            d_emb_dim = dna_emb_dim

    if embedding[0] == 'embedding':
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
            a_emb_dim = amino_emb_dim
            d_emb_dim = dna_emb_dim
        if Occulusion and not motif_detect:
            vector = Occullusion_Embedding(amino_dict, dna_dict, stride, mer, amino_stride, amino_mer, NULL_length,
                                           NULL_stride)
        else:
            vector = Embedding(amino_dict, dna_dict, stride, mer, amino_stride, amino_mer)
            print('OK')
    emb_dim = [a_emb_dim, d_emb_dim]

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
    '''
        import seaborn as sns
        omomi = np.array(model.state_dict()['linear1.weight'])
        plt.figure()
        sns.heatmap(omomi)
        plt.savefig(check_dir + 'fc1_heatmap_ndarray.pdf')
        '''
    '''
    if motif_detect:
        test_aseq = test.drop_duplicates(subset='tf')['tf_seq']
        test_aseq.reset_index(drop=True, inplace=True)

        make_meme(test_aseq, dir1='motifs_database', data='amino', dir2='prosite_alignments_for_logos')


        max_len = len(test_aseq[0])
        if not os.path.isfile('./motifs_database/prosite.meme'):
            make_meme(test_aseq, dir1='motifs_database', data='amino', dir2='prosite_alignments_for_logos')

        test_aseq = test_aseq[:10]

        amino_hoge, amino_hoge_fill = vector.amino_convert_vector(test_aseq, max_len)
        output_length = 0
        #  全ての出力の平均値を求める（0埋めをすると平均が非常に小さくなってしまうため一つずつ出力を出して足していき最後に配列長の合計で割る）
        for i in range(len(amino_hoge)):
            aseq = model.amino_embed(torch.tensor(amino_hoge[i]))
            aseq = torch.reshape(aseq, [1, aseq.size()[0], aseq.size()[1]])
            tmp_output = model.amino_cnn(aseq).detach().numpy().copy()
            output_length += tmp_output.shape[2]
            tmp_output = np.reshape(tmp_output, [tmp_output.shape[1], -1])
            tmp_output = np.sum(tmp_output, axis=1)
            if i == 0:
                output_sum = tmp_output
            else:
                output_sum = output_sum + tmp_output
        output_mean = output_sum / output_length

        amino_hoge = torch.tensor(amino_hoge_fill)
        aseq = model.amino_embed(amino_hoge)
        output = model.amino_cnn(aseq).detach().numpy().copy()
        cnn_para = model.state_dict()['a_conv.weight'][:, 0, :, :]

        #cnn_para = cnn_para[:10]

        cnn_para = torch.transpose(cnn_para, 1, 2).numpy()
        get_motif(cnn_para, output, test_aseq, dir1='a_motifs', emb=True,
                  data='amino', kmer=amino_mer, s=amino_stride, tomtom='./meme-5.5.0/src/tomtom', out_mean=output_mean)
        #check_prosite('a_motifs')
'''
    if motif_detect:
        test_dseq = test.drop_duplicates(subset='target')['ta_seq']
        test_dseq.reset_index(drop=True, inplace=True)
        #test_dseq = test_dseq[:10]
        dna_hoge = vector.dna_convert_vector(test_dseq)
        #dna_hoge = np.array(dna_hoge, dtype=np.float32)
        #dna_hoge = torch.from_numpy(dna_hoge).to(torch.float32)
        dna_hoge = torch.tensor(dna_hoge)
        dseq = model.dna_embed(dna_hoge)
        output = model.dna_cnn(dseq).detach().numpy().copy()
        cnn_para = model.state_dict()['d_conv.weight'][:, 0, :, :]

        #cnn_para = cnn_para[:10]

        cnn_para = torch.transpose(cnn_para, 1, 2).numpy()

        #output = output[:, :10, :]
        #cnn_para = cnn_para[:10, :, :]


        get_motif(cnn_para, output, test_dseq, dir1='motifs', emb = True,
                  data='DNA', kmer=mer, s=stride, tomtom='./meme-5.5.0/src/tomtom')
    exit()
    output_PWM = True
    if output_PWM:
        os.makedirs(check_dir + '../../Occlusion/weblogo/tomtom', exist_ok=True)
        oc_dir = check_dir + '../../Occlusion/'
        weblogo_opts = '-X NO --fineprint ""'
        weblogo_opts += ' -C "#0C8040" A A'
        weblogo_opts += ' -C "#34459C" C C'
        weblogo_opts += ' -C "#FBB116" G G'
        # embed
        weblogo_opts += ' -C "#CB2026" T T'

        test_dseq = test.drop_duplicates(subset='target')['ta_seq']
        test_dseq.reset_index(drop=True, inplace=True)
        meme_out = meme_intro(oc_dir + 'weblogo/outs.txt', test_dseq)

        if train_or_test == 0:
            score = pd.read_csv(check_dir + '../../result/trainscore.txt', sep='\t')
        else:
            score = pd.read_csv(check_dir + '../../result/testscore.txt', sep='\t')

        print(score)
        test = test[:len(score)]
        score['ta_seq'] = test['ta_seq']
        score['tf_seq'] = test['tf_seq']
        print(score)
        t_score = score[score['label'] == 1]
        t_score = t_score[t_score['pred'] == 1.0]

        t_score = t_score.sort_values('tf')
        t_score.reset_index(drop=True, inplace=True)
        all_seqs = t_score['ta_seq']
        a_all_seqs = t_score['tf_seq']


        t_score = t_score.sort_values('tf')
        t_score.reset_index(drop=True, inplace=True)


        tf_list = t_score.drop_duplicates(subset='tf')
        tf_list.reset_index(drop=True, inplace=True)
        tf_list = tf_list['tf']
        first_tf = tf_list[0]

        all_a_cat_attns, all_d_cat_attns = [], []

        if not os.path.isfile(oc_dir + 'A_all.fa'):
            for tf in tf_list:
                tmp_score = t_score[t_score['tf'] == tf]
                tmp_score.reset_index(drop=True, inplace=True)
                ta_list = tmp_score['target']

                a_cat_attns, d_cat_attns = [], []
                seqs = []

                tf_tests = test[test['tf'] == tf]
                for i in range(len(ta_list)):
                    if i == 0:
                        tmp_tests = tf_tests[tf_tests['target'] == ta_list[i]]
                    else:
                        tmp = tf_tests[tf_tests['target'] == ta_list[i]]
                        tmp_tests = pd.concat([tmp_tests, tmp], axis=0)

                for index in range(len(tmp_tests)):
                    tmp_test = tmp_tests[tmp_tests['target'] == ta_list[index]]

                    tmp_test = tmp_test.reset_index(drop=True)
                    test_size = len(tmp_test)
                    test_tf, test_ta, test_aseq, test_dseq, test_ans \
                        = tmp_test['tf'], tmp_test['target'], tmp_test['tf_seq'], tmp_test['ta_seq'], tmp_test[
                        'label']
                    seqs.append(tmp_test.iat[0, 3])
                    # view_motif(targets[i], seqs[i], tf, check_dir, matches[i])
                    dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
                    test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True)
                    # 1epochあたりの繰り返し数
                    test_iter_per_epoch = max(int(test_size / batch_size), 1)
                    test_num = test_iter_per_epoch * batch_size

                    a_cat_attn, d_cat_attn = test_model(vector, test_dataset, batch_size, model, test_num, device,
                                                        test_tf,
                                                        test_ta,
                                                        view, embedding, check_dir, att, stride, mer, amino_stride,
                                                        amino_mer, map,
                                                        epoch, Occulusion, tmp_test)
                    a_cat_attns.append(np.array(a_cat_attn))
                    d_cat_attns.append(np.array(d_cat_attn))
                    all_a_cat_attns.append(np.array(a_cat_attn))
                    all_d_cat_attns.append(np.array(d_cat_attn))

                    if index == 0:
                        a_all_attns = np.array(a_cat_attn)
                        d_all_attns = np.array(d_cat_attn)
                        a_length = a_all_attns.shape[0]
                        d_length = d_all_attns.shape[0]
                    else:
                        a_all_attns = np.concatenate([a_all_attns, np.array(a_cat_attn)], 0)
                        d_all_attns = np.concatenate([d_all_attns, np.array(a_cat_attn)], 0)

                    if index == 0 and tf == first_tf:
                        a_allall_attns = np.array(a_cat_attn)
                        d_allall_attns = np.array(d_cat_attn)
                    else:
                        a_allall_attns = np.concatenate([a_all_attns, np.array(a_cat_attn)], 0)
                        d_allall_attns = np.concatenate([d_all_attns, np.array(d_cat_attn)], 0)

                a_mean = np.mean(a_all_attns)
                d_mean = np.mean(d_all_attns)
                a_max = np.amax(a_all_attns)
                d_max = np.amax(d_all_attns)

                a_raw_t = 0.65 * (a_max - a_mean) + a_mean
                d_raw_t = 0.65 * (d_max - d_mean) + d_mean
                filter_fasta_out = open(oc_dir + '/A_' + tf + '.fa', 'w')
                filter_count = 0
                filter_size = NULL_length + mer - 1

                # CNNの出力が閾値(raw_t)を超えた配列を出力しまくる
                for i in range(len(tmp_test)):  # 各配列を順々に
                    for j in range(d_length):  # 1配列の文字を順々に
                        if d_cat_attns[i][j] > d_raw_t:
                            kmer = seqs[i][j * NULL_stride:j * NULL_stride + filter_size]
                            #  テストのうちi番目のデータのj文字目スタートの配列
                            print('>%d_%d' % (i, j), file=filter_fasta_out)
                            print(kmer, file=filter_fasta_out)
                            filter_count += 1
                filter_fasta_out.close()

            a_mean = np.mean(a_allall_attns)
            d_mean = np.mean(d_allall_attns)
            a_max = np.amax(a_allall_attns)
            d_max = np.amax(d_allall_attns)

            a_raw_t = 0.60 * (a_max - a_mean) + a_mean
            d_raw_t = 0.60 * (d_max - d_mean) + d_mean
            filter_fasta_out = open(oc_dir + '/D_all.fa', 'w')
            filter_count = 0
            filter_size = NULL_length + mer - 1

            # CNNの出力が閾値(raw_t)を超えた配列を出力しまくる
            for i in range(len(t_score)):  # 各配列を順々に
                for j in range(d_length):  # 1配列の文字を順々に
                    if all_d_cat_attns[i][j] > d_raw_t:
                        kmer = all_seqs[i][j * NULL_stride:j * NULL_stride + filter_size]
                        #  テストのうちi番目のデータのj文字目スタートの配列
                        print('>%d_%d' % (i, j), file=filter_fasta_out)
                        print(kmer, file=filter_fasta_out)
                        filter_count += 1
            filter_fasta_out.close()

            filter_fasta_out = open(oc_dir + '/A_all.fa', 'w')
            filter_count = 0
            filter_size = NULL_length + mer - 1

            # CNNの出力が閾値(raw_t)を超えた配列を出力しまくる
            for i in range(len(t_score)):  # 各配列を順々に
                for j in range(all_a_cat_attns[i].shape[0]):  # 1配列の文字を順々に
                    if all_a_cat_attns[i][j] > a_raw_t:
                        kmer = a_all_seqs[i][j * NULL_stride:j * NULL_stride + filter_size]
                        #  テストのうちi番目のデータのj文字目スタートの配列
                        print('>%d_%d' % (i, j), file=filter_fasta_out)
                        print(kmer, file=filter_fasta_out)
                        filter_count += 1
            filter_fasta_out.close()

        '''
        for tf in tf_list:
            weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % \
                          (weblogo_opts, oc_dir + 'A_' + tf, oc_dir + 'weblogo/A_' + tf)
            subprocess.call(weblogo_cmd, shell=True)

            make_filter_pwm(oc_dir + 'A_' + tf + '.fa', meme_out, tf)
        '''
        # all DNA
        weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % \
                      (weblogo_opts, oc_dir + 'D_all', oc_dir + 'weblogo/D_all')
        subprocess.call(weblogo_cmd, shell=True)

        make_filter_pwm(oc_dir + 'D_all.fa', meme_out)

        # all amino
        df = pd.read_csv(oc_dir + 'A_all.fa', header=None)
        name = df[::2]
        name.reset_index(drop=True, inplace=True)
        seq = df[1::2]
        seq.reset_index(drop=True, inplace=True)
        df = pd.concat([name, seq], axis=1)
        df = df.set_axis(['name', 'seq'], axis=1)
        df = df[df['seq'].str.len() >= 14]
        df.reset_index(drop=True, inplace=True)
        filter_fasta_out = open('%s.fa' % (oc_dir + 'new_A_all'), 'w')
        for index in range(len(df)):
            print(df.iat[index, 0], file=filter_fasta_out)
            print(df.iat[index, 1], file=filter_fasta_out)

        weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % \
                      (weblogo_opts, oc_dir + 'new_A_all', oc_dir + 'weblogo/A_all')
        print(weblogo_cmd)
        subprocess.call(weblogo_cmd, shell=True)
        meme_out = amino_meme_intro(oc_dir + 'weblogo/a_outs.txt')
        make_filter_pwm(oc_dir + 'new_A_all.fa', meme_out, tf='all', nts='amino')

        # koko
        tomtom_dir = './meme-5.5.0/src/tomtom'
        subprocess.call(tomtom_dir + ' -dist pearson -thresh 0.05 -oc %s/tomtom %s/a_outs.txt %s' % (
            oc_dir + 'weblogo', oc_dir + 'weblogo', 'motifs_database/prosite2021_04.meme'), shell=True)
        subprocess.call('cp %s/tomtom/tomtom.tsv %s/tomtom/tomtom.txt' % (oc_dir + 'weblogo', oc_dir + 'weblogo'),
                        shell=True)

        subprocess.call(tomtom_dir + ' -dist pearson -thresh 0.05 -oc %s/tomtom %s/outs.txt %s' % (
        oc_dir + 'weblogo', oc_dir + 'weblogo', 'motifs_database/Jaspar.meme'), shell=True)
        subprocess.call('cp %s/tomtom/tomtom.tsv %s/tomtom/tomtom.txt' % (oc_dir + 'weblogo', oc_dir + 'weblogo'), shell=True)

        check_tomtom('%s/tomtom/tomtom.txt' % (oc_dir + 'weblogo'), 'motifs_database/Jaspar.meme')


        exit()

















    elif Occulusion:
        #tf = 'HCFC1'
        #targets, seqs, matches = choose_target(motif, check_dir, tf, test)
        tmp_test = test[test['tf'] == tf]
        try:
            tmp_test = tmp_test.sample(n=100)
        except:
            print('smaller than 100.')
        tmp_test.reset_index(drop=True, inplace=True)
        tf_list = tmp_test['target'].tolist()
        print(tf_list)
        print(tf_list)

        for i in range(len(tf_list)):
            test = tmp_test[tmp_test['target'] == tf_list[i]]
            test_size = len(test)
            test.reset_index(drop=True, inplace=True)
            test_tf, test_ta, test_aseq, test_dseq, test_ans \
                = test['tf'], test['target'], test['tf_seq'], test['ta_seq'], test['label']
            #view_motif(targets[i], seqs[i], tf, check_dir, matches[i])
            dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
            test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True)
            # 1epochあたりの繰り返し数
            test_iter_per_epoch = max(int(test_size / batch_size), 1)
            test_num = test_iter_per_epoch * batch_size


            a_cat_attn, d_cat_attn = test_model(vector, test_dataset, batch_size, model, test_num, device, test_tf, test_ta,
                                           view, embedding, check_dir, att, stride, mer, amino_stride, amino_mer, map,
                                           epoch, Occulusion, test)
        exit()



    if Occulusion:
        tmp_test = test.copy()
        targets, seqs, matches = choose_target(motif, check_dir, tf, test, train_or_test)
        #targets = ['TMEM217']
        print(targets)
        for i in range(len(targets)):
            test = tmp_test[tmp_test['target'] == targets[i]]
            test = test[test['tf'] == tf]
            test_size = len(test)
            test.reset_index(drop=True, inplace=True)
            test_tf, test_ta, test_aseq, test_dseq, test_ans \
                = test['tf'], test['target'], test['tf_seq'], test['ta_seq'], test['label']
            #view_motif(targets[i], seqs[i], tf, check_dir, matches[i])
            dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
            test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True)
            # 1epochあたりの繰り返し数
            test_iter_per_epoch = max(int(test_size / batch_size), 1)
            test_num = test_iter_per_epoch * batch_size

            a_cat_attn, d_cat_attn = test_model(vector, test_dataset, batch_size, model, test_num, device, test_tf, test_ta,
                                           view, embedding, check_dir, att, stride, mer, amino_stride, amino_mer, map,
                                           epoch, Occulusion, test)
            exit()
    else:
        test_size = len(test)
        test.reset_index(drop=True, inplace=True)
        test_tf, test_ta, test_aseq, test_dseq, test_ans \
            = test['tf'], test['target'], test['tf_seq'], test['ta_seq'], test['label']

        dataset_test = NewsDataset(test_aseq, test_dseq, test_ans)
        test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True)
        # 1epochあたりの繰り返し数
        test_iter_per_epoch = max(int(test_size / batch_size), 1)
        test_num = test_iter_per_epoch * batch_size
        TPs, FPs, FNs, TNs, score = test_model(vector, test_dataset, batch_size, model, test_num, device, test_tf,
                                               test_ta,
                                               view, embedding, check_dir, att, stride, mer, amino_stride, amino_mer,
                                               map,
                                               epoch, Occulusion, test)
        test["score"] = score
        testlist = test.drop(['tf_seq', 'ta_seq'], axis=1)
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
