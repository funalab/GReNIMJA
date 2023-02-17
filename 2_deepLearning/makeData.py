import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from func import make_group
import random
import os

def make_Data(data_para):
    device2 = data_para[0]
    dataset = data_para[1]
    filename = data_para[2]
    miss_datafile = data_para[3]
    TF_bunpu = data_para[4]
    order = data_para[5]
    label = data_para[6]
    valid_sample = data_para[7]
    test_sample = data_para[8]
    valid_Datanum = data_para[9]
    bp = data_para[10]
    data_dir = data_para[11]

    if device2 == 'cpu':  # cpu用のdfを作成
        # 全てのデータを読み込みpickle形式で "longestData.pickle" に保存する。最初に一度実行した。
        df = pd.read_csv(filename, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))
        groups = pd.read_csv(TF_bunpu, header=None, names=['group'])
        df = make_group(df, groups, order)
        #df = pd.concat([df, group], axis=1, names='group')

    else:  # gpu用のdfを作成
        # データのOpen
        with open(filename, 'rb') as f:
            df = pickle.load(f)
        with open(miss_datafile, 'rb') as f:
            miss_df = pickle.load(f)
        groups = pd.read_csv(TF_bunpu, sep=' ', header=None, names=('tf', 'length', 'group'))

        df = make_group(df, groups, order)
        miss_df = make_group(miss_df, groups, order)
        positive = df[df['confidence'] <= label]
        encode = df[df['confidence'] == 'na']
        positive = pd.concat([positive, encode])
        df = pd.concat([positive, miss_df], axis=0)
        df.reset_index(drop=True, inplace=True)

    # target配列を指定した上流塩基数分にする
    df['ta_seq'] = df['ta_seq'].str[-bp:]
    # df['ta_seq'] = df['ta_seq'].str[:50]
    blen = len(df)
    # 未知の塩基および残基を含むデータを削除
    df = df[~df['ta_seq'].str.contains('N')]
    df = df[~df['tf_seq'].str.contains('X')]
    alen = len(df)
    print(f'before size: {blen}, ->->->  after size: {alen}')

    if dataset == 'dchoice':
        random.seed(0)
        target = list(set(df['target'].tolist()))
        target = random.sample(target, test_sample)
        for i, ta in enumerate(target):
            tmp = df[df['target'] == ta]
            df = df[df['target'] != ta]
            if i == 0:
                test = tmp
            else:
                test = pd.concat([test, tmp], axis=0)

        target = list(set(df['target'].tolist()))
        target = random.sample(target, valid_sample)
        for i, ta in enumerate(target):
            tmp = df[df['target'] == ta]
            df = df[df['target'] != ta]
            if i == 0:
                valid = tmp
            else:
                valid = pd.concat([valid, tmp], axis=0)

        train = df

    if dataset == 'achoice':
        random.seed(0)
        tf = list(set(df['tf'].tolist()))
        tf = random.sample(tf, test_sample)
        for i, ta in enumerate(tf):
            tmp = df[df['tf'] == ta]
            df = df[df['tf'] != ta]
            if i == 0:
                test = tmp
            else:
                test = pd.concat([test, tmp], axis=0)

        tf = list(set(df['tf'].tolist()))
        tf = random.sample(tf, valid_sample)
        for i, ta in enumerate(tf):
            tmp = df[df['tf'] == ta]
            df = df[df['tf'] != ta]
            if i == 0:
                valid = tmp
            else:
                valid = pd.concat([valid, tmp], axis=0)

        train = df

    if dataset == 'Normal':
        testNum = 316298
        valid_Datanum = 62655
        #testNum = int(len(df) * 0.2)
        #Num = int(testNum / 2)
        posi = df[df['label'] == 1]
        nega = df[df['label'] == 0]
        # true_nega = df[df['confidence'] == 'M']
        # false_nega = df[df['confidence'] == 'T']

        ptrain, ptest = train_test_split(posi, test_size=int(testNum / 2), shuffle=True, random_state=123)
        ptrain, pvalid = train_test_split(ptrain, test_size=int(valid_Datanum / 2), shuffle=True, random_state=123)
        ntrain, ntest = train_test_split(nega, test_size=int(testNum / 2), shuffle=True, random_state=123)
        ntrain, nvalid = train_test_split(ntrain, test_size=int(valid_Datanum / 2), shuffle=True, random_state=123)
        # ntrain = pd.concat([ntrain, false_nega], axis=0)

        train = pd.concat([ptrain, ntrain], axis=0)
        valid = pd.concat([pvalid, nvalid], axis=0)
        test = pd.concat([ptest, ntest], axis=0)

    if dataset == 'unbalance':
        posi = df[df['label'] == 1]
        nega = df[df['label'] == 0]

        ptrain, pvalid_test = train_test_split(posi, test_size=int(testNum / 2), shuffle=True, random_state=123)
        pvalid, ptest = train_test_split(pvalid_test, test_size=int(valid_Datanum / 2), shuffle=True,
                                         random_state=123)
        ntrain, nvalid_test = train_test_split(nega, test_size=int(testNum / 2), shuffle=True, random_state=123)
        nvalid, ntest = train_test_split(nvalid_test, test_size=int(valid_Datanum / 2), shuffle=True,
                                         random_state=123)

        train = pd.concat([ptrain, ntrain], axis=0)
        valid = pd.concat([pvalid, nvalid], axis=0)
        test = pd.concat([ptest, ntest], axis=0)

    train = train.sample(frac=1, random_state=123)
    valid = valid.sample(frac=1, random_state=123)
    test = test.sample(frac=1, random_state=123)

    train = train.sort_values('group', ascending=False)
    valid = valid.sort_values('group', ascending=False)
    test = test.sort_values('group', ascending=False)

    trainlist = train.drop(['tf_seq', 'ta_seq'], axis=1)
    validlist = valid.drop(['tf_seq', 'ta_seq'], axis=1)
    testlist = test.drop(['tf_seq', 'ta_seq'], axis=1)
    os.makedirs('../data/dataset', exist_ok=True)
    trainlist.to_csv('../data/dataset/train.txt', sep='\t', index=False)
    validlist.to_csv('../data/dataset/valid.txt', sep='\t', index=False)
    testlist.to_csv('../data/dataset/test.txt', sep='\t', index=False)

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    os.makedirs('../data/pickle', exist_ok=True)
    with open('../data/' + data_dir + '/train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('../data/' + data_dir + '/valid.pickle', 'wb') as f:
        pickle.dump(valid, f)
    with open('../data/' + data_dir + '/test.pickle', 'wb') as f:
        pickle.dump(test, f)



def cross_validation(K, data_dir, dataset):
    with open('../data/' + data_dir + '/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('../data/' + data_dir + '/valid.pickle', 'rb') as f:
        valid = pickle.load(f)

    train = pd.concat([train, valid], axis=0)
    train.reset_index(drop=True, inplace=True)
    true_train = train
    trains, valids = [], []

    if dataset == 'Normal':
        posi = true_train[true_train['label'] == 1]
        nega = true_train[true_train['label'] == 0]
        Num = int(len(true_train) / K)
        for i in range(K):
            ptrain, pvalid = train_test_split(posi, test_size=int(Num / 2), shuffle=True, random_state=i)
            ntrain, nvalid = train_test_split(nega, test_size=int(Num / 2), shuffle=True, random_state=i)
            train = pd.concat([ptrain, ntrain], axis=0)
            valid = pd.concat([pvalid, nvalid], axis=0)

            train = train.sample(frac=1, random_state=123)
            valid = valid.sample(frac=1, random_state=123)
            train = train.sort_values('group', ascending=False)
            valid = valid.sort_values('group', ascending=False)
            train.reset_index(drop=True, inplace=True)
            valid.reset_index(drop=True, inplace=True)
            trains.append(train)
            valids.append(valid)
            print(trains[i])
            print(valids[i])

    else:
        if dataset == 'dchoice':
            target_list = sorted(list(set(train['target'].tolist())))
        elif dataset == 'achoice':
            target_list = sorted(list(set(train['tf'].tolist())))
        target_num = int(len(target_list) / K)
        target_nums = []

        #  validデータに含まれるDNA配列の種類数をtarget_numsに格納する
        for i in range(K):
            target_nums.append(target_num)
        i = 0
        while sum(target_nums) < len(target_list):
            target_nums[i] += 1
            i += 1

        random.seed(0)
        random.shuffle(target_list)

        l = 0
        for i in range(len(target_nums)):
            train = true_train
            for j, ta in enumerate(target_list[l:l + target_nums[i]]):
                if dataset == 'dchoice':
                    tmp = train[train['target'] == ta]
                    train = train[train['target'] != ta]
                elif dataset == 'achoice':
                    tmp = train[train['tf'] == ta]
                    train = train[train['tf'] != ta]
                if j == 0:
                    valids.append(tmp)
                else:
                    valids[i] = pd.concat([valids[i], tmp], axis=0)
            l += target_nums[i]
            train = train.sample(frac=1, random_state=123)
            valids[i] = valids[i].sample(frac=1, random_state=123)
            train = train.sort_values('group', ascending=False)
            valids[i] = valids[i].sort_values('group', ascending=False)
            train.reset_index(drop=True, inplace=True)
            valids[i].reset_index(drop=True, inplace=True)
            trains.append(train)

    os.makedirs('../data/' + data_dir + '/' + str(K) + 'CV', exist_ok=True)
    for i in range(K):
        with open('../data/' + data_dir + '/' + str(K) + 'CV/train' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(trains[i], f)
        with open('../data/' + data_dir + '/' + str(K) + 'CV/valid' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(valids[i], f)

