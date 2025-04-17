# 使わないgeneをcut_geneに追加し、それに伴いrename_tfにも追加する

import csv
import pandas as pd

filename = './gene_gff/new_gene_edge.txt'

def cut_gene(df, moji):
    if moji == '+':
        df_plus_duplicate = (df['strand'].duplicated(keep=False)) & (df['init'].duplicated(keep=False))
        plus = df[df_plus_duplicate]
        plus = plus[plus['strand'] == '+']
        df_plus_duplicate = (df['strand'].duplicated(keep='first')) & (df['init'].duplicated(keep='first'))
        keep_plus = plus[~df_plus_duplicate]
        keep_plus = keep_plus[keep_plus['strand'] == '+']
    elif moji == '-':
        df_plus_duplicate = (df['strand'].duplicated(keep=False)) & (df['finish'].duplicated(keep=False))
        plus = df[df_plus_duplicate]
        plus = plus[plus['strand'] == '-']
        df_plus_duplicate = (df['strand'].duplicated(keep='first')) & (df['finish'].duplicated(keep='first'))
        keep_plus = plus[~df_plus_duplicate]
        keep_plus = keep_plus[keep_plus['strand'] == '-']

    for index in range(len(keep_plus)):
        tmp = plus[plus['scaffold'] == keep_plus.iat[index, 2]]
        if moji == '+':
            tmp = tmp[tmp['init'] == keep_plus.iat[index, 3]]
        elif moji == '-':
            tmp = tmp[tmp['finish'] == keep_plus.iat[index, 4]]
        print(tmp)
        max = 0
        for i in range(len(tmp) - 1):  # 文字数が最小の遺伝子名は？
            if len(tmp.iat[i, 0]) > len(tmp.iat[i + 1, 0]):
                max = i + 1
        cut_gene = tmp.drop(tmp.index[max])
        print(cut_gene)
        for i in range(len(cut_gene)):
            with open('./cut_gene', 'a') as f:
                print(cut_gene.iat[i, 0].replace('>', ''), file=f)

            write_list = [cut_gene.iat[i, 0].replace('>', ''), tmp.iat[max, 0].replace('>', '')]
            f = open('./rename_ta', 'a')
            f.write('\t'.join(write_list))
            f.write('\n')
            f.close()

datas = []
data = []

with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) == 2:
            for i in range(len(row)):
                data.append(row[i])
        if len(row) == 3:
            for i in range(len(row)):
                data.append(row[i])
            datas.append(data)
            data = []

df = pd.DataFrame(datas, columns=['gene', 'strand', 'scaffold', 'init', 'finish'])

with open('./cut_gene', 'a') as f:
    print('\n', file=f)

cut_gene(df, '+')
cut_gene(df, '-')
