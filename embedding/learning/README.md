# ヒトの全配列を用いてベクトルの事前学習を行うためのディレクトリ。
output は../all_vectorに出る。
これらのベクトルを使いたいときは、学習を進める際にemb_num='all' に設定する。

## ヒトの全アミノ酸配列を用いてベクトルの事前学習を行う

1. アミノ酸配列の元データから、「配列名　配列」を１行にまとめたデータを作成する。
```
R --no-save < edit_seq.R
```

2. アミノ酸配列に関してベクトルの事前学習を行う
```
python learn.py
```





## ヒトの全DNA配列を用いてベクトルの事前学習を行う
1. dna2vecを実行
learn_dnaディレクトリのREADME.mdを参考にして下準備を行い、ベクトルの事前学習を行う。

2. dna2vecの出力結果を編集する
```
less ./learn_dna/dna2vec/results/[dna2vecの出力ファイル名] | awk 'BEGIN{ORS=""}{print $1"\t";{for(i=2;i<NF;i++){print $i" "}};{{i=NF}print $i"\n"}}' >! edit_dna2vec
python make_dnaVec.py
```

