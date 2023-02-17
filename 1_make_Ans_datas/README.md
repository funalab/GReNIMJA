
1. ヒトのゲノムデータの取得と展開

```
R --no-save < ./code/gene_download.R
find ./ -type f -name "*.gz" -exec gunzip {} \; 
```
※注意　本研究時はGCF_000001405.39が最新であったため、GCF_000001405.39のゲノム情報を取得していました。
しかし、最新版のゲノムじゃないと、gene_download.Rは機能しないため、最新版のゲノムを取ってくる場合はgene_download.Rの28行目を書き換えてください。
それに伴いget_chr.R, get_seq.R, get_longest_seq.R の最初のGCF*を書き換える必要があります。
本研究で用いたGCF_00001505.3はすでにディレクトリ内に用意してあるので、このゲノム情報を用いてデータセットを作成する場合はこの行程は飛ばし、以下の2番より実行してください。


2. dorotheaからデータの取得と解凍
https://api.github.com/repos/saezlab/dorothea/tarball/HEAD よりダウンロード
ダウンロードするたびにファイル名が変わるため、解凍する際はファイル名を自身で打つ

```
tar -zxvf [ファイル名]
```

3. ENCODEからデータの取得と解凍

```
curl -O https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_attribute_edges.txt.gz
gunzip gene_attribute_edges.txt.gz
```

4. データベースの遺伝子名をNCBIの遺伝子名に対応づける

```
cd ./gene_gff
zsh gene_gff.sh  # Synonymsをまとめる ここでgene_gff.sh内に書かれている操作をする必要があるが面倒なので推奨しない。最初の二つのコマンドと最後の7行のみ実行することを推奨。
cd ../
```

5. 対応表を作成する（ここから下は以下を実行することで自動で最後まで実行できる）
```
zsh matome.sh [dorotheaのファイル名の下7桁]
```

# 一つずつ実行するなら以下の通りに実行する
```
Rscript ./code/rename.R [dorotheaのファイル名の下７桁]  # 対応表の作成

# rename_tfより"ZNF286B"を削除(pseudogeneでCDSがないため)
less rename_tf | grep -v "ZNF286B" > rename_tf2
rm rename_tf
mv rename_tf2 rename_tf

6. 遺伝子の上流1000bpを取ってくる

```
zsh ./code/get_not_cutGene.sh
```


# 遺伝子名は異なるけど遺伝子の位置が同じものを削除する(strandは考慮している)
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene
```

7. Synonymsを対応させることができなかった遺伝子を削除し、正しいTF-target関係とランダムに選んだ誤ったTF-target関係を1, 0でラベル付けする

```
Rscript ./code/make_data.R [dorotheaのファイル名の下７桁]
```


8. スプライシングヴァリアントを考慮して最長のタンパク質を取ってくる
```
R --no-save < ./code/get_longest_seq.R
```


9. 指定した上流bp分の配列とTFの配列を取ってくる
```
Rscript ./code/get_seq.R ans_Data
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py
```

9. TFの長さの分布を取得する(バッチ学習用)
```
R --no-save <  bunpu/TF_long_bunpu.R
```


