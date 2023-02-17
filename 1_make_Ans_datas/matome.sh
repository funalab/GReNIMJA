第一引数: ダウンロードしてきたdorotheaの末尾7桁

# 遺伝子名変更の対応表を作成する
Rscript ./code/rename.R $1

# rename_tfより"ZNF286B"を削除(pseudogeneでCDSがないため)
less rename_tf | grep -v "ZNF286B" > rename_tf2
rm rename_tf
mv rename_tf2 rename_tf
zsh ./code/get_not_cutGene.sh

# 遺伝子名は異なるけど遺伝子の位置が同じものを削除する(strandは考慮している)
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene

# 正解データの作成（遺伝子名とラベルのみ）
Rscript ./code/make_data.R $1

# 最長のタンパク質を取ってくる
R --no-save < ./code/get_longest_seq.R

# 指定した上流bp分の配列とTFの配列を取ってくる
Rscript ./code/get_seq.R ans_Data 
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py # pickleに圧縮する

# TFの長さの分布を取得する(バッチ学習用)
R --no-save <  ./code/TF_bunpu.R
