# TFが可変長で、一つずつ入力すると学習にありえん時間がかかる。
# →0埋めをして入力したいけど、長さにあまりにも差がある場合に0埋めをするのはよくなさそう
# →TFの長さの分布を知り、長さごとにグループ分けしたい


##### parameta #####

TF_file = "../rename_tf"
gene_protein_file = "../gene_gff/longestGene_proteinid.txt"
protein_seq_file = "../GCF_000001405.39/proteomes/GCF_000001405.39_protein_refseq.faa"
TF_list = "./TF"
outfile = "./TF_group"
TF_bunpu = "TF_bunpu"

cut_group <- 50

####################


TF_group <- read.table(TF_bunpu)
print(TF_group)

# 各TFの名前と長さをベクトルとして保存する
for (i in 1:nrow(TF_group)){
  TF_name = TF_group[i,1]
  length = TF_group[i,2]
  if(i == 1){
    lengths<-c(length)
    TF_names <- c(TF_name)
  }else{
    lengths <- append(lengths, length)
    TF_names <- append(TF_names, TF_name)
  }
}

print(TF_names)
print(lengths)
# ベクトルとして保存していたTFの名前と長さをデータフレームに変換する
TF_length <- data.frame("names" = TF_names, "length" = lengths)

# TFの長さに関するヒストグラムを作成する 
kugiri <- seq(0, 4100, cut_group)
pdf("test.pdf", width = 10, height = 10)
hist(lengths, right = F, breaks = kugiri, ylim=c(0,150))

dev.off() 

exit()


###### ライブラリ #######

library(seqinr)

#########################

# TFの長さの分布をグラフにする





######## main ########

#setwd("~/m1okubo/1_make_Ans_data/bunpu")　# この行は実際はいらない


# TFのリストを読み込む
TF_table <- read.table(TF_file, header = T)

# スプライシングヴァリアントのうち最長のタンパク質のIDのリストを読み込む
longest <- read.table(gene_protein_file, header = F)
colnames(longest) <- c("tf_ID", "protein_ID")

# ヒトのタンパク質配列を読み込む
pseq <- read.fasta(protein_seq_file)

# 各TFの名前と長さをベクトルとして保存する
for (i in 1:nrow(TF_table)){
  TF_name = TF_table[i,2]
  longest_protein <- longest[longest$tf_ID == TF_name,]
  protein <- longest_protein[1,2]
  seq <- getSequence(pseq[protein], as.string = F)
  length <- length(seq[[1]])
  if(i == 1){
    lengths<-c(length)
    TF_names <- c(TF_name)
  }else{
    lengths <- append(lengths, length)
    TF_names <- append(TF_names, TF_name)
  }
}

# ベクトルとして保存していたTFの名前と長さをデータフレームに変換する
TF_length <- data.frame("names" = TF_names, "length" = lengths)

# TFの長さに関するヒストグラムを作成する 
kugiri <- seq(0, 4100, cut_group)
pdf("test.pdf", width = 10, height = 10)
hist(lengths, right = F, breaks = kugiri, ylim=c(0,150))

dev.off() 


# TFに対してヒストグラムのグループごとにラベル付けする
TF_length <- TF_length[order(TF_length$length),]
rownames(TF_length) <- 1:nrow(TF_length)
print("finish TF_lenght")

n = 150
index = 1
i <- 1
while(i <= nrow(TF_length)){
  while(TF_length[i,2] < n && i <= nrow(TF_length)){
    print(i)
    print(TF_length[i, 2])
    if(i == 1){
      group<-c(index)
    }else{
      group <- append(group, index)
    }
    i <- i + 1
  }
  n <- n + cut_group
  index <- index + 1
}

print("2")

# どのグループにいくつのTFが属するかを調べる
TF_length2 <- transform(TF_length, group = group)
write.table(TF_length2, file = TF_bunpu, quote=F, row.names = FALSE, col.names = FALSE)
write.table(TF_length2, file = paste("../../data/", TF_bunpu, sep=""), quote=F, row.names = FALSE, col.names = FALSE)

TF_group <- read.table(TF_bunpu)
TF <- read.table(TF_list)

uni <- unique(TF)

for ( i in 1:nrow(uni)){
  x <- uni[i,1]
  if (i == 1){
    TF_lists <- c(x)
    TF_num <- c(length(TF[TF$V1 == x,]))
  }else{
    TF_lists <- append(TF_lists, x)
    TF_num <- append(TF_num, length(TF[TF$V1 == x,]))
  }
}


for ( i in 1:nrow(uni)){
  group <- TF_group[TF_group$V1 == TF_lists[i],][1,3]
  d <- rep(group, TF_num[i])
  data.frame(d)
  write.table(d, file=outfile, quote=F, row.names = FALSE, col.names = FALSE, append = T)
}


