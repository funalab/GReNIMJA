# Program to create answer data

#### パラメータの設定 ####
# input filename
inputfile = "./rename_tf"
gene_protein_file = "./gene_gff/longestGene_proteinid.txt"
protein_seq_file = "./GCF_000001405.39/proteomes/GCF_000001405.39_protein_refseq.faa"
# output filename
outputfile = "../data/TF_bunpu"
#########################


######### main #########

TF_table <- read.table(inputfile, header = T)  # データを読み込み
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

TF_length <- TF_length[order(TF_length$length),]
rownames(TF_length) <- 1:nrow(TF_length)

cut_group <- 50
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


# どのグループにいくつのTFが属するかを調べる
TF_length <- transform(TF_length, group = group)
write.table(TF_length, file = outputfile, quote=F, row.names = FALSE, col.names = FALSE)
######################################