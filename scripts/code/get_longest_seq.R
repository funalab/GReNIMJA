###### ライブラリ #######

library(seqinr)

#########################



#### パラメータの設定 ####

# input filename
gene_protein_file = "./gene_gff/gene_proteinid.txt"
protein_seq_file = "./GCF_000001405.39/proteomes/GCF_000001405.39_GRCh38.p13_protein.faa"
# output filename
outfile="./gene_gff/longestGene_proteinid.txt"

#########################

######### 関数 #########

# TFのIDを入力として、もっとも長いスプライシングヴァリアントを取得する関数
get_longest_gene <- function(tf_id, i){
  protein <- gene_protein[gene_protein$tf_ID == tf_id, ]
  seq_len = 0
  for (j in 1:nrow(protein)){
    id <- protein[j,2]
        if(length(pseq[[id]]) > seq_len){
        protein_id <- id
        seq_len <- length(pseq[[id]])
      }
  }
  e <- try(paste(tf_id, protein_id, sep = "\t"))
  if( class(e) == "try-error") {
    print(paste("ERROR: ", tf_id, sep = ""))
  }else{
  v <- paste(tf_id, protein_id, sep = "\t")
  write.table(v, outfile, sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE)
  }
  i <- i + nrow(protein)
  return(i)
}

#########################

######### main #########

# データの読み込み
gene_protein <- read.table(gene_protein_file, header = F)
colnames(gene_protein) <- c("tf_ID", "protein_ID")
pseq <- read.fasta(protein_seq_file)

i = 1
while (i < nrow(gene_protein)){
  print(i)
  i <- get_longest_gene(gene_protein$tf_ID[i], i)
}

#########################

