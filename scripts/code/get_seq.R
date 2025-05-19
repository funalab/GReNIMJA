# 正解データを入力すると、転写因子のアミノ酸配列と制御される遺伝子の指定した上流bp分の塩基配列を取ってきてくれる

###### ライブラリ #######

library(seqinr)

#########################



#### パラメータの設定 ####

# input filename
inputfile = commandArgs(trailingOnly=TRUE)[1]
renameTF = "./rename_tf"
renameTarget = "./rename_ta"

# スプライシングヴァリアントを考慮せず全ての遺伝子を使う場合
#gene_protein_file = "./gene_gff/gene_proteinid.txt"
# スプライシングヴァリアントのうち最長の配列のみを利用する場合
gene_protein_file = "./gene_gff/longestGene_proteinid.txt"

protein_seq_file = "./GCF_000001405.40/proteomes/GCF_000001405.40_protein_refseq.faa"
gene_seq_file = "./gene_gff/upstream_10000bp"

TF_outfile = "TF_seq"
TA_outfile = 'TA_seq'

# bp
bp = 1000

#########################


get_amino_seq <- function(tf_id){
  for (j in 1:nrow(rename_tf)){
    if(tf_id == rename_tf[j,2]){
      # rename_tf[j,2]に目的の転写因子IDが入っている
      print(rename_tf[j,1])
      x <- gene_protein[gene_protein$tf_ID == rename_tf[j,2],]
      y <- x[1]
      y <- transform(y, seqence=numeric(nrow(x)))
      for (k in 1:nrow(x)){
        seq <- getSequence(pseq[x[k,2]], as.string = T)
        seq <- gsub("[(\")]", "", toupper(seq))
        seq <- sub("LIST", "", seq)
        y[k,2] <- seq
      }
      write.table(y, "./tmp1", quote = F, row.names = F, col.names = F, sep = "\t")
      system("zsh ./code/Duplicate2.sh")
      y <- read.table("./tmp2", header = F)
      if(nrow(y) != 1){
        for (k in 2:nrow(y)){
          y[k,1] <- sub("$", paste("_",k, sep =""), y[k,1])
        }
      }
      return(y)
    }
  }
}


get_gene_seq <- function(w, ta_id, i){
  for (j in 1:nrow(rename_ta)){
    if(ta_id == rename_ta[j,1]){
      # rename_ta[j,2]に目的の遺伝子IDが入っている
      #seq <- getSequence(gseq[rename_ta[j,2]][[1]][as.numeric(10000-bp):10000], as.string = T)
      seq <- getSequence(gseq[rename_ta[j,2]], as.string = T)
      seq <- gsub("[(\")]", "", toupper(seq))
      seq <- sub("LIST", "", seq)
      y <- transform(w, ta_id=numeric(nrow(w)))
      y <- transform(y, ta_seq=numeric(nrow(w)))
      y <- transform(y, ta_seq=numeric(nrow(w)))
      y <- transform(y, ta_seq=numeric(nrow(w)))
      for (k in 1:nrow(w)){
        y[k,3] <- rename_ta[j,2]
        y[k,4] <- seq
        y[k,5] <- data[i,4]
        y[k,6] <- data[i,1]
      }
      return(y)
    }
  }
}

#########################




######### main #########

# データの読み込み
data <- read.table(paste("../data/", inputfile, sep = ""))
colnames(data) <- c("confidence", "tf", "target", "label")

rename_tf <- read.table(renameTF, header = T)
rename_ta <- read.table(renameTarget, header = T)
gene_protein <- read.table(gene_protein_file, header = F)
colnames(gene_protein) <- c("tf_ID", "protein_ID")
pseq <- read.fasta(protein_seq_file)
gseq <- read.fasta(gene_seq_file)


transcript_id <- unique(data['tf'])
target_id <- unique(data['target'])
transcript_id <- transcript_id[transcript_id != "TF_random"]
target_id <- target_id[target_id != "DNA_random"]
if(0){
get_tf_seq()
get_ta_seq()
}

for (i in 1:nrow(data)){
  print(i)
  if(data[i,2] == "TF_random"){
    TF_name <- sample(transcript_id, 1)
    protein <- gene_protein[gene_protein$tf_ID == TF_name,][1,2]
    seqs <- toupper(getSequence(pseq[protein])[[1]])
    seqs <- seqs[sample(1:length(seqs))]
    seq <- c()
    for (s in 1:length(seqs)){
      seq <- paste(seq, seqs[s], sep = "")
    }
    #seq <- gsub("[(\")]", "", toupper(seq))
    #seq <- sub("LIST", "", seq)
  }else{
  TF_name <- data[i,2]
    protein <- gene_protein[gene_protein$tf_ID == data[i,2],][1,2]
    seq <- getSequence(pseq[protein], as.string = T)
    seq <- gsub("[(\")]", "", toupper(seq))
    seq <- sub("LIST", "", seq)
  }
  if(data[i, 3] == "DNA_random"){
  exit()
    DNA_name <- sample(target_id, 1)
    dseqs <- toupper(getSequence(gseq[DNA_name])[[1]])[9001:10000]
    dseqs <- dseqs[sample(1:length(dseqs))]
    dseq <- c()
    for (s in 1:length(dseqs)){
      dseq <- paste(dseq, dseqs[s], sep = "")
    }
  }else{
  DNA_name <- data[i, 3]
    dseq <- getSequence(gseq[data[i,3]], as.string = T)
    dseq <- gsub("[(\")]", "", toupper(dseq))
    dseq <- sub("LIST", "", dseq)
  }
  result <- cbind(TF_name, seq, DNA_name, dseq, data[i,4], data[i,1])
  write.table(result, paste("../data/", inputfile, "_seq", sep=""), quote = F, row.names = F, col.names = F, sep = "\t", append = T)
}

#########################