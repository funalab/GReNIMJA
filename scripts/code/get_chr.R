# targetのうち、NCBIのgeneIDと対応づけることができたものだけのスキャフォールドの位置をピックアップし、
# 指定した塩基bp分(10000bp)遡ったfastaファイルを用意する(今後好きなbpを指定した時に高速化するため)


######### package ######### 
requiredPackage_BC <- c("Biostrings")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

for( p in requiredPackage_BC ){
  if( !( p %in% installed.packages( ))){
    BiocManager::install( p )
  }
}

library(seqinr)
library(Biostrings)
#########################


#### パラメータの設定 ####
# input filename
#inputfile = ""
inputfile = "gene_edge.txt"
gff_file="new_gene_edge.txt"
genome_file="../GCF_000001405.40/genomes/GCF_000001405.40_genomic_refseq.fna"

# output filename
out_edge_file= "new_gene_edge.txt"
out_bp_file="upstream_10000bp"

# bp
bp = 1000
#########################

######### 関数 #########
reverse_RNA <- function(pre_mrna){
  mrna <- rev(pre_mrna)
  mrna <- gsub("A", "1", mrna)
  mrna <- gsub("T", "2", mrna)
  mrna <- gsub("G", "3", mrna)
  mrna <- gsub("C", "4", mrna)
  mrna <- gsub("1", "T", mrna)
  mrna <- gsub("2", "A", mrna)
  mrna <- gsub("3", "C", mrna)
  mrna <- gsub("4", "G", mrna)
  return(mrna)
}
#########################



######### main #########

#setwd("~/m1okubo/1_make_Ans_data/")  # この行は実際はいらない
setwd("./gene_gff/")

# ファイルの読み込み
f <- file(inputfile, "r")
lines <- readLines(con=f)

data.list <- list()
for (i in 1:length(lines)) {
  #読み込んだ行をコンマで分割して、ベクトル化
  line.vec <- strsplit(lines[i], ",")
  data.list <- c(data.list, line.vec)
}
close(f)

j = 1
while(j <= length(data.list)){
  if (regexpr("--",data.list[j] ) != TRUE){
    write.table(data.list[j], file = paste("./", out_edge_file, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
    write.table(data.list[j+1], file = paste("./", out_edge_file, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
    j = j + 2
  }
  else{
    j = j + 3
  }
}


# 遺伝子名と染色体の位置の読み込み
augustusfile <- as.matrix( read.delim(gff_file, header=F) )
# genomeデータの読み込み
seq <- readDNAStringSet(genome_file)

for (i in 1:nrow(augustusfile)){
  print(i)
  if(i %% 2 == 1){
    gene_name <- substring(augustusfile[i,1], 2)
    mode <- augustusfile[i,2]
  }
  else{
    scafname <- augustusfile[i,1]
    for (j in 1:length(seq)){
      if(sub(" .*", "", names(seq[j])) == scafname){
        if(mode == "-"){
          if(as.numeric(augustusfile[i,3]) != nchar(seq[j])){
            s <- as.numeric(augustusfile[i,3])+1
            if(as.numeric(augustusfile[i,3])+bp > nchar(seq[j])){
              f <- as.numeric(nchar(seq[j]))
            }
            else{
              f <- as.numeric(augustusfile[i,3])+bp
            }
            fa <- seq[[j]][as.numeric(s):as.numeric(f)]
            fa <- reverseComplement(fa)
            write.fasta(sequences=fa, names=gene_name, file.out=out_bp_file, open="a")
          }
        }
        else{
          if(as.numeric(augustusfile[i,2]) > bp + 1){
            s <- as.numeric(augustusfile[i,2])-bp
          }
          else{
            s <- 0
          }
          f <- as.numeric(augustusfile[i,2])-1
          fa <- seq[[j]][as.numeric(s):as.numeric(f)]
          write.fasta(sequences=fa, names=gene_name, file.out=out_bp_file, open="a")
        }
      }
    }
  }
}

#########################
