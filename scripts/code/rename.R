# rename gene_synonym gene_name

#### パラメータの設定 ####
# dorothea load data (これに伴い21行目を書き直す必要あり)
loaddata="entire_database.rda"
# inputfile
inputfile <- "./gene_gff/synonym"
tf_file <- "./tftmp2"
target_file <- "./tatmp2"
geneName_file <- "gene_name"
# outputfile
outfile <- "rename_ans_Data"
#########################


######### main #########
args1 = commandArgs(trailingOnly=TRUE)[1]
print(args1)
setwd(paste("saezlab-dorothea-", args1, "/data", sep = ""))
# load data
load(loaddata)
dataframe <- entire_database
write.table(dataframe, file = './tmp.txt', sep = "\t", append=F, quote=F, row.names = FALSE)
setwd("../../")
system(paste("zsh ./code/rename.sh", args1, sep=" "))

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

tf <- read.table(tf_file)
ta <- read.table(target_file)
gene_name <- read.table(geneName_file)


rename = c()
cut_gene = c()
#rename = rep("AA", 27770)
for (i in 1:nrow(ta)){
  print(paste("a", i, ": ", ta[i,1], sep = ""))
  if(ta[i,1] %in% gene_name[[1]]){  # すでに本来の遺伝子名だったら
    rename <- append(rename, ta[i,1])
    #print(paste("i: ", i, "   rename: ", length(rename), sep = ""))
  }
  else if(sub("ORF", "orf", ta[i,1]) %in% gene_name[[1]]){
    rename <- append(rename, sub("ORF", "orf", ta[i,1]))
    print(paste(ta[i,1], "->", sub("ORF", "orf", ta[i,1]), sep="" ))
  }
  else{  # そうではなかったら(名前を変える必要があったら)
  for (j in 1:length(data.list)){
    if( ta[i,1] %in% data.list[[j]]){
      rename <- append(rename, data.list[[j]][1])
      print(paste(ta[i,1], "->", data.list[[j]][1], sep="" ))
      
      #print(paste("b", ta[i,1], ": ", data.list[[j]][1], sep = ""))
      #print(paste("i: ", i, "   rename: ", length(rename), sep = ""))
    }
  }
      if(length(rename) != i){
        print("Cut Gene")
        rename <- append(rename, ta[i,1])
        print(paste("i: ", i, "   rename: ", length(rename), sep = ""))
        cut_gene <- append(cut_gene, ta[i,1])
        if(length(rename)!=i){
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          print("\n")
          print("\n")
          print("\n")
          print("\n")
          exit
        }
      }
  }
}
rename_ta <- cbind(ta, rename)
name <- cbind("name", "rename")
names(rename_ta) <- name
cut_gene <- unique(cut_gene)
write.table(rename_ta,file = "rename_ta", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(cut_gene,file = "cut_gene", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)

rename = c()
cut_gene = c()
for (i in 1:nrow(tf)){
  print(paste("a", i, ": ", tf[i,1], sep = ""))
  if(tf[i,1] %in% gene_name[[1]]){  # すでに本来の遺伝子名だったら
    rename <- append(rename, tf[i,1])
  }
  else{  # そうではなかったら(名前を変える必要があったら)
    for (j in 1:length(data.list)){
      if( tf[i,1] %in% data.list[[j]]){
        rename <- append(rename, data.list[[j]][1])
        print(paste("b", tf[i,1], ": ", data.list[[j]][1], sep = ""))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      }
    }
    if(length(rename) != i){
      print("CCC")
      rename <- append(rename, tf[i,1])
      cut_gene <- append(cut_gene, tf[i,1])
    }
  }
}
rename_tf <- cbind(tf, rename)
name <- cbind("name", "rename")
names(rename_tf) <- name
write.table(rename_tf,file = "rename_tf", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(cut_gene,file = "cut_gene", append = TRUE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
file.remove("tftmp2")
file.remove("tatmp2")
#########################