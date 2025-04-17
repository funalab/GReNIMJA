# Program to create answer data

#### パラメータの設定 ####
# dorothea load data (これに伴い77行目を書き直す必要あり)
loaddata="entire_database.rda"
# ENCODE input filename
inputfile = "gene_attribute_edges.txt"
# output filename
ans_outfile = "ans_Data"
miss_outfile="miss_Data"
# cutgene file
cutfile="./cut_gene"
# rename
rename_tf = './rename_tf'
rename_ta = './rename_ta'
# confidence assignment
level = "D"

# Ratio of incorrect data to correct data
percentage = 1

#########################


######### 関数 #########
cut_gene <- function(renameTF, renameTA){
  for (i in 1:nrow(cut)){
    dataframe <- dataframe[dataframe$target != cut[i,1],]
  }
  for (i in 1:nrow(renameTF)){
    print(renameTF[i,1])
    dataframe["tf"] <- lapply(dataframe["tf"], gsub, pattern=paste("^", renameTF[i,1], "$", sep = ""), replacement = renameTF[i,2])
    write.table(dataframe, "./tetete")
  }
  for (i in 1:nrow(renameTA)){
    print(renameTA[i,1])
    dataframe["target"] <- lapply(dataframe["target"], gsub, pattern=paste("^", renameTA[i,1], "$", sep = ""), replacement = renameTA[i,2])
  }
  return(dataframe)
}

cut_gene_ENCODE <- function(renameTF, renameTA){
  for (i in 1:nrow(cut)){
    print(paste("before: ", nrow(data), sep = ""))
    print(cut[i,1])
    data <- data[data$source != cut[i,1],]
    print(paste("after: ", nrow(data), sep = ""))
  }
  
  for (i in 1:nrow(renameTF)){
    data["target"] <- lapply(data["target"], gsub, pattern=paste("^", renameTF[i,1], "$", sep = ""), replacement = renameTF[i,2])
  }
  for (i in 1:nrow(renameTA)){
    print(renameTA[i,1])
    data["source"] <- lapply(data["source"], gsub, pattern=paste("^", renameTA[i,1], "$", sep = ""), replacement = renameTA[i,2])
  }
  return(data)
}

make_ans_data <- function(lev){  # 設定した信頼レベル以下のデータを切り捨てた後に正解データを作成する
  dataframe <- cut_dataframe[cut_dataframe$confidence <= lev,]
  list <- c("confidence", "tf", "target") # dorothea_hsから抽出する列のリスト
  x <- dataframe[, list] # 抽出
  correct <- transform(x, label=numeric(nrow(dataframe))+1) # tfとtargetに正解ラベルである1だけの列を足す
  write.table(correct, file = paste("../data/", ans_outfile, sep = ""), sep = "\t", append=F, quote=F, row.names = FALSE) # ファイル出力
  return(dataframe) 
}


make_miss_data <- function(percent){
  # tfを重複なしでunitfに保存
  unitf <- unique(dataframe['tf'])
  # targetを重複無しでunitarに保存
  unitar <- unique(dataframe['target'])
  # tmp1: target遺伝子の一覧
  write.table(unitar, './tmp1', sep = "\t", append=F, quote=F, row.names = FALSE)
  
  for (i in 1:nrow(unitf)){
    print(paste("i: ", i, sep = ""))
    # ある特定のtfのtargetをtmpに保存
    tmp <- dataframe[dataframe$tf == unitf$tf[i],]
    tmptar <- tmp['target']
    # tmp2: 注目しているTFのtarget遺伝子一覧
    write.table(tmptar, './tmp2', sep = "\t", append=F, quote=F, row.names = FALSE)
    
    # 着目しているtfのtargetではないtargetをランダムに抽出してくる
    system("zsh ./code/csv.sh")
    x <- as.matrix(read.table("./tmp3",sep=",",strip.white=T))
    y <- as.matrix(read.table("./tmp4",sep=",",strip.white=T))
    # diff: 注目しているTFのtargetではない全てのtarget遺伝子
    diff <- setdiff(x,y)
    #misstar <- setdiff(x,y)

    #if( length(diff) > nrow(tmp) + 300){
    #misstar <- sample(diff, nrow(tmp) + 300)
    #misstf <- rep(unitf$tf[i], nrow(tmp) + 300)
    #conf <- rep("M", nrow(tmp) + 300)
    #}
    if(length(diff) > nrow(tmp)){
      misstar <- sample(diff, nrow(tmp))
      misstf <- rep(unitf$tf[i], nrow(tmp))
      conf <- rep("M", nrow(tmp))
    }
    else{
      misstar <- diff
      misstf <- rep(unitf$tf[i], length(diff))
      conf <- rep("M", length(diff))
    }
    
    #misstf <- rep(unitf$tf[i], length(misstar))
    #conf <- rep("M", length(misstar))
    
    # 誤りデータをデータフレームとして保存  
    incorrect <- data.frame("confidence" = conf, "tf" = misstf, "target" = misstar, "label" = numeric(length(misstar))) 
    # ファイル出力
    write.table(incorrect, file = paste("../data/", level, "_", miss_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
  }
  file.remove("tmp1")
  file.remove("tmp2")
  file.remove("tmp3")
  file.remove("tmp4")
}

make_ans_data_ENCODE <- function(){
  list <- c("source_desc", "target", "source") # dorothea_hsから抽出する列のリスト
  data <- cut_data[, list] # 抽出
  data<-data[-1,] # １列目の削除
  colnames(data) <- c("confidence", "tf", "target")
  correct <- transform(data, label=numeric(nrow(data))+1) # tfとtargetに正解ラベルである1だけの列を足す
  write.table(correct, file = paste("../data/", ans_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
  return(data) 
}

make_miss_data_ENCODE <- function(level){
  # tfを重複なしでunitfに保存
  unitf <- unique(data['target'])
  # targetを重複無しでunitarに保存
  unitar <- unique(data['source'])
  write.table(unitar, './tmp1', sep = "\t", append=F, quote=F, row.names = FALSE)
  
  for (i in 1:nrow(unitf)){
    print(i)
    # ある特定のtfのtargetをtmpに保存
    tmp <- data[data$source == unitf$target[i],]
    tmptar <- tmp['target']
    write.table(tmptar, './tmp2', sep = "\t", append=F, quote=F, row.names = FALSE)
    
    # 着目しているtfのtargetではないtargetをランダムに抽出してくる
    system("zsh ./code/csv.sh")
    x <- as.matrix(read.table("./tmp3",sep=",",strip.white=T))
    y <- as.matrix(read.table("./tmp4",sep=",",strip.white=T))
    diff <- setdiff(x,y)
    print(length(diff))
    print(nrow(tmp))
    misstar <- sample(diff, nrow(tmp) * percent, replace = T)
    misstf <- rep(unitf$target[i], nrow(tmp) * percent)
    print(length(misstar))
    print(length(misstf))
    incorrect <- data.frame("tf" = misstf, "target" = misstar, "label" = numeric(length(misstar))) 
    # ファイル出力
    write.table(incorrect, file = paste("./", outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
  }
  file.remove("tmp1")
  file.remove("tmp2")
  file.remove("tmp3")
  file.remove("tmp4")
}

#########################





######### main #########

args1 = commandArgs(trailingOnly=TRUE)[1]
setwd(paste("./saezlab-dorothea-", args1, "/data", sep = ""))
# load data
load(loaddata)
dataframe <- entire_database
write.table(dataframe, "./tmp.txt", quote = F, col.names = F, row.names = F)

setwd("../../")

cut <- read.table(cutfile) # cutする遺伝子のリストを格納
renameTF = read.table(rename_tf, header = T)
renameTA = read.table(rename_ta, header = T)
renameTF <- renameTF[renameTF$name != renameTF$rename,]
renameTA <- renameTA[renameTA$name != renameTA$rename,]


cut_dataframe <- cut_gene(renameTF, renameTA)  # gene cut
write.table(dataframe, file = './cut.txt', sep = "\t", append=F, quote=F, row.names = FALSE)  # cutした結果を出力
# Create answer data after deleting data below the set confidence assignment level.
dataframe <- make_ans_data(level)  # dorotheaの正解データを作成

data <- read.table("gene_attribute_edges.txt", header = T)  # ENCODEのデータを読み込み
data <- data.frame(data)
cut_data <- cut_gene_ENCODE(renameTF, renameTA)  # gene cut
data <- make_ans_data_ENCODE()

system(paste("zsh ../data/Duplicate.sh ", ans_outfile, sep=""))

dataframe <- read.table(paste("../data/", ans_outfile, sep=""))
colnames(dataframe) <- c("confidence", "tf", "target", "label")
make_miss_data(percentage)

#########################