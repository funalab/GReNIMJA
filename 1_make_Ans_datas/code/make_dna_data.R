# Program to create answer data

#### パラメータの設定 ####
# dorothea load data (これに伴い77行目を書き直す必要あり)
loaddata="entire_database.rda"
# ENCODE input filename
inputfile = "gene_attribute_edges.txt"
# output filename
ans_outfile = "ans_Data"
miss_outfile="dna_miss_Data2"
# cutgene file
cutfile="./cut_gene"
# rename
rename_tf = './rename_tf'
rename_ta = './rename_ta'
# confidence assignment
level = "D"

# Ratio of incorrect data to correct data
percentage = 0.8

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


make_ans_data_ENCODE <- function(){
  list <- c("source_desc", "target", "source") # dorothea_hsから抽出する列のリスト
  data <- cut_data[, list] # 抽出
  data<-data[-1,] # １列目の削除
  colnames(data) <- c("confidence", "tf", "target")
  correct <- transform(data, label=numeric(nrow(data))+1) # tfとtargetに正解ラベルである1だけの列を足す
  write.table(correct, file = paste("../data/", ans_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
  return(data) 
}

make_miss_data <- function(percent){
  # tfを重複なしでunitfに保存
  unitf <- unique(dataframe['tf'])
  freq <- c()
  freq2 <- c()
  count <- c()
  remain <- tflist$tf
  for (l in 1:nrow(unitf)){
    freq <- append(freq, nrow(dataframe[dataframe$tf == unitf[l, 1],]))
    freq2 <- append(freq2, nrow(dataframe[dataframe$tf == unitf[l, 1],]) ^ 2)
    count <- append(count, 0)
  }
  tflist <- transform(unitf, "freq" = freq)
  tflist <- transform(tflist, "freq2" = freq2)
  tflist <- transform(tflist, "count" = count)

  tflist1 <- tflist[tflist$freq > 8000,]
  tflist2 <- tflist[tflist$freq < 8001,]
  tflist2 <- tflist2[tflist2$freq > 3000,]
  tflist3 <- tflist[tflist$freq < 3001,]
  tflist3 <- tflist3[tflist3$freq > 1000,]
  tflist4 <- tflist[tflist$freq < 1001,]
  tflist4 <- tflist4[tflist4$freq > 20,]
  tflist5 <- tflist[tflist$freq < 21,]
  tflist5 <- tflist5[tflist5$freq > 5,]
  tflist6 <- tflist[tflist$freq < 6,]
  tflist6 <- tflist6[tflist6$freq > 1,]
  tflist7 <- tflist[tflist$freq < 2,]
  print(nrow(tflist1) + nrow(tflist2) + nrow(tflist3) + nrow(tflist4) + nrow(tflist5) + nrow(tflist6) + nrow(tflist7))
  remain <- tflist$tf
  # targetを重複無しでunitarに保存(データ数が多い順に並び替える)
  unitar <- unique(dataframe['target'])
  dna_freq <- c()
  for (l in 1:nrow(unitar)){
    dna_freq <- append(dna_freq, nrow(dataframe[dataframe$target == unitar[l, 1],]))
  }
  tar <- transform(unitar, "freq" = dna_freq)
  unitar <- data.frame("target" = tar[order(tar$freq,decreasing = FALSE),]$target)

  for (i in 1:nrow(unitar)){
    print(paste("i: ", i, sep = ""))
    # ある特定のtfのtargetをtmpに保存
    tmp <- dataframe[dataframe$target == unitar$target[i],]
    tmptf <- tmp['tf']
    
    # 着目しているtfのtargetではないtargetをランダムに抽出してくる
    # diff: 注目しているTFのtargetではない全てのtarget遺伝子
    #diff <- setdiff(as.matrix(unitf), as.matrix(tmptf))
    #diff <- intersect(diff, remain)
    #diff1 <- intersect()
    misstf <- c()
    
    for(lists in list(tflist1, tflist2, tflist3, tflist4, tflist5, tflist6, tflist7)){
      diff <- setdiff(as.matrix(lists$tf), as.matrix(tmptf))
      diff <- intersect(diff, remain)
      if(max(1, as.integer(nrow(tmp) * percentage)) - length(misstf) != 0){
      if(length(diff) > as.integer(nrow(tmp) * percentage) - length(misstf)){
        f <- c()
        for (l in 1:length(diff)){
          f <- append(f, tflist[tflist$tf == diff[l],][[3]])
        }
        misstf <- append(misstf, sample(diff, max(as.integer(nrow(tmp) * percentage) - length(misstf), 1), prob = f))
      }else if(length(misstf) < nrow(tmp)){
        misstf <- append(misstf, diff)
      }
      }
    }
    if(length(misstf) < nrow(tmp)){
      print(unitar$target[i])
      print(length(misstf))
      print(nrow(tmp))
    }
    misstar <- rep(unitar$target[i], length(misstf))
    conf <- rep("M", length(misstf))
    
if(0){
    if(length(diff) > nrow(tmp) * percentage){
      f <- c()
      for (l in 1:length(diff)){
        f <- append(f, tflist[tflist$tf == diff[l],][[3]])
      }
      #misstf <- sample(diff, nrow(tmp))
      misstf <- sample(diff, max(nrow(tmp) * percentage, 1), prob = f)
      misstar <- rep(unitar$target[i], max(nrow(tmp) * percentage, 1))
      conf <- rep("M", max(nrow(tmp) * percentage, 1))
    }else{
      print(unitar$target[i])
      misstf <- diff
      misstar <- rep(unitar$target[i], length(diff))
      conf <- rep("M", length(diff))
    }
}
    
    # 誤りデータをデータフレームとして保存  
    incorrect <- data.frame("confidence" = conf, "tf" = misstf, "target" = misstar, "label" = numeric(length(misstar))) 
    # ファイル出力
    write.table(incorrect, file = paste("../data/", level, "_", miss_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
    
    if(length(misstf) != 0){
    # もうpositiveの個数と同じになったTFは除外する
    for (l in 1:length(misstf)){
      tflist[tflist$tf == misstf[l],][[4]] <- tflist[tflist$tf == misstf[l],][[4]] + 1
    }
    remove <- tflist[tflist$freq == tflist$count,]$tf
    remain <- setdiff(tflist$tf, remove)
    }else{
      print(unitar$target[i])
    }
  }
}
#########################


######### main #########

setwd("~/m1okubo/1_make_Ans_data/")　# この行は実際はいらない
dir.create("../data")
setwd("./saezlab-dorothea-a814f45/data")
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



make_miss_data(level)

# 正例と負例のバランスチェック
miss_dataframe <- read.table(paste("../data/", level, "_", miss_outfile, sep = ""))
colnames(miss_dataframe) <- c("confidence", "tf", "target", "label")
data <- rbind(dataframe, miss_dataframe)

unitar <- unique(dataframe['target'])
unitf <- unique(dataframe['tf'])

# DNA配列について
ratio <- c()
difference <- c()
for(i in 1:nrow(unitar)){
  tmp <- data[data$target == unitar[i,1],]
  posi <- nrow(tmp[tmp$label == 1,])
  ratio <- append(ratio, posi / nrow(tmp))
  
  difference <- append(difference, posi - nrow(tmp[tmp$label == 0,]))
}
pdf(file = paste("../data/1_0histgram", level, "_", miss_outfile, ".pdf", sep = ""))
hist(ratio, breaks=seq(0,1,0.1), main="Histogram", xlab="range", col="#993435")
dev.off()
# 転写因子について
tf_ratio <- c() 
tf_difference <- c()
for(i in 1:nrow(unitf)){
  tmp <- data[data$tf == unitf[i,1],]
  posi <- nrow(tmp[tmp$label == 1,])
  tf_ratio <- append(tf_ratio, posi / nrow(tmp))
  
  tf_difference <- append(tf_difference, posi - nrow(tmp[tmp$label == 0,]))
}
pdf(file = paste("../data/1_0TFhistgram", level, "_", miss_outfile, ".pdf", sep = ""))
hist(tf_ratio, breaks=seq(0,1,0.1), main="Histogram", xlab="range", col="#993435")
dev.off()


# negativeデータが足りない分のTFを追加する
unitar <- transform(unitar, "difference" = difference)
for( i in 1:nrow(unitar)){
  misstar <- rep(unitar$target[i], unitar$difference[i])
  misstf <- rep("TF_random", unitar$difference[i])
  conf <- rep("T", unitar$difference[i])
  incorrect <- data.frame("confidence" = conf, "tf" = misstf, "target" = misstar, "label" = numeric(unitar$difference[i])) 
  # ファイル出力
  write.table(incorrect, file = paste("../data/", level, "_", miss_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
}

# negativeデータが足りない分の遺伝子を追加する
unitf <- transform(unitf, "difference" = tf_difference)
for( i in 1:nrow(unitf)){
  misstf <- rep(unitf$tf[i], unitf$difference[i])
  misstar <- rep("DNA_random", unitf$difference[i])
  conf <- rep("T", unitf$difference[i])
  incorrect <- data.frame("confidence" = conf, "tf" = misstf, "target" = misstar, "label" = numeric(unitf$difference[i])) 
  # ファイル出力
  write.table(incorrect, file = paste("../data/", level, "_", miss_outfile, sep = ""), sep = "\t", append=T, quote=F, row.names = FALSE, col.names = FALSE) # ファイル出力
}

# 正例と負例のバランスチェック
miss_dataframe <- read.table(paste("../data/", level, "_", miss_outfile, sep = ""))
colnames(miss_dataframe) <- c("confidence", "tf", "target", "label")
data <- rbind(dataframe, miss_dataframe)

# DNA配列について
ratio <- c()
difference2 <- c()
for(i in 1:nrow(unitar)){
  tmp <- data[data$target == unitar[i,1],]
  posi <- nrow(tmp[tmp$label == 1,])
  ratio <- append(ratio, posi / nrow(tmp))
  
  difference2 <- append(difference2, posi - nrow(tmp[tmp$label == 0,]))
}
pdf(file = paste("../data/final_histgram", level, "_", miss_outfile, ".pdf", sep = ""))
hist(ratio, breaks=seq(0,1,0.1), main="Histogram", xlab="range", col="#993435")
dev.off()
# 転写因子について
tf_ratio <- c() 
tf_difference2 <- c()
for(i in 1:nrow(unitf)){
  tmp <- data[data$tf == unitf[i,1],]
  posi <- nrow(tmp[tmp$label == 1,])
  tf_ratio <- append(tf_ratio, posi / nrow(tmp))
  
  tf_difference2 <- append(tf_difference2, posi - nrow(tmp[tmp$label == 0,]))
}
pdf(file = paste("../data/final_TFhistgram", level, "_", miss_outfile, ".pdf", sep = ""))
hist(tf_ratio, breaks=seq(0,1,0.1), main="Histogram", xlab="range", col="#993435")
dev.off()


#########################