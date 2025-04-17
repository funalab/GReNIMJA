TF_list = "../../data/D_TF"
outfile = "../../data/D_TF_group"

setwd("~/m1okubo/1_make_Ans_data/bunpu")

TF_group <- read.table("./TF_bunpu")
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



