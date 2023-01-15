af=as.matrix(read.table("after",sep=",",strip.white=T))
bf=as.matrix(read.table("before",sep=",",strip.white=T))
diff <- setdiff(af,bf)
print(diff)