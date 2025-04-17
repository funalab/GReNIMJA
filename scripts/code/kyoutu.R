af=as.matrix(read.table("all_target_gene",sep=",",strip.white=T))
bf=as.matrix(read.table("target_cut_gene",sep=",",strip.white=T))
diff <- setdiff(af,bf)
write.table(diff, "./use_gene", row.names = F, col.names=F, quote=F)