protein_file <- "../../1_make_Ans_datas/GCF_000001405.39/proteomes/GCF_000001405.39_protein_refseq.faa"
#dna_file <- "../../1_make_Ans_datas/GCF_000001405.39/genomes/GCF_000001405.39_genomic_refseq.fna"
a_outfile = "amino_seq"
#d_outfile = "dna_seq"

library(seqinr)
library(Biostrings)

#setwd("~/m1okubo/learn_embedding/")


#### main ####
aseq <- read.fasta(protein_file)
#dseq <- readDNAStringSet(dna_file)

for (i in 1:length(aseq)){
  seq <- getSequence(aseq[i], as.string = T)
  seq <- gsub("[(\")]", "", toupper(seq))
  seq <- sub("LIST", "", seq)
  write.table(seq, a_outfile, quote = F, row.names = F, col.names = F, sep = "\t", append = T)
}

if(0){
for (i in 1:length(dseq)){
  if ( length(grep("chromosome [0-9]", names(dseq[i])) ==1 )){
    print(names(dseq[i]))
    if(width(dseq[i])> 10000){
    for (j in 1:as.integer(width(dseq[i]) / 10000)-1){
      seq <- dseq[[i]][as.numeric(j*10000+1) : as.numeric((j+1)*10000)]
      write.fasta(sequences=seq, names = names(dseq[i]), file.out=d_outfile, open="a")
    }}else{
      seq <- dseq[[i]]
      write.fasta(sequences=seq, names = names(dseq[i]), file.out=d_outfile, open="a")
    }
    }
  else{
  }
}

system("zsh edit_seq.sh")
}