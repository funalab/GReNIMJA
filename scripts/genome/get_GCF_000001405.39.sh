#!/bin/zsh

# Download genome data files from FTP server of NCBI
foreach i ( "cds_from_genomic.fna.gz" "genomic.gff.gz" "genomic.fna.gz" "protein.faa.gz" "rna_from_genomic.fna.gz" "assembly_stats.txt" )
TARGET=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_${i}
echo "Try to download from FTP server of NCBI: GCF_000001405.39_GRCh38.p13_${i}"
wget ${TARGET}
sleep 5
end

# Gunzip and get md5sum
gunzip *.gz
md5sum *.txt *.fna *.faa *.gff > md5sum.txt

