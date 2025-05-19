#!/bin/zsh

# Download genome data files from FTP server of NCBI
FILES=("cds_from_genomic.fna.gz" "genomic.gff.gz" "genomic.fna.gz" "protein.faa.gz" "rna_from_genomic.fna.gz" "assembly_stats.txt")
DIRS=("CDSs" "annotations" "genomes" "proteomes" "RNA" ".")

# function to add directory
addir () {
    if [ ! -e $* ];then
        mkdir -p $*
        echo " Generated: $*"
    fi
}

# Make output directory
foreach dir (${DIRS})
addir ../GCF_000001405.39/${dir}
end

foreach i (`seq 1 6`)
TARGET=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_${FILES[$i]}
echo "Try to download from FTP server of NCBI: GCF_000001405.39_GRCh38.p13_${TARGET}"
wget -P ../GCF_000001405.39/${DIRS[$i]} ${TARGET}
sleep 5
end

# Gunzip and get md5sum
gunzip ../GCF_000001405.39/*/*.gz
md5sum ../GCF_000001405.39/*/* > ../GCF_000001405.39/md5sum.txt

