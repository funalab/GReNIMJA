#!/bin/zsh
set -ex

# First argument: Last 7 digits of the downloaded dorothea

# Create a table for renaming genes
Rscript ./code/rename.R $1

## Remove “ZNF286B” from rename_tf (because it is a pseudogene with no CDS)
less rename_tf | grep -v "ZNF286B" > rename_tf2
rm rename_tf
mv rename_tf2 rename_tf
zsh ./code/get_not_cutGene.sh

# Delete genes with the same position but different names (strand is considered)
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene

# Create correct data (gene names and labels only)
Rscript ./code/make_data.R $1

# Get the longest protein
R --no-save < ./code/get_longest_seq.R

# Get the sequence for the specified upstream bp and the TF sequence
Rscript ./code/get_seq.R ans_Data
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py # Compress to pickle

# Get TF length distribution (for batch learning)
R --no-save <  ./code/TF_bunpu.R
