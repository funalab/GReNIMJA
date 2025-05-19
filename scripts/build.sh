#!/bin/zsh

# 1. Download human genome data (accession number: ```GCF_000001405.39```) from FTP server of NCBI
zsh ./genome/get_GCF_000001405.39.sh
# 2. Download and unzip data from ```dorothea```
curl --output dorothea_download.tar.gz "https://codeload.github.com/saezlab/dorothea/legacy.tar.gz/HEAD"
tar -zxvf dorothea_download.tar.gz
filename=$(find . -type d -name "saezlab-dorothea-*" -exec basename {} \; | grep -oE '[a-f0-9]{7}$')
# 3. Download and unzip data from ```ENCODE```
curl -O https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_attribute_edges.txt.gz
gunzip gene_attribute_edges.txt.gz
# 5. Create a mapping table
zsh mapping.sh $filename
# 6. Get the upstream 1000 bp of the gene
zsh ./code/get_not_cutGene.sh
# 7. Delete genes with the same position but different names (considering strand)
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene
# 8. Delete genes for which synonyms could not be matched, and label the correct TF-target relationships and randomly selected incorrect TF-target relationships with 1 and 0, respectively
Rscript ./code/make_data.R $filename
# 9. Take the longest protein considering splicing variants
R --no-save < ./code/get_longest_seq.R
# 10. Take the sequence of the specified upstream bp and the TF sequence
Rscript ./code/get_seq.R ans_Data
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py
# 11. Obtain the TF length distribution (for batch learning)
R --no-save <  bunpu/TF_long_bunpu.R $filename