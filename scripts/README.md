
1. Acquisition and deployment of human genome data

```
R --no-save < ./code/gene_download.R
find ./ -type f -name “*.gz” -exec gunzip {} \; 
```
Note: At the time of this study, GCF_000001405.39 was the latest version, so we obtained the genome information from GCF_000001405.39.
However, gene_download.R does not work with an older version of the genome, so if you want to obtain the latest version of the genome, please modify line 28 of gene_download.R.
Additionally, you will need to replace the first occurrence of “GCF*” in get_chr.R, get_seq.R, and get_longest_seq.R.
The GCF_00001505.3 used in this study is already prepared in the directory. If you are using this genome information to create the dataset, skip this step and proceed to step 2 below.


2. Download and unzip data from dorothea
Download from https://api.github.com/repos/saezlab/dorothea/tarball/HEAD
The file name changes each time you download, so enter the file name yourself when unzipping.

```
tar -zxvf [file name]
```

3. Download and unzip data from ENCODE

```
curl -O https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_attribute_edges.txt.gz
gunzip gene_attribute_edges.txt.gz
```

4. Map gene names in the database to NCBI gene names

```
cd ./gene_gff
zsh gene_gff.sh  # Consolidate synonyms. The operations described in gene_gff.sh are necessary, but they are tedious, so we do not recommend them. We recommend executing only the first two commands and the last seven lines.
cd ../
```

5. Create a mapping table (the following commands will automatically execute the rest of the process)
```
zsh matome.sh [last 7 digits of dorothea file name]
```

# If executing one by one, execute as follows
```
Rscript ./code/rename.R [last 7 digits of dorothea file name]  # Create correspondence table

# Delete “ZNF286B” from rename_tf (because it is a pseudogene with no CDS)
less rename_tf | grep -v “ZNF286B” > rename_tf2
rm rename_tf
mv rename_tf2 rename_tf

6. Get the upstream 1000 bp of the gene

```
zsh ./code/get_not_cutGene.sh
```


# Delete genes with the same position but different names (considering strand)
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene
```

7. Delete genes for which synonyms could not be matched, and label the correct TF-target relationships and randomly selected incorrect TF-target relationships with 1 and 0, respectively

```
Rscript ./code/make_data.R [last 7 digits of dorothea file name]
```


8. Take the longest protein considering splicing variants
```
R --no-save < ./code/get_longest_seq.R
```


9. Take the sequence of the specified upstream bp and the TF sequence
```
Rscript ./code/get_seq.R ans_Data
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py
```

9. Obtain the TF length distribution (for batch learning)
```
R --no-save <  bunpu/TF_long_bunpu.R
```

