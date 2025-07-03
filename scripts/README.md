# Description
Scripts for downloading raw data and formatting it to use as a dataset

(*) Please make sure ```R``` and ```Python``` are downloaded before running the following scripts.
The scripts have been tested on mac OS using ```R 4.4.2``` and ```Python 3.10.11```.

# Install requirements
```
% python -m venv venv_build
% source ./venv_build/bin/activate
% pip install --upgrade pip
% pip install pandas==2.2.3
```

# Build datasets

If you want to execute all scripts at once, execute the following command. 
```
zsh ./build.sh
```

If you want to execute them one by one, follow the instructions below.

1. Download human genome data (accession number: ```GCF_000001405.39```) from FTP server of NCBI
```
cd ./genome
zsh get_GCF_000001405.39.sh
cd ../
```
<!--
```
R --no-save < ./code/gene_download.R
find ./ -type f -name "*.gz" -exec gunzip {} \; 
```
Note: At the time of this study, ```GCF_000001405.39``` was the latest version, 
so we obtained the genome information from ```GCF_000001405.39```.
However, ```./code/gene_download.R``` does not work with the above version at now, 
so we need to modify the ```./code/gene_download.R``` to download ```GCF_000001405.40```.
Additionally, we need to modify the relevant files in ```./code/get_chr.R```, ```./code/get_seq.R```, 
```./code/get_longest_seq.R```, and ```./_gene_gff/gene_gff.sh```.
-->

2. Download and unzip data from ```dorothea```  
<!--
Download from https://api.github.com/repos/saezlab/dorothea/tarball/HEAD.
-->
```
curl --output dorothea.tar.gz "https://codeload.github.com/saezlab/dorothea/legacy.tar.gz/HEAD"
tar -zxvf dorothea.tar.gz
```
Make a note of the last 7 digits of ```dorothea``` file name, as you will need them later.

3. Download and unzip data from ```ENCODE```

```
curl -O https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_attribute_edges.txt.gz
gunzip gene_attribute_edges.txt.gz
```

4. Map gene names in the database to NCBI gene names  

```
cd ./gene_gff
zsh gene_gff.sh
cd ../
```

In this repository, post-processing for synonym has already been performed and 
the files have been shared under ```./gene_gff/synonym```. 
If you are interested in the post-processing for synonym, 
please refer``` ./_gene_gff/gene_gff.sh```. 
You will need to manually delete lines with duplicate gene names.


5. Create a mapping table
```
zsh mapping.sh [last 7 digits of dorothea file name]
```
<!--
If executing one by one, execute as follows
```
Rscript ./code/rename.R [last 7 digits of dorothea file name]

# Delete “ZNF286B” from rename_tf (because it is a pseudogene with no CDS)
less rename_tf | grep -v “ZNF286B” > rename_tf2
rm rename_tf
mv rename_tf2 rename_tf
```
-->
6. Get the upstream 1000 bp of the gene
```
zsh ./code/get_not_cutGene.sh
```

7. Delete genes with the same position but different names (considering strand)
```
python ./code/cut_gene.py
less cut_gene |grep -v -e '^\s*#' -e '^\s*$' > tmp
rm cut_gene
mv tmp cut_gene
```

8. Delete genes for which synonyms could not be matched, and label the correct TF-target relationships and randomly selected incorrect TF-target relationships with 1 and 0, respectively

```
Rscript ./code/make_data.R [last 7 digits of dorothea file name]
```


9. Take the longest protein considering splicing variants
```
R --no-save < ./code/get_longest_seq.R
```

10. Take the sequence of the specified upstream bp and the TF sequence
```
Rscript ./code/get_seq.R ans_Data
Rscript ./code/get_seq.R D_miss_Data
python ./code/Data_toPickle.py
```

11. Obtain the TF length distribution (for batch learning)
```
R --no-save <  bunpu/TF_long_bunpu.R
```

