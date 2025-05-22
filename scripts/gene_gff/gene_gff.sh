# Extract the gene position of each gene
less ../GCF_000001405.39/annotations/GCF_000001405.39_GRCh38.p13_genomic.gff | grep ";gene=.*;" | awk 'BEGIN{FS="\t"} $3 ~/gene/ {print $0}' >! tmp1

# Gene name strand
# Chromosome Start End
# Extract in this format
less tmp1 | awk 'BEGIN{FS="\t"} {print ">" substr($9, index($9,";gene=")+6, index($9, ";gene_biotype=")-index($9,";gene=")-6) "\t" $7 "\n" $1 "\t" $4 "\t" $5}' >! gene_extraction.txt

less ../GCF_000001405.39/annotations/GCF_000001405.39_GRCh38.p13_genomic.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";product=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }' | sed 's/;.*//' | uniq > gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_GRCh38.p13_genomic.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";inference=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_GRCh38.p13_genomic.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";partial=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_GRCh38.p13_genomic.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";pseudo=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt

less gene_proteinid.txt | awk '{if(NF > 1) {print $0}}' | sort | uniq > gene_proteinid2.txt
rm gene_proteinid.txt
mv gene_proteinid2.txt gene_proteinid.txt