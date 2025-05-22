#!/bin/zsh
set -ex

less rename_ta | awk 'NR != 1 {print $2}' | awk 'BEGIN{ORS=","}{print $0}' | sed 's/,$//' > all_target_gene
less cut_gene | awk 'BEGIN{ORS=","}{print $0}' | sed 's/,$//' > target_cut_gene

R --no-save < ./code/kyoutu.R
#rm all_target_gene target_cut_gene

for i in `less use_gene`
do
    less gene_gff/gene_extraction.txt | grep -A 1 -E ">$i\t" >> ./gene_gff/gene_edge.txt
done

R --no-save < ./code/get_chr.R
