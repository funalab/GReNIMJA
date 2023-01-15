less ./saezlab-dorothea-$1/data/tmp.txt | awk 'NR!=1{print $1}' | sort | uniq > ./tftmp 
less ./saezlab-dorothea-$1/data/tmp.txt | awk 'NR!=1{print $2}' | sort | uniq > ./tatmp 
less gene_attribute_edges.txt | awk 'NR>2{print $4}' | sort | uniq >> tftmp
less gene_attribute_edges.txt | awk 'NR>2{print $1}' | sort | uniq >> tatmp 
less tftmp | sort | uniq > tftmp2
less tatmp | sort | uniq > tatmp2
less gene_gff/gene_extraction.txt | grep ">" | awk 'NR>2{print substr($1, 2)}' > gene_name 
rm tftmp tatmp
