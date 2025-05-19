# Extract the gene position of each gene
less ../GCF_000001405.40/annotations/GCF_000001405.40_genomic_refseq.gff | grep ";gene=.*;" | awk 'BEGIN{FS="\t"} $3 ~/gene/ {print $0}' >! tmp1

# > Gene name strand
# Chromosome Start End
# Extract in this format
less tmp1 | awk 'BEGIN{FS="\t”} {print ">” substr($9, index($9,";gene=”)+6, index($9, ";gene_biotype=”)-index($9,";gene=”)-6) "\t” $7 "\n” $1 "\t” $4 "\t” $5}' >! gene_extraction.txt

# Extract all aliases for genes
less tmp1 | grep ";gene_synonym=" | awk 'BEGIN{FS="\t”} {print substr($9, index($9,";gene=”)+6, index($9, ";gene_biotype=”)-index($9,";gene=”)-6) ",” substr($9, index($9, ";gene_synonym=”) +14)}' >! tmp2
less tmp2 | sed 's/;.*//' > tmp3
less tmp3 | sort | uniq > synonym

# There are cases where there are two or more genes with the same name when synonyms are included. Therefore, for such genes, check the details of the genes in the database to determine which gene is which, and perform the following operation to delete the one that does not match from the file. This is tedious, so it is recommended to use the original file.
# Delete "C2orf27B” from line 1997 of synonym (it is a paralog and not an alias) maru
# Delete "DUSP27” from line 4782 maru
# Delete "GGTA1P” from line 6571 maru
# Delete "LINC00846” from line 22700 maru
# Delete "LOR” from line 10221 maru
# Delete "MPP6” from line 12500 maru
# Delete lines 16984 and 16985 (RNA18SN4 and RNA18SN5 cannot be distinguished)
Delete lines #16986 and #16987 (RNA28SN4 and RNA28SN5 cannot be distinguished)
Delete ST2 from line #20194
Delete ST2 from line #22328
Delete TRAP from lines #2587, #20186, #22722, and #23482
#1657 Remove DEC1
#6343 Remove ADC
#10232 Remove AGPAT9
#1448, 20970 Remove AIM1
#2847 Remove APITD1
#1486 Remove B3GNT1
# Delete C10orf2 from line 3042
# Delete C2orf47 from line 5322
# Delete C7orf55 from line 6078
# Delete CD97 from lines 350 and 353 (unable to determine due to absence in web database)
# Delete CGB from line 2986
#15088 Delete CSRP2BP
#3760, 3761 Delete CT45A4
#5081 Delete EFTUD1
#24788 Delete FAM21A
#6429 Delete GARS
# Delete GIF from lines 10972 and 12735
# Delete HIST1H2BC, HIST1H2BE, HIST1H2BF, HIST1H2BG, and HIST1H2BI from line 7229
# Delete HN1 from line 12801
# Delete HN1L from line 3598
#9700 Delete LIMS3L
#21168 Delete LINC00086
#21167, 21168 Delete LINC00087
#20678 Delete MARS
#8747 Delete MUM1
Delete NAT6 from line #7964
Delete NOTCH2NL from lines #13542 and #13543
Delete NOV from lines #15499 and #17904
Delete PMS2L2 from lines #15530 and #15534
Delete QARS from line #5282
# Delete RNA45S5 from line 16987
# Delete SARS from line 20046
# Delete SEPT2 from line 20343
# Delete SHFM1 from line 4481
# Delete SLC35E2 from line 20909
# Delete SMA4 from line 21199
Delete STRA13 from line #1657
Delete TRAPPC2P1 from lines #23506 and #25474
Delete TRIM49D2P from line #23800

rm tmp1 tmp2 tmp3


less ../GCF_000001405.40/annotations/GCF_000001405.40_genomic_refseq.gff | awk 'BEGIN{FS = "\t”} $3 == "CDS” {print substr($9, index($9,";gene=”)+6, index($9, ";product=”)-index($9,”; gene=")-6), ‘\t’, substr($9, index($9, ‘;protein_id=’)+12) }‘ | sed ’s/;.*//' | uniq > gene_proteinid.txt
less ../GCF_000001405.40/annotations/GCF_000001405.40_genomic_refseq.gff | awk 'BEGIN{FS = "\t”} $3 == "CDS” {print substr($9, index($9,";gene=”)+6, index($9, ";inference=”)-index($9,”; gene=")-6), ‘\t’, substr($9, index($9, ‘;protein_id=’)+12) }‘| sed ’s/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.40/annotations/GCF_000001405.40_genomic_refseq.gff | awk 'BEGIN{FS = "\t”} $3 == "CDS” {print substr($9, index($9,";gene=”)+6, index($9, ";partial=”)-index($9,”; gene=")-6), ‘\t’, substr($9, index($9, ‘;protein_id=’)+12) }‘ | sed ’s/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.40/annotations/GCF_000001405.40_genomic_refseq.gff | awk 'BEGIN{FS = "\t”} $3 == "CDS” {print substr($9, index($9,”; gene=")+6, index($9, ‘;pseudo=’)-index($9,”;gene=")-6), ‘\t’, substr($9, index($9, ‘;protein_id=’)+12) }‘| sed ’s/;.*//' | uniq >> gene_proteinid.txt
less gene_proteinid.txt | awk '{if(NF > 1) {print $0}}' | sort | uniq > gene_proteinid2.txt
rm gene_proteinid.txt
mv gene_proteinid2.txt gene_proteinid.txt