# それぞれの遺伝子のgeneの位置を抜き出す
less ../GCF_000001405.39/annotations/GCF_000001405.39_genomic_refseq.gff | grep ";gene=.*;" | awk 'BEGIN{FS = "\t"} $3 ~/gene/ {print $0}' >! tmp1

# >遺伝子名 strand
# 染色体 開始 終わり
# の形で抜き出す
less tmp1 | awk 'BEGIN{FS="\t"} {print ">" substr($9, index($9,";gene=")+6, index($9, ";gene_biotype=")-index($9,";gene=")-6) "\t" $7 "\n" $1 "\t" $4 "\t" $5}' >! gene_extraction.txt

# 遺伝子の別名がある場合全て抜き出す
less tmp1 |grep ";gene_synonym=" | awk 'BEGIN{FS="\t"} {print substr($9, index($9,";gene=")+6, index($9, ";gene_biotype=")-index($9,";gene=")-6) "," substr($9, index($9, ";gene_synonym=") +14)}' >! tmp2
less tmp2 | sed 's/;.*//' > tmp3
less tmp3 | sort | uniq > synonym

# 別名を含めると同じ名前の遺伝子が2つ以上あるものがある。なのでそういう遺伝子についてデータベースの遺伝子の詳細をみてどの遺伝子なのかを一つ一つ調査し、該当しない方をファイルから削除するために以下の操作を行う。面倒臭いので元々置いてあるファイルを使うことを推奨する。 
#synonymの1997行目から"C2orf27B"を削除(パラログであり、別名ではないため)maru
#4782行目から"DUSP27"を削除maru
#6571行目から"GGTA1P"を削除maru
#22700行目から"LINC00846"を削除maru
#10221行目から"LOR"を削除maru
#12500行目から"MPP6"を削除maru
#16984, 16985行目を削除(RNA18SN4, RNA18SN5が見分けられないため)
#16986, 16987行目を削除(RNA28SN4,RNA28SN5が見分けられないため)
#20194行目のST2を削除
#22328行目のST2を削除
#2587, 20186, 22722, 23482行目のTRAPを削除
#1657行目からDEC1を削除
#6343行目からADCを削除
#10232行目からAGPAT9を削除
#1448, 20970行目からAIM1を削除
#2847行目からAPITD1を削除
#1486行目からB3GNT1を削除
#3042行目からC10orf2を削除
#5322行目からC2orf47を削除
#6078行目からC7orf55を削除
#350, 353行目からCD97を削除(Webのデータベースになく判断ができないため)
#2986行目からCGBを削除
#15088行目からCSRP2BPを削除
#3760, 3761行目からCT45A4を削除
#5081行目からEFTUD1を削除
#24788行目からFAM21Aを削除
#6429行目からGARSを削除
#10972, 12735行目からGIFを削除
#7229行目からHIST1H2BC, HIST1H2BE, HIST1H2BF, HIST1H2BG, HIST1H2BIを削除
#12801行目からHN1を削除
#3598行目からHN1Lを削除
#9700行目からLIMS3Lを削除
#21168行目からLINC00086を削除
#21167, 21168行目からLINC00087を削除
#20678行目からMARSを削除
#8747行目からMUM1を削除
#7964行目からNAT6を削除
#13542, 13543行目からNOTCH2NLを削除
#15499, 17904行目からNOVを削除
#15530, 15534行目からPMS2L2を削除
#5282行目からQARSを削除
#16987行目からRNA45S5を削除
#20046行目からSARSを削除
#20343行目からSEPT2を削除
#4481行目からSHFM1を削除
#20909行目からSLC35E2を削除
#21199行目からSMA4を削除
#1657行目からSTRA13を削除
#23506, 25474行目からTRAPPC2P1を削除
#23800行目からTRIM49D2Pを削除

rm tmp1 tmp2 tmp3


less ../GCF_000001405.39/annotations/GCF_000001405.39_genomic_refseq.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";product=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }' | sed 's/;.*//' | uniq > gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_genomic_refseq.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";inference=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_genomic_refseq.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";partial=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt
less ../GCF_000001405.39/annotations/GCF_000001405.39_genomic_refseq.gff | awk 'BEGIN{FS = "\t"} $3 == "CDS" {print substr($9, index($9,";gene=")+6, index($9, ";pseudo=")-index($9,";gene=")-6), "\t", substr($9, index($9, ";protein_id=")+12) }'| sed 's/;.*//' | uniq >> gene_proteinid.txt
less gene_proteinid.txt | awk '{if(NF > 1) {print $0}}' | sort | uniq > gene_proteinid2.txt
rm gene_proteinid
mv gene_proteinid2 gene_proteinid
