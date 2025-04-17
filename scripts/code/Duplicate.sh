less ./D_ans_Data | sort -k 2 -k 3 | uniq -f 2 | grep -v "tf" > ans_Data2
rm D_ans_Data
mv ans_Data2 D_ans_Data
