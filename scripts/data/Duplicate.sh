#!/bin/zsh
set -ex

file=$1

less ./data/$1 | sort -k 2 -k 3 | uniq -f 2 | grep -v "tf" > ./data/tmp
rm ./data/$1
mv ./data/tmp ./data/$1
