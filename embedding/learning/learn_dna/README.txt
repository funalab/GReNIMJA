# dna2vec

# dna2vec用の仮想環境構築
python -m venv dna2vec_1125 (2021/11/25に作成)
source dna2vac_1125/bin/activate

# まずはcloneしてくる
git clone https://github.com/pnpnpn/dna2vec

# pipをupdate
/Users/okubo/m1okubo/embedding/learning/learn_dna/dna2vec_1125/bin/python -m pip install --upgrade pip

# 以下を実行するとエラーが大量に出た
pip3 install -r requirements.txt
# 原因はscipy==0.19.0がpython2.7およびpython3-6でしかインストールできないのに、python3.7の環境でやっていたから。
#エラー文は以下の通り
Traceback (most recent call last):
File "/Users/okubo/m1okubo/embedding/dna2vec/dna2vac_1125/lib/python3.7/site-packages/gensim/models/ldamodel.py", line 50, in <module>
from  scipy.maxentropy import logsumexp
ModuleNotFoundError: No module named 'scipy.maxentropy'

# scipy.maxentropyがないと言われているため、https://hivecolor.com/id/62を参考にして
# /Users/okubo/m1okubo/embedding/learning/learn_dna/dna2vac_1125/lib/python3.7/site-packages/gensim/matutils.py
# と
# /Users/okubo/m1okubo/embedding/learning/learn_dna/dna2vac_1125/lib/python3.7/site-packages/gensim/models/ldamodel.py
# を編集したら上のエラーは解決した。


# その後
assert(word2vec.FAST_VERSION >= 0)
# でエラーとなって止まってしまうようになった。word2vec.FAST_VERSIONは計算が最適化されてるかを表す指標で、-1だと止まってしまう。
# https://github.com/RaRe-Technologies/gensim/issues/2794より、どうやらgensim==3.8.1はword2vec.FAST_VERSION == 0 みたいなので
pip install gensim==3.8.1
# を実行したところ、プログラムが動いた！！


# v103では
# python3.9でtime.clock()が使えなくなっていたので、全てtime.time()に変更した。



Training dna2vec embeddings
---

1. Download `hg38` from <http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz>.
    This will take a while as it's 938MB.
2. Untar with `tar -zxvf hg38.chromFa.tar.gz`. You should see FASTA files for
   chromosome 1 to 22: `chr1.fa`, `chr2.fa`, ..., `chr22.fa`.
3. Move the 22 FASTA files to folder `inputs/hg38/`
        
# dna2vecディレクトリで以下を実行する
python3 ./scripts/train_dna2vec.py -c ../../5mer50dim.yml
