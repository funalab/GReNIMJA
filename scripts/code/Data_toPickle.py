import pandas as pd
import pickle

ans_filename = '../data/ans_Data_seq'
miss_filename = '../data/D_miss_Data_seq'

df = pd.read_csv(ans_filename, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))
with open('../data/ans.pickle', 'wb') as f:
    pickle.dump(df, f)

df = pd.read_csv(miss_filename, sep='\t', names=('tf', 'tf_seq', 'target', 'ta_seq', 'label', 'confidence'))
with open('../data/D_miss.pickle', 'wb') as f:
    pickle.dump(df, f)
