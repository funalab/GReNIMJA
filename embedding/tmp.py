import pickle

#with open('6_dna_dict.pickle', 'rb') as f:

with open('tmp.pickle', 'rb') as f:
    dna_dict = pickle.load(f)
print(len(dna_dict))
