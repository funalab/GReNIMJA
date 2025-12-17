#!/bin/zsh

# sensitivity analysis
# mer
python src/main.py --epoch_num 10 --mer 1 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 3 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 8 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 10 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50

# amino_mer
python src/main.py --epoch_num 10 --mer 5 --amino_mer 2 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 6 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 8 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 10 --amino_emb_dim 100 --dna_emb_dim 50

# amino_emb_dim
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 10 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 50 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 250 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 500 --dna_emb_dim 50

# dna_emb_dim
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 10
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 100
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 250
python src/main.py --epoch_num 10 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 500
