#!/bin/zsh

# sensitivity analysis
# encoding
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50 --embedding embedding
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 20 --dna_emb_dim 4 --embedding one_hot
# mer
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 1 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 3 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 8 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 10 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50

# amino_mer
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 2 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 6 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 8 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 10 --amino_emb_dim 100 --dna_emb_dim 50

# amino_emb_dim
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 10 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 50 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 250 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 500 --dna_emb_dim 50

# dna_emb_dim
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 10
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 50
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 100
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 250
CUDA_VISIBLE_DEVICES=1 python src/main_sensitivity_analysis.py --epoch_num 3 --mer 5 --amino_mer 4 --amino_emb_dim 100 --dna_emb_dim 500
