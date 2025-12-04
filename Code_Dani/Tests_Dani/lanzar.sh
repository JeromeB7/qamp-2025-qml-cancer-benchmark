#!/bin/bash
#SBATCH -t 100:00:00        # Time
#SBATCH --mem=32G         # Memory 
#SBATCH -n 1               # Number of MPI tasks 
#SBATCH -c 4               # Number of cores per task 

module load cesga/2022 miniconda3/22.11.1-1
#module load miniconda3
conda activate qamp_linux
python main.py
