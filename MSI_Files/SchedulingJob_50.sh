#!/bin/bash
#SBATCH --job-name=Rnn_50
#SBATCH --time=32:00:00
#SBATCH --ntasks=2
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marso093@umn.edu

module load python3
source activate AdvGeoProject

cd ~/MountainProject
python3 Mountain_model_training_MSI_50.py






