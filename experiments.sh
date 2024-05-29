#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --job-name=Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_ExpThree_QM9_%A.out
#SBATCH --export=WANDB_API_KEY

module purge
module load 2022
module load Anaconda3/2022.05

cd ./src/ponita/

source ~/.bashrc

conda activate empsn_ponita

#experiment 1:
#srun python3 -u main_qm9.py --num_workers 36 --num_ori 0 --simplicial

#experiment 2:
#srun python3 -u main_qm9.py --num_workers 36 --num_ori 0 --simplicial --preserve_edges

#experiment 3:
srun python3 -u main_qm9_debug.py --num_workers 36 --num_ori 0 --simplicial --initial_edges