#!/bin/bash
#
#SBATCH --job-name=finetune             # Descriptive name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Single process
#SBATCH --cpus-per-task=8               # 8 physical cores (1 GPU)
#SBATCH --mem=96G                       # 8 cores × 12 GB per core
#SBATCH --time=24:00:00                 # Walltime (adjust if needed)
#SBATCH --partition=gpu                 # Your GPU partition
#SBATCH --gres=gpu:rtx8000:3
#SBATCH --output=logs/finetune_%j.out   # STDOUT log
#SBATCH --error=logs/finetune_%j.err    # STDERR log



# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clm_finetune
# pip install --upgrade accelerate

# Move to your project directory
cd /home/ctisseau/llm_lcd_finetune

# Run finetune (all params are inside finetune.py)
echo "Job started on $(hostname) at $(date)"
python finetune.py
echo "Job finished at $(date)"
