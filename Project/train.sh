#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=3g.90gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-3:00:00

unset HF_HUB_CACHE

curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"

# python train.py --p_uncond 0.1
python train.py --p_uncond 0.05
# python train.py --p_uncond 0