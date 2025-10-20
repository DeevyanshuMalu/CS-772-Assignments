#!/bin/bash
#SBATCH --job-name=gpu_testing
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-3:00:00

curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"

python test.py --checkpoint_num 1975 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3
python test.py --checkpoint_num 5925 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3
# python test.py --checkpoint_num 0