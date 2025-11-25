#!/bin/bash
#SBATCH --job-name=gpu_testing
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-3:00:00

unset HF_HUB_CACHE

curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"

cfg_scales=(1.0 1.4 1.8)
num_cands=(4)
# checkpoints=(0 2321 4642 6963)
checkpoints=(4642 6963)
num_unmask_steps=(128)

# python test.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0
# python test.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.1
# python test.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.1 --cfg_scale 1.5
# python test.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.05 --cfg_scale 1.5
# python test.py --checkpoint_num 0

# python test_instruct.py --checkpoint_num 0 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.0
# python test_instruct.py --checkpoint_num 2321 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.0
# python test_instruct.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.0
# python test_instruct.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.2
# python test_instruct.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.4

# python test_instruct_svdd.py --checkpoint_num 4642 --num_unmask_steps 128 --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale 1.0 --num_cands 4

for ckpt_num in "${checkpoints[@]}"; do
	for num_unmask in "${num_unmask_steps[@]}"; do
		for cfg_scale in "${cfg_scales[@]}"; do
			for num_cand in "${num_cands[@]}"; do
				curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
				echo "Testing SVDD with checkpoint $ckpt_num, $num_unmask unmask steps, CFG scale $cfg_scale, and $num_cand candidates"
				python test_instruct_svdd.py --checkpoint_num $ckpt_num --num_unmask_steps $num_unmask --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale $cfg_scale --num_candidates $num_cand
			done
			curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
			echo "Testing checkpoint $ckpt_num with $num_unmask unmask steps and CFG scale $cfg_scale"
			python test_instruct.py --checkpoint_num $ckpt_num --num_unmask_steps $num_unmask --batch_size 8 --lr 2e-4 --epochs 3 --p_uncond 0.0 --cfg_scale $cfg_scale
		done
	done
done