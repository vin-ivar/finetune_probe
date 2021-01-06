#!/bin/bash
#SBATCH --account=project_2000582
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

set -o errexit
source ~/.bashrc
conda activate nlp

export UD="/projappl/project_2000582/corpora/ud"

for seed in 42 420 4200; do
	python3 runner.py --train "$UD/$1/$1-ud-train.conllu" --val "$UD/$1/$1-ud-dev.conllu" \
			  --config "./configs/gpu.jsonnet" --save "$LOCAL_SCRATCH/$1.$2.$3" --model "bert" \
			  --lca "experiments/stats/$1.$2.$3" --kill $2 --lca_mode $3 --seed $seed --epochs 20
	rm -rf $LOCAL_SCRATCH/$1.$2.$3
done
