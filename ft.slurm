#!/bin/bash
#SBATCH --account=project_2000582
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=16G

set -o errexit
source ~/.bashrc
conda activate nlp

export UD="/projappl/project_2000582/corpora/ud"
export savedir="/scratch/project_2000582/naacl"

python3 runner.py --train "$UD/$1/$1-ud-train.conllu" --val "$UD/$1/$1-ud-dev.conllu" \
                  --config "./configs/gpu.jsonnet" --save "$savedir/$1.$2.$3.$4" --model "bert" \
                  --lca "experiments/stats/$1.$2.$3.$4" --kill $2 --lca_mode $3 --seed $4 --epochs 20
rm -rf "$savedir/$1.$2.$3.$4"
