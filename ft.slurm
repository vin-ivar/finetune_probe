#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=10G

set -o errexit
source ~/.bashrc
module purge
conda activate debug

export UD="$HOME/corpora/ud"
python3 runner.py --train "$UD/$1/$1-ud-train.conllu" --val "$UD/$1/$1-ud-dev.conllu" \
                  --config "./configs/gpu" --save "experiments/models/$1.$2" --model $2
