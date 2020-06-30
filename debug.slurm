#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=16G

set -o errexit
source ~/.bashrc
module purge
conda activate debug

export UD="$HOME/corpora/ud"
python3 runner.py --train "$UD/en_ewt/en_ewt-ud-train.conllu" --val "$UD/en_ewt/en_ewt-ud-dev.conllu" \
                  --config "./configs/gpu" --save "experiments/models/en_ewt.bert" --model bert
