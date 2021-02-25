#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=10G
#SBATCh --priority=TOP

set -o errexit
source ~/.bashrc
module purge
conda activate debug

export ROOT="./corpora/pud"

python3 runner.py --train "$ROOT/train.conllu" --val "$ROOT/val.conllu" \
                  --config "./configs/gpu.jsonnet" --model "bert" \
                  --save "./models/$1.$2.underparam" \
                  --freeze $1 --epochs $2
