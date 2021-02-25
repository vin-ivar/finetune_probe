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

python3 evaluate.py --config configs/gpu.jsonnet --path $1 --test ${@:2}
