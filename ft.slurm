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

if [[ $1 == "pud" ]]; then
	export UD="/cluster/projects/nn9447k/corpora/pud"
	export train="test"
	export dev="test"
elif [[ $1 == "ud" ]]; then
	export UD="/cluster/projects/nn9447k/corpora/ud"
	export train="train"
	export dev="dev"
fi

python3 runner.py --train "$UD/$2/$2-ud-$train.conllu" --val "$UD/$2/$2-ud-$dev.conllu" \
                  --config "./configs/gpu.jsonnet" --save "experiments/models/$1.$2.$3" --model "bert" \
                  --param_save "experiments/models/$1.$2.$3.lca" --param_freeze $3
rm -rf ./experiments/models/$1.$2.$3
