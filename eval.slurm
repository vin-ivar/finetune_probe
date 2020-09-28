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

tb_short=(ar_padt eu_bdt zh_gsd en_ewt
          fi_tdt he_htb hi_hdtb it_isdt
          ja_gsd ko_gsd ru_syntagrus
          sv_talbanken tr_imst)
UD="/cluster/projects/nn9447k/corpora/ud"

for lang in ${tb_short[@]}; do
    printf "%s\t" $lang
    python3 evaluate.py --config configs/gpu.jsonnet --test $UD/$lang/$lang-ud-test.conllu --path $1
    printf "\n"
done
