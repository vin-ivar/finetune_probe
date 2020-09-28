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

tb_short=(ar_pud zh_pud cs_pud en_pud 
          fi_pud fr_pud de_pud hi_pud 
          id_pud it_pud ja_pud ko_pud 
          pl_pud pt_pud ru_pud es_pud 
          sv_pud th_pud tr_pud)
UD="/cluster/projects/nn9447k/corpora/pud"

for lang in ${tb_short[@]}; do
    printf "%s\t" $lang
    python3 evaluate.py --config configs/gpu.jsonnet --test $UD/$lang/$lang-ud-test.conllu --path $1 --lang $lang
    printf "\n"
done
