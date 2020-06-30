#!/bin/bash
tb_short=(ar_padt eu_bdt zh_gsd en_ewt 
          fi_tdt he_htb hi_hdtb it_isdt
          ja_gsd ko_gsd ru_syntagrus
          sv_talbanken tr_imst)
model=bert

for trg in ${tb_short[@]}; do
    sbatch -J dep.${trg}.${model} -e experiments/logs/${model}/ft_0/${trg} -o experiments/logs/${model}/ft_0/${trg} ft.slurm ${trg} ${model}
done

