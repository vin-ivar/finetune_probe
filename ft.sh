#!/bin/bash
tb_short=(ar_padt eu_bdt zh_gsd en_ewt 
          fi_tdt he_htb hi_hdtb it_isdt
          ja_gsd ko_gsd ru_syntagrus
          sv_talbanken tr_imst)
tb_short=(ja_gsd)
model=bert

for trg in ${tb_short[@]}; do
    sbatch -J dep.${trg}.${model} -e experiments/logs/${trg}.log -o experiments/logs/${trg}.log ft.slurm ${trg} ${model}
done

