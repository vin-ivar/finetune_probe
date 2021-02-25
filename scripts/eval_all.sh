#!/bin/bash
tb_long_list=(PADT GSD PDT EWT TDT GSD GSD HDTB GSD ISDT GSD GSD
         PDB Bosque SynTagRus AnCora Talbanken IMST)
tb_short_list=(padt gsd pdt ewt tdt gsd gsd hdtb gsd isdt gsd gsd
         pdb bosque syntagrus ancora talbanken imst)
lang_long_list=(Arabic Chinese Czech English Finnish French
         German Hindi Indonesian Italian Japanese Korean
         Polish Portuguese Russian Spanish Swedish Turkish)
lang_short_list=(ar zh cs en fi fr de hi id it ja ko pl pt ru es sv tr)
paths=()

UD="/cluster/projects/nn9447k/artku750/data/ud-treebanks-v2.4"
MODEL="./experiments/models/ud.all_pud.${1}.20.underparam"
for idx in "${!lang_long_list[@]}"; do
    tb_long=${tb_long_list[${idx}]}
    tb_short=${tb_short_list[${idx}]}
    lang_long=${lang_long_list[${idx}]}
    lang_short=${lang_short_list[${idx}]} 
    paths+="${UD}/UD_${lang_long}-${tb_long}/${lang_short}_${tb_short}-ud-test.conllu"
    paths+=" "
done

sbatch -J $1 -o ./eval/ud/$1 -e ./eval/ud/$1 eval.slurm $MODEL $paths
