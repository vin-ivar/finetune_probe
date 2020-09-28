#!/bin/bash
module purge
conda activate debug

tb_short
tb_short=(ar_padt eu_bdt zh_gsd en_ewt
          fi_tdt he_htb hi_hdtb it_isdt
          ja_gsd ko_gsd ru_syntagrus
          sv_talbanken tr_imst)
UD="/cluster/projects/nn9447k/corpora/ud"

for lang in $tb_short[@]; do
	printf "%s" $lang
	python3 finetune.py --test $UD/$lang/$lang-ud-test.conllu --path $1
done