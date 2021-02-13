#!/usr/bin/env bash
python runner.py \
       --train ~/Work/ud/ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.100.conllu \
       --val ~/Work/ud/ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-dev.100.conllu \
       --model "bert" --config ./configs/sa_cpu.jsonnet --save shit
