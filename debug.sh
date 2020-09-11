#!/usr/bin/env bash
python3 runner.py --train "$UD/$1/$1-ud-train.conllu" --val "$UD/$1/$1-ud-dev.conllu" \
                  --config "./configs/gpu" --save "experiments/models/$1.$2" --model $2