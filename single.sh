#!/bin/bash
mkdir -p experiments/logs
mkdir -p experiments/stats
mkdir -p $SCRATCH/naacl

sbatch -J $1.$2.$3 -e experiments/logs/$1.$2.$3.log -o experiments/logs/$1.$2.$3.log ft.slurm $1 $2 $3
