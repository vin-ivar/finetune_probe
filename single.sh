#!/bin/bash
sbatch -J $5.$2.$3.$4 -e experiments/logs/naacl/$5.$2.$3.$4.log -o experiments/logs/naacl/$5.$2.$3.$4.log ft.slurm $1 $2 $3 $4 $5
