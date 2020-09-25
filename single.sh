#!/bin/bash
sbatch -J u.$2.$3.$4 -e experiments/logs/underparam/$2.$3.$4.log -o experiments/logs/underparam/$2.$3.$4.log ft.slurm $1 $2 $3 $4
