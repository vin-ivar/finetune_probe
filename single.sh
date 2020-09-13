#!/bin/bash
sbatch -J u.$2.$3 -e experiments/logs/underparam/$2.$3.log -o experiments/logs/underparam/$2.$3.log ft.slurm $1 $2 $3 $4
