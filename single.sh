#!/bin/bash
sbatch -J $2.$3 -e experiments/logs/lca/$2.$3.log -o experiments/logs/lca/$2.$3.log ft.slurm $1 $2 $3
