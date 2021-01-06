#!/bin/bash
mkdir -p experiments/logs/$4
mkdir -p experiments/stats
mkdir -p $SCRATCH/naacl

if [ `whoami` == "ravishan" ]; then
	export server="puhti"
else
	export server="saga"
fi

sbatch -J $1.$2.$3 -e experiments/logs/$4/$1.$2.$3.log -o experiments/logs/$4/$1.$2.$3.log $server.slurm $1 $2 $3
