for exp in none best dense kq_embed kq_net only_kq; do
    sbatch -J ${exp} -o ./eval/${exp}.log -e ./eval/${exp}.log eval.slurm ./experiments/models/ud.all_pud.${exp}.20.underparam
done
