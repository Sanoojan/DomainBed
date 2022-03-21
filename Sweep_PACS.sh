#!/bin/bash
#SBATCH --job-name=sim0.5
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5
#SBATCH --partition=default-long

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/outputs/CrossImageVIT_self_SepCE_SINF_sim12/Ws0.5\
      --command_launcher multi_gpu \
      --algorithms CrossImageVIT_self_SepCE_SINF_sim \
      --single_test_envs \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation \
      --hparams """{\"batch_size\":32,\"Ws\":0.5}""" > outs/CrossImageVIT_self_SepCE_SINF_sim.out
   done