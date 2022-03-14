#!/bin/bash
#SBATCH --job-name=CrossImageVIT
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=multigpu

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/outputs/CrossImageVIT_2\
      --command_launcher multi_gpu \
      --algorithms CrossImageVIT \
      --single_test_envs \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation \
      --hparams """{\"batch_size\":32}""" > CrossImageVIT.out
   done