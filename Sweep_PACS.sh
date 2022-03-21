#!/bin/bash
#SBATCH --job-name=deitfeatvit
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=default-long

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/outputs/DeitSmallDtest/Wd \
      --command_launcher multi_gpu \
      --algorithms DeitSmallDtest \
      --single_test_envs \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation \
      --hparams """{\"batch_size\":32,\"Wd\":0.5}""" > outs/DeitSmallDtest.out
   done