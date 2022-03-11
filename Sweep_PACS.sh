#!/bin/bash
#SBATCH --job-name=CrsSepCESInf
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=4
#SBATCH --partition=default-short

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/outputs/CrossImageVITSepCE_SINF\
      --command_launcher multi_gpu \
      --algorithms CrossImageVITSepCE_SINF \
      --single_test_envs \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation \
      --hparams """{\"batch_size\":32}""" > CrossImageVITSepCE_SINF_q7_mlp1536.out
   done