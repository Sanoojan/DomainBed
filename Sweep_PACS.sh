#!/bin/bash
#SBATCH --job-name=PACS_DeitTiny
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --cpus-per-task=4
#SBATCH --partition=multigpu

 python -m domainbed.scripts.sweep delete_incomplete \
    --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
    --output_dir=./domainbed/outputs/DeitTiny_check \
    --command_launcher multi_gpu \
    --algorithms DeitTiny \
    --single_test_envs \
    --datasets PACS \
    --n_hparams 1  \
    --n_trials 3 \
    --skip_confirmation \
    --hparams """{\"batch_size\":32}""" > sanoojan2.out

 python -m domainbed.scripts.sweep launch \
    --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
    --output_dir=./domainbed/outputs/DeitTiny_check \
    --command_launcher multi_gpu \
    --algorithms DeitTiny \
    --single_test_envs \
    --datasets PACS \
    --n_hparams 1  \
    --n_trials 3 \
    --skip_confirmation \
    --hparams """{\"batch_size\":32}""" > sanoojan2.out
