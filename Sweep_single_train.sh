#!/bin/bash
#SBATCH --job-name=sngl_train
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=multigpu

for algo in ERM DeitSmall
    do
        for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command \
                    --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
                    --output_dir=./domainbed/outputs/single_train_models/$algo\
                    --command_launcher multi_gpu \
                    --algorithms $algo \
                    --one_train_rest_test \
                    --datasets PACS \
                    --n_hparams 1  \
                    --n_trials 3 \
                    --skip_confirmation \
                    --hparams """{\"batch_size\":32}""" > single_train_models.out
            done
    done
