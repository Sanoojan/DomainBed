#!/bin/bash
#SBATCH --job-name=Exp_PACS
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=4
#SBATCH --partition=default-long


for lambda1 in 0.5 0.2 0.1 0.05
do
    for lambda2 in 0.5 0.2 0.1 0.05
    do
        for lambda3 in 0.5 0.2 0.1 0.05
        do
        python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ \
        --output_dir=/domainbed/Random_block_outputs_distillation/PACS/New_idea/DeiT_Small_H_New/Final_Exp \
          --command_launcher multi_gpu  --algorithms DeiT_Small_Distilled_Soft_RB_H  --single_test_envs  --datasets PACS  --n_hparams 1 --n_trials 1 \
           --hparams """{\"batch_size\":39,\"lr\":2.7028930742148706e-05,\"resnet_dropout\":0.5,\"weight_decay\":0.00044832883881609976,\"RB_loss_weight1\":$lambda1,\"RB_loss_weight2\":$lambda2,\"RB_loss_weight3\":$lambda3}"""
        done
    done 

done > maryam.out
