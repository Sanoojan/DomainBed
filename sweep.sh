# CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/JustTransformer_2 --command_launcher gpu_2 --algorithms JustTransformer --single_test_envs --datasets PACS --n_hparams 1  --n_trials 3 --steps 10000 --hparams """{\"batch_size\":32,\"lr\":0.001}"""
# CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/ERM_Res50_pacs_Full_sweep --command_launcher multi_gpu_2_3 --algorithms ERM --single_test_envs --datasets PACS --n_hparams 5  --n_trials 3
for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep $command\
        --data_dir=/share/data/drive_2/DG/data \
        --output_dir=./domainbed/outputs/PACS/CrossImageVIT_self_SepCE_SINF_sim \
        --command_launcher multi_gpu_0_1\
        --algorithms CrossImageVIT_self_SepCE_SINF_sim \
        --single_test_envs \
        --datasets PACS \
        --n_hparams 1  \
        --n_trials 3 \
        --hparams """{\"batch_size\":32}"""
done

# CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.sweep launch \
#     --data_dir=/home/computervision1/DG_new_idea/domainbed/data \
#     --output_dir=./domainbed/outputs/CVTSmall \
#     --command_launcher multi_gpu_0_1 \
#     --algorithms CVTSmall \
#     --single_test_envs \
#     --datasets PACS \
#     --n_hparams 1  \
#     --n_trials 3 \
#     --hparams """{\"batch_size\":32,\"lr\":0.00005}"""