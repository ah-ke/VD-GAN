#!/bin/bash
#SBATCH --account=caibo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=360:00:00
#SBATCH --output=filename.out

#module load nvidia/cuda/10.1

cd /project/keaihua/SwinUnet/model2

nvidia-smi
python train.py --continue_train Fasle --name cityscapes --dataset_mode cityscapes --dataroot /project/keaihua/dataset/cityscapes --batchSize 24 --gpu_ids 0 --save_epoch_freq 1