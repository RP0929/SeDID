CUDA_VISIBLE_DEVICES=2 python -m main --train \
--dataset CIFAR100 --logdir ./logs/DDPM_CIFAR100_EPS & 
CUDA_VISIBLE_DEVICES=3 python -m main --train \
--dataset TINY-IN --logdir ./logs/DDPM_TINY-IN_EPS &
CUDA_VISIBLE_DEVICES=5 python -m main --train \
--dataset CIFAR10 --logdir ./logs/DDPM_CIFAR10_EPS
wait