CUDA_VISIBLE_DEVICES=3 python main.py --eval --logdir /home/mrp_929/projects/SeDID/logs/DDPM_CIFAR10_EPS \
--dataset CIFAR10 --img_size 32 &
CUDA_VISIBLE_DEVICES=2 python main.py --eval --logdir /home/mrp_929/projects/SeDID/logs/DDPM_TINY-IN_EPS \
--dataset TINY-IN --img_size 32 