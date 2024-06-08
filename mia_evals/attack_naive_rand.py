import torch
import sys
import os
import numpy as np
import random

from .dataset_utils import load_member_data
from absl import flags
from model import UNet

from rich.progress import track
import logging
from continuous_ddim import attack_concrete_middle, attack_forward_backward, attack_concrete, attack_concrete_halv, attack_concrete_groundtruth, attack_concrete_groundtruth_backward_rand, attack_concrete_groundtruth_backward, attack_concrete_groundtruth_backward_fgsm
from continuous_ddim import attack_concrete_groundtruth_eps, attack_concrete_groundtruth_eps_fgsm, attack_concrete_groundtruth_eps_pgd, attack_concrete_groundtruth_eps_rand
from continuous_ddim import attack_concrete_groundtruth_eps_naive_rand_pgd
from functools import partial
import warnings
import fire
from rich.logging import RichHandler
import torch

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler())
DEVICE = 'cuda'


def get_FLAGS(flag_path):
    FLAGS = flags.FLAGS
    flags.DEFINE_bool('train', False, help='train from scratch')
    flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
    # UNet
    flags.DEFINE_integer('ch', 128, help='base channel of UNet')
    flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
    flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
    flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
    flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
    # Gaussian Diffusion
    flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
    flags.DEFINE_float('beta_T', 0.02, help='end beta value')
    flags.DEFINE_integer('T', 1000, help='total diffusion steps')
    flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
    flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
    # Training
    flags.DEFINE_float('lr', 2e-4, help='target learning rate')
    flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
    flags.DEFINE_integer('total_steps', 800000, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', False, help='multi gpu training')
    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 80000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def get_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


def fix_seed():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eps_getter(model, x0, xt, cum_alpha_sqrt, concrete_t):
    concrete_t = torch.ones([x0.shape[0]], device=xt.device).long() * concrete_t
    return model(xt, t=concrete_t)


def white_attack(model, FLAGS, batch_size):
    # from training data
    start_point, end_point, total_time = FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
    delta_t = 10
    n_timesteps = 30

    logger.info("loading dataset...")
    _, _, train_dataloader, test_dataloader = load_member_data(dataset_name='cifar10', batch_size=batch_size,
                                                               shuffle=False, randaugment=False)
    logger.info("attacking...")
    members, nonmembers = [], []
    for member, nonmember in track(zip(train_dataloader, test_dataloader), total=len(test_dataloader)):
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        intermediate_denoise_member, intermediate_reverse_member = attack_concrete_groundtruth_eps_rand(
            member, partial(eps_getter, model),
            delta_t, start_point, end_point, total_time, n_timesteps, None, None
        )
        intermediate_denoise_nonmember, intermediate_reverse_nonmember = attack_concrete_groundtruth_eps_rand(
            nonmember, partial(eps_getter, model),
            delta_t, start_point, end_point, total_time, n_timesteps, None, None
        )

        members.append(((intermediate_reverse_member - intermediate_denoise_member)**2).flatten(2).sum(dim=-1))
        nonmembers.append(((intermediate_reverse_nonmember - intermediate_denoise_nonmember)**2).flatten(2).sum(dim=-1))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

        torch.save({'member': members[0], 'nonmember': nonmembers[0]},
                   f'/home/kongfei/workspace/DiffusionMIA/results/statistics/attack_concrete_groundtruth_eps_rand.pt')


fix_seed()
logger.info("loading params...")
ckpt = os.path.join('/home/kongfei/models_state_dict/mia_shixiong/ckpt-step800000.pt')
flag_path = os.path.join("/home/kongfei/models_state_dict/mia_shixiong/flagfile.txt")
FLAGS = get_FLAGS(flag_path)
FLAGS(sys.argv)
FLAGS.T = 1000
logger.info("loading checkpoint...")
model = get_model(ckpt, FLAGS, WA=True).to(DEVICE)
white_attack(model, FLAGS, 64)
model.to(DEVICE)
