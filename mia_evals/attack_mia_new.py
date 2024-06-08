import numpy as np
import torch
from rich.progress import track
import fire
import logging
from rich.logging import RichHandler
from pytorch_lightning import seed_everything
import components
from typing import Type, Dict
from itertools import chain
from model import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC


def get_FLAGS():

    def FLAGS(x): return x
    FLAGS.T = 1000
    FLAGS.ch = 128
    FLAGS.ch_mult = [1, 2, 2, 2]
    FLAGS.attn = [1]
    FLAGS.num_res_blocks = 2
    FLAGS.dropout = 0.1
    FLAGS.beta_1 = 0.0001
    FLAGS.beta_T = 0.02

    return FLAGS


def get_model(ckpt, WA=True):
    FLAGS = get_FLAGS()
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


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "iter": components.IterDDIMAttacker,
    "groundtruth": components.GroundtruthAttacker,
    "naive": components.NaiveAttacker,
    # "naive_pgd": components.NaivePGDAttacker,
    # "groundtruth_pgd": components.GroundtruthPGDAttacker,
    "groundtruth_abs": components.GroundtruthAbsAttacker,
}


DEVICE = 'cuda'


@torch.no_grad()
def main(
        checkpoint="/home/mrp_929/projects/SeDID/logs/DDPM_CIFAR10_EPS/ckpt.pt",
        dataset="cifar10",
        attacker_name="naive",
        name="cifar10_pretrain",
        attack_num=10, interval=1,
        save_logger=None,
        seed=0,
):
    seed_everything(seed)
    # settings = read_setting("mia_trial_setting", "image_diffusion", name, "postgresql+psycopg2://kongfei:262300aa@43.142.23.186:5432/Experiment")

    FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.disabled = True if save_logger else False
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model = get_model(checkpoint, WA = True).to(DEVICE)
    model.eval()

    logger.info("loading dataset...")
    if dataset == 'cifar10':
        _, _, train_loader, test_loader = load_member_data(dataset_name='cifar10', batch_size=64,
                                                           shuffle=False, randaugment=False)
    if dataset == 'TINY-IN':
        _, _, train_loader, test_loader = load_member_data(dataset_name='TINY-IN', batch_size=64,
                                                           shuffle=False, randaugment=False)

    attacker = attackers[attacker_name](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), interval, attack_num, EpsGetter(model), lambda x: x * 2 - 1)

    logger.info("attack start...")
    members, nonmembers = [], []

    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        intermediate_reverse_member, intermediate_denoise_member = attacker(member)
        intermediate_reverse_nonmember, intermediate_denoise_nonmember = attacker(nonmember)

        members.append(((intermediate_reverse_member - intermediate_denoise_member)**2).flatten(2).sum(dim=-1))
        nonmembers.append(((intermediate_reverse_nonmember - intermediate_denoise_nonmember)**2).flatten(2).sum(dim=-1))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

        if save_logger is not None:
            save_logger({
                'member': members[0], 'nonmember': nonmembers[0]})
        else:
            torch.save({'member': members[0], 'nonmember': nonmembers[0]}, f'/home/mrp_929/projects/SeDID/logs/DDPM_CIFAR10_EPS/statistics/{name}_{attacker_name}_{interval}.pt')

    auroc = [BinaryAUROC().cuda()(torch.cat([member[i] / max([member[i].max().item(), nonmember[i].max().item()]), nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()).item() for i in range(member.shape[0])]
    tpr_fpr = [BinaryROC().cuda()(torch.cat([1 - nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()]), 1 - member[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()) for i in range(member.shape[0])]
    tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    tpr_fpr_01 = [i[1][(i[0] < 0.001).sum() - 1].item() for i in tpr_fpr]
    cp_auroc = auroc[:]
    cp_auroc.sort(reverse=True)
    cp_tpr_fpr_1 = tpr_fpr_1[:]
    cp_tpr_fpr_1.sort(reverse=True)
    cp_tpr_fpr_01 = tpr_fpr_01[:]
    cp_tpr_fpr_01.sort(reverse=True)

    torch.save({"roc": auroc, 'top_1': cp_auroc[0], 'top_2': cp_auroc[1], 'top_3': cp_auroc[2], 'top_4': cp_auroc[3], 'top_5': cp_auroc[4],
                "tpr_fpr_1": tpr_fpr_1, "1_top1": cp_tpr_fpr_1[0], "1_top2": cp_tpr_fpr_1[1], "1_top3": cp_tpr_fpr_1[2],
                "tpr_fpr_01": tpr_fpr_01, "top1_01": cp_tpr_fpr_01[0], "top2_01": cp_tpr_fpr_01[1], "top3_01": cp_tpr_fpr_01[2]}, 'result.pt')


if __name__ == '__main__':
    fire.Fire(main)
