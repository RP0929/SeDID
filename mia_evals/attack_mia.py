import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
from torch.nn import functional as F
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
import torchvision
from torch.utils.data import DataLoader, Dataset,Subset
from PIL import Image
import glob

def asr(
    preds,
    target,
):

    with torch.no_grad():
        desc_score_indices = torch.argsort(preds, descending=True)
        preds = preds[desc_score_indices]
        target = target[desc_score_indices]
        distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
        tps = F.pad(torch.cumsum(target, dim=0)[threshold_idxs], [1, 0], value=0)
        tns = F.pad((1 - target).sum() - torch.cumsum((1 - target), dim=0)[threshold_idxs], [1, 0], value=(1 - target).sum())
        return (tps + tns) / target.size(0), preds[threshold_idxs]


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

# 定义CustomImageDataset
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_generated_data_loader(checkpoint_path, output_folder, transform, batch_size=128):

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 准备图像数据
    generated_images = checkpoint['samples'][:50000]
    generated_images = np.transpose(generated_images, (0, 2, 3, 1))

    os.makedirs(output_folder, exist_ok=True)

    for img_idx, img in enumerate(generated_images):
            img_path = os.path.join(output_folder, f'image{img_idx}.png')
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(img_path)

    # 收集图像路径
    image_paths = glob.glob(os.path.join(output_folder, "*.png"))

    # 创建标签列表
    labels = [0] * len(image_paths)  # 所有生成的图像都标记为类别0

    # 实例化CustomImageDataset
    generated_dataset = CustomImageDataset(image_paths, labels, transform=transform)

    # 创建并返回DataLoader
    return DataLoader(generated_dataset, batch_size=batch_size, shuffle=True)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "iter": components.IterDDIMAttacker,
    "groundtruth": components.GroundtruthAttacker,
    # "naive_pgd": components.NaivePGDAttacker,
    # "groundtruth_pgd": components.GroundtruthPGDAttacker,
    "groundtruth_abs": components.GroundtruthAbsAttacker,
    "SecMI": components.SecMIAttacker,
    "PIA": components.PIA,
    "naive": components.NaiveAttacker,
    "PIAN": components.PIAN,
}


DEVICE = 'cuda'


@torch.no_grad()
def main(
        checkpoint="/home/mrp_929/projects/SeDID/logs/DDPM_STL10_U_EPS/ckpt-step1200000.pt",
        ckpt_path = "/home/mrp_929/projects/SeDID/logs/DDPM_STL10_U_EPS/ckpt-step1200000_samples.pt",
        output_folder_path = '/home/mrp_929/projects/SeDID/generated_images/DDPM_SVNH',
        dataset="STL10_U",
        attacker_name="naive",
        name="STL10_U_pretrain",
        attack_num=5, interval=165,
        save_logger=None,
        seed=0,
        batch_size=128
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
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
    ])
    if dataset == 'CIFAR10':
        train_dataset = CIFAR10(root='/home/mrp_929/projects/data', train=True,
                              transform=transforms)
    if dataset == 'TINY-IN':
        train_dataset = torchvision.datasets.ImageFolder(root='/home/mrp_929/projects/data/tiny-imagenet-200/train',
                                                       transform=transforms)
    if dataset == "CELEBA":
        transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
    ])
        train_dataset = CelebA(root='/home/mrp_929/projects/data', split='train',
                             transform=transforms, download=False)
    if dataset == 'CIFAR100':
        train_dataset = CIFAR100(root='/home/mrp_929/projects/SeDID/datasets', train=True,
                              transform=transforms)
    if dataset == "SVHN":
        train_dataset = torchvision.datasets.SVHN(root='/home/mrp_929/projects/SeDID/datasets', download=True, transform=transforms)
    if dataset == "STL10_U":
        train_dataset  = torchvision.datasets.STL10(root='/home/mrp_929/projects/SeDID/datasets', split='unlabeled',download=True, transform=transforms)
    # 选择前50000个数据
    train_dataset = Subset(train_dataset, range(50000))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = prepare_generated_data_loader(ckpt_path, output_folder_path, transforms, batch_size=128)

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
            torch.save({'member': members[0], 'nonmember': nonmembers[0]}, f'/home/mrp_929/projects/SeDID/statistics/{name}_{attacker_name}_{interval}.pt')

    member = members[0]
    nonmember = nonmembers[0]

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
                "tpr_fpr_01": tpr_fpr_01, "top1_01": cp_tpr_fpr_01[0], "top2_01": cp_tpr_fpr_01[1], "top3_01": cp_tpr_fpr_01[2]}, f'{name}_{attacker_name}_{interval}_result.pt')
    print("roc:",auroc,"tpf_fpr_1:",tpr_fpr_1,"tpr_fpr_01:",tpr_fpr_01)

if __name__ == '__main__':
    fire.Fire(main)
