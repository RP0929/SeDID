import copy

import torch
import sys
import os

import SeDID_nn
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import numpy as np
import random
# import lpips
import tqdm

from sklearn import metrics
from dataset_utils import load_member_data, load_synthetic_dataset
from torchvision.utils import make_grid
from absl import flags
from diffusion import DDIMSampler
from model import UNet
import matplotlib
import math
# from measures import ssim
from dataset_utils import ReconsDataset

from torchvision.utils import save_image
import torch
import torchvision

matplotlib.rcParams['figure.dpi'] = 300
import setproctitle
setproctitle.setproctitle(f'mrp_mia_TinyIN')
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

def fix_seed():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    flags.DEFINE_string('logdir', './logs/DDPM_TINY-IN_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def sample(model, sampler, FLAGS, x_T, device='cuda'):
    model.eval()
    with torch.no_grad():
        batch_images, internal_samples = sampler(x_T)
    return batch_images, internal_samples


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


def ddim_denoise(model, FLAGS, x_T, device='cuda', eta=0, ddim_step=1000):
    sampler = DDIMSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, eta=eta, n_step=ddim_step, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)

    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    samples, internal_samples = sample(model, sampler, FLAGS, x_T, device=device)
    return samples, internal_samples


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


import glob
import random
    
@torch.no_grad()
def ddim_reverse(model, x_0, beta_1, beta_T, T, steps=None, return_intermediate=False):
    """
    Get latent features by adding noises predicted from model
    """
    x_0 = x_0.cuda()
    betas = torch.linspace(beta_1, beta_T, T).double().cuda()
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    # inverse process to reconstruct
    x_t = None
    x_t_prev = x_0
    if isinstance(steps, list):
        time_steps = steps[0]
        prev_time_steps = steps[1]
    else:
        if steps is None:
            time_steps = range(0, T)
        else:
            time_steps = list(range(0, T))[(T // steps) - 1::T // steps]
        prev_time_steps = time_steps[:1] + time_steps[:-1]
    # print(time_steps, prev_time_steps)
    intermediate_results = []
    for time_step, prev_time_step in zip(steps[0], steps[1]):
        t_prev = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * (prev_time_step)
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * (time_step)
        alphas_t_prev = extract(alphas, t=t_prev, x_shape=x_0.shape)
        alphas_t = extract(alphas, t=t, x_shape=x_0.shape)
        # model predicts epsilon
        epsilon = model(x_t_prev, t=t_prev)
        pred_x_0 = (x_t_prev - ((1 - alphas_t_prev).sqrt() * epsilon)) / (alphas_t_prev.sqrt())
        x_t = alphas_t.sqrt() * pred_x_0 + (1 - alphas_t).sqrt() * epsilon
        x_t_prev = x_t
        intermediate_results.append(x_t.detach().clone())
    if return_intermediate:
        return x_t, intermediate_results
    else:
        return x_t

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

    # 检查输出文件夹是否已存在
    
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

def calculate_auc_acc_asr_stat(member_scores, nonmember_scores):
    print(f'member score: {member_scores.mean():.4f}, nonmember score: {nonmember_scores.mean():.4f}')

    total = member_scores.size(0) + nonmember_scores.size(0)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = max(member_scores.max(), nonmember_scores.max()).item()
    print(f'Min score: {min_score}, Max score: {max_score}')

    TPR_list = []
    FPR_list = []
    best_asr = 0
    best_acc = 0

    TPRatFPR_1 = 0
    TPRatFPR_01 = 0

    for threshold in torch.arange(min_score, max_score, (max_score - min_score) / 1000):
        TP = (member_scores >= threshold).sum().item()
        TN = (nonmember_scores < threshold).sum().item()
        FP = (nonmember_scores >= threshold).sum().item()
        FN = (member_scores < threshold).sum().item()

        TPR = TP / (TP + FN) if TP + FN else 0
        FPR = FP / (FP + TN) if FP + TN else 0
        ACC = (TP + TN) / total
        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR
        if ACC > best_acc:
            best_acc = ACC

        if abs(FPR - 0.01) < abs(TPRatFPR_1 - TPR):
            TPRatFPR_1 = TPR
        if abs(FPR - 0.001) < abs(TPRatFPR_01 - TPR):
            TPRatFPR_01 = TPR

        TPR_list.append(TPR)
        FPR_list.append(FPR)

    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc:.4f}, ACC: {best_acc:.4f}, ASR: {best_asr:.4f}, TPR@FPR=1%: {TPRatFPR_1:.4f}, TPR@FPR=0.1%: {TPRatFPR_01:.4f}')

def secmi_stat(model, FLAGS, batch_size=128, score_type='mse'):
    timestep = 165
    print(timestep)
    target_steps = list(range(0, 1000, timestep))[1:]
    ddim_forward_step = [target_steps, [0] + target_steps[:-1]]
    ddim_gen_step = [list(reversed(target_steps)), list(reversed(target_steps[:-1])) + [0]]

    dataset = torchvision.datasets.ImageFolder(root='/home/mrp_929/projects/data/tiny-imagenet-200/train',
                                                       transform=transforms)
    # 选择前50000个数据
    dataset = Subset(dataset, range(50000))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def norm(x):
        return (x + 1) / 2

    def get_recon_score(data_loader, ddim_reverse_step, ddim_denoise_step, score_type='ssim'):
        scores_list = []
        diff_list = []
        all_internal_diffusions = []  # 收集所有批次的逆向结果
        all_internal_samples = []  # 收集所有批次的去噪结果
        for batch_idx, x in tqdm.tqdm(enumerate(data_loader)):
            x = x[0].cuda()
            x = x * 2 - 1
            x_T, internal_diffusions = ddim_reverse(model, x, beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T, T=FLAGS.T,
                                                    steps=ddim_reverse_step, return_intermediate=True)

            samples, internal_samples = ddim_denoise(model, FLAGS, internal_diffusions, device=device,
                                                     eta=0, ddim_step=ddim_denoise_step)

            internal_diffusions = torch.cat([inter.unsqueeze(0) for inter in internal_diffusions])
            internal_samples = torch.cat([inter.unsqueeze(0) for inter in reversed(internal_samples)])
            all_internal_diffusions.append(internal_diffusions)
            all_internal_samples.append(internal_samples)
            matched_internal_diffusions = torch.concat([x.unsqueeze(0), internal_diffusions[:-1]])

            diff = ((norm(matched_internal_diffusions) - norm(internal_samples)) ** 2).flatten(2).sum(dim=-1)
            # print(diff.shape)
            diff_list.append(diff)
            
            if score_type == 'mse':
                score = diff
            else:
                raise NotImplementedError

            scores_list.append(score)

            # break
        # 加concat difflist mor
        # print(torch.concat(scores_list,dim=1).shape, torch.concat(diff_list,dim=1).shape)
        return torch.concat(scores_list,dim=1), torch.cat(diff_list,dim=1), torch.cat(all_internal_diffusions,dim=1), torch.cat(all_internal_samples,dim=1)

    member_scores, member_timestep_diff,member_diffusions, member_internal_samples = get_recon_score(test_loader, ddim_forward_step, ddim_gen_step, score_type=score_type)
    nonmember_scores, nonmember_timestep_diff, nonmember_diffusions, nonmember_internal_samples = get_recon_score(generated_loader, ddim_forward_step, ddim_gen_step,
                                                                score_type=score_type)

    for i in range(member_scores.size(0)):
        # print(member_scores[i], member_timestep_diff[i])
        print("i:",i)
        calculate_auc_acc_asr_stat(member_scores[i], nonmember_scores[i])

    t_results = {
    'member_diffusions':member_diffusions[-1],  # 假设这些是成员数据的逆向结果
    'member_internal_samples': member_internal_samples[-1],  # 成员数据的去噪结果
    'nonmember_diffusions': nonmember_diffusions[-1],  # 非成员数据的逆向结果
    'nonmember_internal_samples': nonmember_internal_samples[-1]  # 非成员数据的去噪结果
}
    # exp_data = SeDID_nn.run_NNs(t_results)
    # print("auc:",exp_data['auc'],"acc",exp_data['asr'],"TPRat1FPR",exp_data['TPRat1FPR'],"TPRat0.1FPR",exp_data['TPRat0.1FPR'])

if __name__ == '__main__':
    fix_seed()
    ckpt = os.path.join('/home/mrp_929/projects/SeDID/logs/DDPM_TINY-IN_EPS/ckpt-step1200000.pt')
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
    ])
    flag_path = os.path.join("/home/mrp_929/projects/SeDID/logs/DDPM_TINY-IN_EPS/flagfile_TINY-IN.txt")
    ckpt_path = '/home/mrp_929/projects/SeDID/logs/DDPM_TINY-IN_EPS/ckpt-step1200000_samples.pt'
    output_folder_path = 'generated_images/DDPM_TINY-IN'
    device = 'cuda'
    FLAGS = get_FLAGS(flag_path)
    FLAGS(sys.argv)
    FLAGS.mean_type = 'eps_xt_xt-1'
    FLAGS.T = 1000
    model = get_model(ckpt, FLAGS, WA=True).to('cuda')
    generated_loader = prepare_generated_data_loader(ckpt_path, output_folder_path, transforms, batch_size=128)
    secmi_stat(model, FLAGS)
