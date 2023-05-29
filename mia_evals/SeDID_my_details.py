import copy

import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import numpy as np
import random
# import lpips
import tqdm
import pickle
from sklearn import metrics
from dataset_utils import load_member_data, load_synthetic_dataset
from torchvision.utils import make_grid
from absl import flags
from mia_evals.diffusion import DDIMSampler
from model import UNet
import matplotlib
import math
# from measures import ssim
from dataset_utils import ReconsDataset

from torchvision.utils import save_image
import torch
import torchvision

matplotlib.rcParams['figure.dpi'] = 300
from SeDID_nn import run_NNs
import json

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
    flags.DEFINE_integer('total_steps', 1200000, help='total training steps')
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
    flags.DEFINE_integer('save_step', 10000, help='frequency of saving checkpoints, 0 to disable during training')
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


# 修改代码
checkpoint_path = '/home/usr/projects/DiffusionDetect/DiffusionDetect/logs/DDPM_CIFAR10_EPS/ckpt_samples.pt'
checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())

generated_images = checkpoint['samples'][:50000]
generated_images = np.transpose(generated_images, (0, 2, 3, 1))
import os
from PIL import Image

output_folder = 'my_generated_images'
os.makedirs(output_folder, exist_ok=True)
from torch.utils.data import DataLoader,Subset
num_classes = 1  # 更改类别数量
images_per_class = 50000

for class_idx in range(num_classes):
    class_folder = os.path.join(output_folder, f'CIFAR_new_{class_idx}')
    os.makedirs(class_folder, exist_ok=True)
    
    for img_idx in range(images_per_class):
        img = generated_images[class_idx * images_per_class + img_idx]
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(os.path.join(class_folder, f'image{img_idx}.png'))

transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
# Collect all generated image paths
image_paths = []
for class_idx in range(num_classes):
    class_folder = os.path.join(output_folder, f'class{class_idx}')
    class_image_paths = glob.glob(os.path.join(class_folder, "*.png"))
    class_image_paths = random.sample(class_image_paths, 50000)  # Randomly select  images per class
    image_paths.extend(class_image_paths)
from torch.utils.data import Dataset
# Modify CustomImageDataset to accept a list of image paths
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create labels for generated images
labels = []
for class_idx in range(num_classes):
    labels.extend([class_idx] * 50000)  # Assuming 3000 images per class

# Create the CustomImageDataset
generated_dataset = CustomImageDataset(image_paths, labels, transform=transforms)


# Create the DataLoader
generated_loader = DataLoader(generated_dataset, batch_size=128, shuffle=True)

def calculate_auc_asr_stat(member_scores, nonmember_scores):
    print(f'member score: {member_scores.mean():.4f} nonmember score: {nonmember_scores.mean():.4f}')

    total = member_scores.size(0) + nonmember_scores.size(0)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = min(member_scores.max(), nonmember_scores.max()).item()
    print(min_score, max_score)

    TPR_list = []
    FPR_list = []

    best_asr = 0

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    for threshold in torch.arange(min_score, max_score, (max_score - min_score) / 1000):
        acc = ((member_scores >= threshold).sum() + (nonmember_scores < threshold).sum()) / total

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        # print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.4f} \t TPR: {TPR:.4f} \t FPR: {FPR:.4f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc} \t ASR: {best_asr} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')
from PIL import Image
import os
import glob
def save_image(img, path):
    torchvision.utils.save_image(img, path)
def SeDID_stat(model, FLAGS, batch_size=128, score_type='mse'):
    timestep = 200
    print(timestep)
    target_steps = list(range(0, 1001, timestep))[1:]
    ddim_forward_step = [target_steps, [0] + target_steps[:-1]]
    ddim_gen_step = [list(reversed(target_steps)), list(reversed(target_steps[:-1])) + [0]]
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='/home/usr/projects/ComputerVisionModels/data', train=True, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def norm(x):
        return (x + 1) / 2

    def get_recon_score(data_loader, ddim_reverse_step, ddim_denoise_step, data_set_type, score_type='ssim'):
        scores_list = []
        diff_list = []
        matched_internal_diffusions_list = []
        internal_samples_list = []
        for batch_idx, x in tqdm.tqdm(enumerate(data_loader)):
            if batch_idx >= 20:
                break
            x = x[0].cuda()
            x = x * 2 - 1
            x_T, internal_diffusions = ddim_reverse(model, x, beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T, T=FLAGS.T,
                                                    steps=ddim_reverse_step, return_intermediate=True)

            samples, internal_samples = ddim_denoise(model, FLAGS, internal_diffusions, device=device,
                                                     eta=0, ddim_step=ddim_denoise_step)

            internal_diffusions = torch.cat([inter.unsqueeze(0) for inter in internal_diffusions])
            internal_samples = torch.cat([inter.unsqueeze(0) for inter in reversed(internal_samples)])
            internal_samples_list.append(internal_samples)
            matched_internal_diffusions = torch.cat([x.unsqueeze(0), internal_diffusions[:-1]])
            matched_internal_diffusions_list.append(matched_internal_diffusions)
            diff = ((norm(matched_internal_diffusions) - norm(internal_samples)) ** 2).flatten(2).sum(dim=-1)
            diff_list.append(diff)

            if score_type == 'mse':
                score = diff
            else:
                raise NotImplementedError

            scores_list.append(score)

            if batch_idx < 20:
                # Save original, denoised and noised images...
                key_steps = [0, len(internal_diffusions)//6, len(internal_diffusions)//3, len(internal_diffusions)//2, 2*len(internal_diffusions)//3, 5*len(internal_diffusions)//6, len(internal_diffusions)-1]

                # Save diffused images at key steps
                for t in key_steps:
                    save_path = os.path.join("/home/usr/projects/DiffusionDetect/DiffusionDetect/img_plot/cifar10", f"{data_set_type}_diffused_{batch_idx}_t_{t}.png")
                    save_image(internal_diffusions[t][0], save_path)

                # Save denoised samples at key steps
                for t in key_steps:
                    save_path = os.path.join("/home/usr/projects/DiffusionDetect/DiffusionDetect/img_plot/cifar10", f"{data_set_type}_sample_{batch_idx}_t_{t}.png")
                    save_image(internal_samples[t][0], save_path)


        image_dir = "/home/usr/projects/DiffusionDetect/DiffusionDetect/img_plot/cifar10"
        os.makedirs(image_dir, exist_ok=True)

        # 对于每张图片
        for i in range(10,50):
            diffused_images = []
            sample_images = []
        
            # 保存7个关键步骤的扩散和样本图像
            for t in key_steps:
                diffused_path = os.path.join(image_dir, f"{data_set_type}_diffused_{i}_t_{t}.png")
                diffused_images.append(Image.open(diffused_path))
                sample_path = os.path.join(image_dir, f"{data_set_type}_sample_{i}_t_{t}.png")
                sample_images.append(Image.open(sample_path))
        
            # 组合7个关键步骤的扩散图像
            widths, heights = zip(*(img.size for img in diffused_images))
            total_width = sum(widths)
            max_height = max(heights)
            new_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in diffused_images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            new_img.save(os.path.join(image_dir, f"{data_set_type}_diffused_combined_{i}.png"))
        
            # 组合7个关键步骤的样本图像
            widths, heights = zip(*(img.size for img in sample_images))
            total_width = sum(widths)
            max_height = max(heights)
            new_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in sample_images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            new_img.save(os.path.join(image_dir, f"{data_set_type}_sample_combined_{i}.png"))
        return torch.cat(scores_list,dim=1), torch.cat(diff_list,dim=1), torch.cat(matched_internal_diffusions_list,dim=1), torch.cat(internal_samples_list,dim=1)

    member_scores, member_timestep_diff,member_diffusions,member_internal_samples = get_recon_score(test_loader, ddim_forward_step, ddim_gen_step, "real", score_type=score_type)
    nonmember_scores, nonmember_timestep_diff,nonmember_diffusions,nonmember_internal_samples = get_recon_score(generated_loader, ddim_forward_step, ddim_gen_step, "generated", score_type=score_type)

    for i in range(member_scores.size(0)):
        # print(member_scores[i], member_timestep_diff[i])
        print("i:",i)
        calculate_auc_asr_stat(member_scores[i], nonmember_scores[i])
    
    # t_result = {
    #     'member_diffusions': member_diffusions[-2:],
    #     'member_internal_samples': member_internal_samples[-2:],
    #     'nonmember_diffusions': nonmember_diffusions[-2:],
    #     'nonmember_internal_samples': nonmember_internal_samples[-2:]
    # }
    # # Save the data
    # save_dir = '/home/usr/projects/DiffusionDetect/DiffusionDetect/logs/DDPM_CIFAR10_EPS'
    # # Get the data to save
    # exp_data = run_NNs(t_result)
    # print(exp_data)
    # with open(os.path.join(save_dir, "exp_data.pkl"), "wb") as f:
    #     pickle.dump(exp_data, f)

    
# Use the function to save the data
        
if __name__ == '__main__':
    fix_seed()
    ckpt = os.path.join('/home/usr/projects/DiffusionDetect/DiffusionDetect/logs/DDPM_CIFAR10_EPS/ckpt.pt')
    flag_path = os.path.join("/home/usr/projects/DiffusionDetect/DiffusionDetect/mia/flagfile.txt")
    # flag_path = os.path.join(exp_root, 'flagfile.txt')
    device = 'cuda'
    FLAGS = get_FLAGS(flag_path)
    FLAGS(sys.argv)
    FLAGS.mean_type = 'eps_xt_xt-1'
    FLAGS.T = 1000
    model = get_model(ckpt, FLAGS, WA=True).to('cuda')
    SeDID_stat(model, FLAGS)
