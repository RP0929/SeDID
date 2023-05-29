import os.pathtransforms
import sys

sys.path.append('.')

import torch
import torchvision.datasets
import numpy as np

#from DiffusionDetect.score import fid
from score import fid
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN

class ReconsDataset(torch.utils.data.Dataset):

    def __init__(self, data, transforms=None):
        super(ReconsDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image


class MIASTL10(torchvision.datasets.STL10):

    def __init__(self, idxs, **kwargs):
        super(MIASTL10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASTL10, self).__getitem__(item)


class MIACelebA(torchvision.datasets.CelebA):

    def __init__(self, idxs, **kwargs):
        super(MIACelebA, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACelebA, self).__getitem__(item)


class MIASVHN(torchvision.datasets.SVHN):

    def __init__(self, idxs, **kwargs):
        super(MIASVHN, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASVHN, self).__getitem__(item)


class MIACIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)


class MIACIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR100, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR100, self).__getitem__(item)


class MIAImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, idxs, **kwargs):
        super(MIAImageFolder, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIAImageFolder, self).__getitem__(item)


def test_celebA():
    count = 0
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CelebA('/home/jd3734@drexel.edu/datasets/celeba', split='train', download=True,
                                          transform=transforms)
    # dataset = MIACelebA(list(range(1000)), root='/home/jd3734@drexel.edu/datasets/celeba', split='train', download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    for img, label in dataloader:
        # print(img.shape)
        count += img.size(0)
        # print(label.shape)
    print(count)


def calculate_dataset_stats(dataset_name, batch_size=128, only_member=False, only_nonmember=False):
    dataset_root = '/home/jd3734@drexel.edu/datasets'
    splits = np.load(f'./member_splits/{dataset_name.upper()}_train_ratio0.5.npz')
    member_idxs = splits['mia_train_idxs']
    non_member_idxs = splits['mia_eval_idxs']
    if dataset_name.upper() == 'CIFAR10':
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if only_member:
            dataset = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                 transform=transforms)
        elif only_nonmember:
            dataset = MIACIFAR10(non_member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                 transform=transforms)
        else:
            dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, 'cifar10'), train=True,
                                                   transform=transforms)
        total_images = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        m1, s1 = fid.get_statistics_from_dataloader(data_loader, num_images=total_images, use_torch=True, verbose=True)
    elif dataset_name.upper() == 'CIFAR100':
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if only_member:
            dataset = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                  transform=transforms)
        else:
            dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_root, 'cifar100'), train=True,
                                                    transform=transforms)
        total_images = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        m1, s1 = fid.get_statistics_from_dataloader(data_loader, num_images=total_images, use_torch=True, verbose=True)
    elif dataset_name.upper() == 'CELEBA':
        # for CelebA, center crop 140 and resize to 32
        transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(140),
                                                     torchvision.transforms.Resize(32),
                                                     torchvision.transforms.ToTensor()])
        if only_member:
            dataset = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                transform=transforms)
        else:
            dataset = torchvision.datasets.CelebA(root=os.path.join(dataset_root, 'celeba'), split='train',
                                                  transform=transforms)
        total_images = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        m1, s1 = fid.get_statistics_from_dataloader(data_loader, num_images=total_images, use_torch=True, verbose=True)
    elif dataset_name.upper() == 'SVHN':
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if only_member:
            dataset = MIASVHN(member_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                              transform=transforms)
        else:
            dataset = torchvision.datasets.SVHN(root=os.path.join(dataset_root, 'svhn'), split='train',
                                                transform=transforms)
        total_images = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        m1, s1 = fid.get_statistics_from_dataloader(data_loader, num_images=total_images, use_torch=True, verbose=True)
    else:
        raise NotImplemented

    return m1, s1


def load_member_data(dataset_name, batch_size=128, shuffle=False, randaugment=False):
    member_split_root = '/home/kongfei/workspace/DiffusionDetect/mia_evals/member_splits'
    dataset_root = '/home/kongfei/data/datasets/pytorch'
    if dataset_name.upper() == 'CIFAR10':
        splits = np.load(os.path.join(member_split_root, 'CIFAR10_train_ratio0.5.npz'))
        # member_idxs = splits['mia_train_idxs']
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        member_set = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                transform=transforms)
        nonmember_set = MIACIFAR10(nonmember_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                   transform=transforms)
    elif dataset_name.upper() == 'CIFAR100':
        splits = np.load(os.path.join(member_split_root, 'CIFAR100_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        member_set = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                 transform=transforms)
        nonmember_set = MIACIFAR100(nonmember_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                    transform=transforms)
    elif dataset_name.upper() == 'SVHN':
        splits = np.load(os.path.join(member_split_root, 'SVHN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        member_set = MIASVHN(member_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                             transform=transforms)
        nonmember_set = MIASVHN(nonmember_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                                transform=transforms)
    elif dataset_name.upper() == 'CELEBA':
        splits = np.load(os.path.join(member_split_root, 'CELEBA_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                               transform=transforms, download=True)
        nonmember_set = MIACelebA(nonmember_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                  transform=transforms, download=True)
    elif dataset_name.upper() == 'STL10':
        splits = np.load(os.path.join(member_split_root, 'STL10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='train',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'stl10'), split='train',
                                 download=True, transform=transforms)
    elif dataset_name.upper() == 'STL10-U':
        splits = np.load(os.path.join(member_split_root, 'STL10_Unlabeled_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                                 download=True, transform=transforms)
    elif dataset_name.upper() == 'TINY-IN':
        splits = np.load(os.path.join(member_split_root, 'TINY-IN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIAImageFolder(member_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                    transform=transforms)
        nonmember_set = MIAImageFolder(nonmember_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                       transform=transforms)
    else:
        raise NotImplemented

    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, shuffle=shuffle)
    return member_set, nonmember_set, member_loader, nonmember_loader

# checkpoint_path = '/home/usr/projects/DiffusionDetect/DiffusionDetect/logs/DDPM_CIFAR10_EPS/ckpt_samples.pt'
# checkpoint = torch.load(checkpoint_path)

# generated_images = checkpoint['samples'][:50000]
# generated_images = np.transpose(generated_images, (0, 2, 3, 1))
# import os
# from PIL import Image
# import glob
# from torch.utils.data import Dataset
# output_folder = 'model_generated_images'

# os.makedirs(output_folder, exist_ok=True)

# for img_idx in range(len(generated_images)):
#     img = generated_images[img_idx]
#     img = Image.fromarray((img * 255).astype(np.uint8))
#     img.save(os.path.join(output_folder, f'image{img_idx}.png'))

# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((32,32)),
#     torchvision.transforms.ToTensor()
# ])

# # Collect all generated image paths
# image_paths = glob.glob(os.path.join(output_folder, "*.png"))

# # Modify CustomImageDataset to accept a list of image paths
# class GeneratedDataset(Dataset):
#     def __init__(self, image_paths, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path)

#         if self.transform:
#             image = self.transform(image)

#         return image
# 修改代码
import glob
import random
checkpoint_path = '/home/usr/projects/DiffusionDetect/DiffusionDetect/logs/DDPM_CIFAR10_EPS/ckpt_samples.pt'
checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())

generated_images = checkpoint['samples'][:50000]
generated_images = np.transpose(generated_images, (0, 2, 3, 1))
import os
from PIL import Image

output_folder = 'Diffusion_generated_images/CIFAR10'
os.makedirs(output_folder, exist_ok=True)
from torch.utils.data import DataLoader
num_classes = 1  # 更改类别数量
images_per_class = 50000

for class_idx in range(num_classes):
    class_folder = os.path.join(output_folder, f'class{class_idx}')
    os.makedirs(class_folder, exist_ok=True)
    
    for img_idx in range(images_per_class):
        img = generated_images[class_idx * images_per_class + img_idx]
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(os.path.join(class_folder, f'image{img_idx}.png'))

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])
# Collect all generated image paths
image_paths = []
for class_idx in range(num_classes):
    class_folder = os.path.join(output_folder, f'class{class_idx}')
    class_image_paths = glob.glob(os.path.join(class_folder, "*.png"))
    class_image_paths = random.sample(class_image_paths, 50000) 
    image_paths.extend(class_image_paths)
from torch.utils.data import Dataset,DataLoader,Subset
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
    labels.extend([class_idx] * 50000)  # Assuming 50000 images per class


def load_generated_data(dataset_name, batch_size=128, shuffle=False, randaugment=False):
    member_split_root = '/home/kongfei/workspace/DiffusionDetect/mia_evals/member_splits'
    dataset_root = '/home/kongfei/data/datasets/pytorch'
    if dataset_name.upper() == 'CIFAR10':
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor()])
            
        member_set = CustomImageDataset(image_paths, labels, transform=transforms)
        nonmember_set = torchvision.datasets.CIFAR10(root='/home/usr/projects/ComputerVisionModels/data', train=True, download=True, transform=transforms)
        
    elif dataset_name.upper() == 'CELEBA':
        splits = np.load(os.path.join(member_split_root, 'CELEBA_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = CustomImageDataset(image_paths, labels, transform=transforms)
        nonmember_set = CelebA(root='/home/usr/projects/data', split='train',
                             transform=transforms, download=False)
        nonmember_set = Subset(nonmember_set, range(50000))
    elif dataset_name.upper() == 'TINY-IN':
        splits = np.load(os.path.join(member_split_root, 'TINY-IN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = CustomImageDataset(image_paths, labels, transform=transforms)
        nonmember_set = torchvision.datasets.ImageFolder(root='/home/usr/projects/data/tiny-imagenet-200',
                                                       transform=transforms)
        nonmember_set = Subset(nonmember_set, range(50000))
    else:
        raise NotImplemented
    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, shuffle=shuffle)
    return member_set, nonmember_set, member_loader, nonmember_loader

def load_synthetic_dataset(dataset, synthetic_images_path, image_num=50000):
    if dataset.upper() == 'CIFAR10':
        synthetic_images = torch.load(synthetic_images_path)['samples'][:image_num]
        synthetic_dataset = ReconsDataset(synthetic_images)
    else:
        raise NotImplemented

    return synthetic_dataset


def test_stats():
    generated_images = \
        torch.load('../experiments/DDPM_CIFAR10_HalfTrain_baseline/ckpt-step720000_samples_NonMemberForGAN.pt')[
            'samples'][
            :50000]
    m1, s1 = fid.get_statistics(generated_images, num_images=50000, batch_size=128, use_torch=True, verbose=True)
    m2, s2 = calculate_dataset_stats('CIFAR10', batch_size=128, only_nonmember=True)
    fid_score = fid.calculate_frechet_distance(mu1=m1, sigma1=s1, mu2=m2, sigma2=s2, use_torch=True)
    np.savez('./test_stat.npz', mu=m1.cpu().numpy(), sigma=s1.cpu().numpy())
