import torchvision

transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
dataset = torchvision.datasets.ImageFolder(root='/home/mrp_929/projects/data/tiny-imagenet-200/train', transform=transforms)

# 打印图片数量
print(len(dataset))
