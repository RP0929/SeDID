import torch
from pprint import pprint

method = "iter"
print(f"method:{method}")

results = torch.load(f'/home/mrp_929/projects/DiffusionMIA/DiffusionMIA/result.pt')
pprint(results)

# statistics = torch.load(f'/home/mrp_929/projects/DiffusionMIA/DiffusionMIA/statistics/cifar10_pretrain_{method}_10.pt')
# pprint(statistics)
