"""Perform Detect according to the saved data """

import copy

import torch
import numpy as np
import random

import resnet
from sklearn import metrics


class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        # member_data -= member_data.min()
        # member_data /= member_data.max()
        # nonmember_data -= nonmember_data.min()
        # nonmember_data /= nonmember_data.max()
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

        # # norm data
        # self.data -= self.data.min()
        # self.data /= self.data.max()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        # data -= data.min()
        # data /= data.max()
        return data, self.label[item]


def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

        # print(f'Threshold: {threshold:.8f} ASR: {ASR:.4f} TPR: {TPR:.4f} FPR: {FPR:.4f}')
    print('#############################')
    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold


def split_nn_datasets(t_results, train_portion=0.1, batch_size=128):
    # split training and testing
    # [t, 25000, 3, 32, 32]
    member_diffusion = t_results['member_diffusions']
    member_sample = t_results['member_internal_samples']
    nonmember_diffusion = t_results['nonmember_diffusions']
    nonmember_sample = t_results['nonmember_internal_samples']
    if len(member_diffusion.shape) == 4:
        # with one timestep
        # minus
        num_timestep = 1
        member_concat =  ((member_diffusion - member_sample).abs() ** 2)
        nonmember_concat = ((nonmember_diffusion - nonmember_sample).abs() ** 2)
    elif len(member_diffusion.shape) == 5:
        # with multiple timestep
        # minus
        num_timestep = member_diffusion.size(0)
        member_concat = ((member_diffusion - member_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                       num_timestep * 3,
                                                                                                       32, 32)
        nonmember_concat = ((nonmember_diffusion - nonmember_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                                num_timestep * 3,
                                                                                                                32, 32)
    else:
        raise NotImplementedError

    # train num
    num_train = int(member_concat.size(0) * train_portion)
    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label, train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_timestep


# def net_train(epoch, model, optimizer, data_loader, device='cuda'):


def nn_train(epoch, model, optimizer, data_loader, device='cuda'):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)

        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


@torch.no_grad()
def nn_eval(model, data_loader, device='cuda'):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def nns_attack(t_results, train_portion=0.10, device='cuda'):
    print(train_portion)
    print("resnet18")
    n_epoch = 20
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader, num_timestep = split_nn_datasets(t_results, train_portion=train_portion,
                                                                batch_size=batch_size)
    # initialize NNs
    model = resnet.ResNet18(num_channels=3 * num_timestep * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model eval

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    model.eval()
    # generate member_scores, nonmember_scores
    member_scores = []
    nonmember_scores = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

    member_scores = torch.concat(member_scores).reshape(-1)
    nonmember_scores = torch.concat(nonmember_scores).reshape(-1)
    return member_scores, nonmember_scores, model

def fix_seed():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import pickle
def execute_attack(t_result, type):
    model = None
    if type == 'nns':
        member_scores, nonmember_scores, model = nns_attack(t_result)
        member_scores *= -1
        nonmember_scores *= -1
    else:
        raise NotImplementedError

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores, nonmember_scores, n_points=1000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data = {
        'member_scores': member_scores,  # for histogram
        'nonmember_scores': nonmember_scores,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPRat1FPR': tpr_1_fpr,
        'TPRat0.1FPR': tpr_01_fpr,
        'model': model,
        'threshold': threshold
    }

    return exp_data

def run_NNs(t_result):
    """
    t_result contains the t-th timestep reverse and denoise results.
    It should be a dict with the following structure:

    t_result = {
        'member_diffusions': [],
        'member_internal_samples': [],
        'nonmember_diffusions': [],
        'nonmember_internal_samples': []
    }

    member_diffusions: reverse results of member samples at t-th timestep with the shape of [B, C, H, W]
    member_internal_samples: denoising results of member samples at t-th timestep with the shape of [B, C, H, W]
    nonmember_diffusions: reverse results of nonmember samples at t-th timestep with the shape of [B, C, H, W]
    nonmember_internal_samples: denoising results of nonmember samples at t-th timestep with the shape of [B, C, H, W]

    """
    return execute_attack(t_result, type='nns')
