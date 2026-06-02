import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(os.environ.get('SEDID_DATASET_ROOT', PROJECT_ROOT / 'datasets'))
DEFAULT_LOG_ROOT = Path(os.environ.get('SEDID_LOG_ROOT', PROJECT_ROOT / 'logs'))
DEFAULT_GENERATED_ROOT = Path(os.environ.get('SEDID_GENERATED_ROOT', PROJECT_ROOT / 'generated_images'))
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def dataset_root(*parts):
    return str(DEFAULT_DATASET_ROOT.joinpath(*parts))


def generated_root(*parts):
    return str(DEFAULT_GENERATED_ROOT.joinpath(*parts))


def resolve_flagfile(experiment_name, *candidates):
    logdir = DEFAULT_LOG_ROOT / experiment_name
    names = candidates or ('flagfile.txt',)
    for name in names:
        path = logdir / name
        if path.exists():
            return str(path)
    return str(logdir / names[0])


def experiment_paths(experiment_name, ckpt_name, sample_name, generated_name=None, flagfile_candidates=('flagfile.txt',)):
    logdir = DEFAULT_LOG_ROOT / experiment_name
    return {
        'ckpt': str(logdir / ckpt_name),
        'ckpt_sample_path': str(logdir / sample_name),
        'output_folder_path': generated_root(generated_name or experiment_name),
        'flag_path': resolve_flagfile(experiment_name, *flagfile_candidates),
    }


def score_thresholds(member_scores, nonmember_scores, steps=1001):
    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = max(member_scores.max(), nonmember_scores.max()).item()
    if max_score == min_score:
        thresholds = torch.tensor([min_score], device=member_scores.device)
    else:
        thresholds = torch.linspace(min_score, max_score, steps=steps, device=member_scores.device)
    return min_score, max_score, thresholds
