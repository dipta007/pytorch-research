import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import wandb
from wandb import AlertLevel

log = logging.getLogger(__name__)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError
        return self.get(attr, None)

    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # if input size is fixed for every iteration, set to True
    )
    torch.cuda.manual_seed_all(seed)


def get_repo_name():
    repo_name = os.path.basename(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .strip()
        .decode()
    )
    return repo_name


def get_commit_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode()


def print_repo_info():
    log.info("*" * 44)
    log.info(" ".join(sys.argv))
    log.info("")
    git_commit_hash = get_commit_hash()
    repo_name = get_repo_name()
    log.info(f"Git repo: {repo_name}")
    log.info(f"Git commit hash: {git_commit_hash}")
    log.info("*" * 44)


def print_config(config):
    log.info("-" * 44)
    log.info(f"CMD: {' '.join(sys.argv)}")
    log.info("")
    log.info(f"Hydra Configs:\n{json.dumps(config, indent=4)}")
    log.info("-" * 44)


def to_device(data, device):
    device = torch.device(device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    if isinstance(data, nn.Module):
        return data.to(device)
    if isinstance(data, nn.ModuleList):
        return nn.ModuleList([to_device(x, device) for x in data])
    if isinstance(data, nn.Parameter):
        return data.to(device)
    if isinstance(data, nn.ParameterList):
        return nn.ParameterList([to_device(x, device) for x in data])
    if isinstance(data, nn.Embedding):
        return data.to(device)

    return data.to(device)
    raise TypeError(f"Unsupported type: {type(data)}")


def wandb_alert(title, text="", level=AlertLevel.INFO):
    try:
        wandb.alert(get_repo_name() + " - " + title, text, level)
    except Exception as e:
        pass


def wandb_log(wandb_dict):
    try:
        wandb.log(wandb_dict)
    except Exception as e:
        pass


def wandb_log_code(run):
    included_files = [
        ".py",
        ".yaml",
        ".ipynb",
        ".md",
        ".sh",
    ]
    run.log_code(
        "./", include_fn=lambda f: any(f.endswith(ext) for ext in included_files)
    )


def save_model(model, path):
    dir = "/".join(path.split("/")[:-1])
    ensure_dir(dir)
    torch.save(model, path)


def load_model(path):
    return torch.load(path)
