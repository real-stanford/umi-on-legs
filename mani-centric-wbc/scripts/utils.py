import logging
import os
import pickle
import random
import re
import time
import typing
from typing import Any, Dict, List, Sequence, Tuple

import git
import hydra
import numpy as np
from isaacgym import gymapi, gymutil
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

import wandb
from legged_gym.env.obs import ObservationAttribute

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def set_seed(seed: int):
    logging.info("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("DETERMINISTIC", 0) == 1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def recursively_remove_key(
    source: Dict[str, Any], target: Dict[str, Any], remove_key_words: List[str]
):
    for key in source.keys():
        if isinstance(key, str) and any(kw in key for kw in remove_key_words):
            continue
        if isinstance(source[key], dict):
            target[key] = {}
            recursively_remove_key(source[key], target[key], remove_key_words)
        else:
            target[key] = source[key]
    return target


def recursively_lists_to_dict(
    source: Dict[str, Any], target: Dict[str, Any]
) -> Dict[str, Any]:
    # converts all lists into dictionaries with keys [0], [1], ...
    for key in source.keys():
        if isinstance(source[key], dict):
            target[key] = {}
            recursively_lists_to_dict(source[key], target[key])
        elif isinstance(source[key], list):
            target[key] = {str(i): source[key][i] for i in range(len(source[key]))}
        else:
            target[key] = source[key]
    return target


def setup(config: Dict[str, Any], seed: int):
    config["git_hash"] = git.Repo(search_parent_directories=True).head.object.hexsha
    wandb.init(
        **config["wandb"],
        config=recursively_lists_to_dict(
            recursively_remove_key(config, {}, ["wandb", "device"]), {}
        ),
    )
    assert wandb.run.dir is not None, "Wandb initialization failed"
    # save config to path
    OmegaConf.save(config, f"{wandb.run.dir}/config.yaml")
    pickle.dump(
        config,
        open(f"{wandb.run.dir}/config.pkl", "wb"),
    )
    set_seed(seed)
    torch.set_num_threads(1)


def limit_threads(n: int = 1):
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
