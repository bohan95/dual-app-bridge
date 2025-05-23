import argparse
import importlib
import omegaconf.dictconfig

from Register import Registers
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner

import torch
import torch.nn.functional as F
import math

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner


def calculate_psnr(img1, img2):
    """
    Compute PSNR
    :param img1: Original image, shape (B, C, H, W)
    :param img2: Reconstructed image, shape (B, C, H, W)
    :return: PSNR value
    """
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 if img1.dtype == torch.float32 else 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM
    :param img1: Original image, shape (B, C, H, W)
    :param img2: Reconstructed image, shape (B, C, H, W)
    :param window_size: Size of the Gaussian window
    :param C1: Constant 1 used in SSIM calculation
    :param C2: Constant 2 used in SSIM calculation
    :return: SSIM value
    """
    def gaussian_window(window_size, sigma):
        gauss = torch.tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    _, C, H, W = img1.size()
    window = gaussian_window(window_size, 1.5).unsqueeze(1)
    window = window @ window.t() 
    window = window.unsqueeze(0).unsqueeze(0)  
    window = window.expand(C, 1, window_size, window_size).to(img1.device)  
    window = window / window.sum()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


