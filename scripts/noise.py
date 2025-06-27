import numpy as np
import torch
from torch.utils.data import Dataset
import random

def add_gaussian_noise(img: torch.Tensor, mean = 0.0, std = 0.1):
    noise = torch.randn_like(img) * std + mean
    return torch.clamp(img + noise, 0., 1.)

def add_salt_and_pepper(img: torch.Tensor, amount = 0.02):
    img_np = img.clone().numpy()
    c, h, w = img_np.shape
    num_pixels = int(amount * h * w)

    for ch in range(c):
        # salt (white)
        coords = [np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels)]
        img_np[ch, coords[0], coords[1]] = 1.0

        # pepper (black)
        coords = [np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels)]
        img_np[ch, coords[0], coords[1]] = 0.0

    return torch.tensor(img_np).clamp(0., 1.)

def add_speckle_noise(img: torch.Tensor, std = 0.1):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + img * noise, 0., 1.)

def add_random_noise(img: torch.Tensor):
    noise_type = random.choice(["gaussian" , "salt_pepper", "speckle"])
    if noise_type == "gaussian":
        return add_gaussian_noise(img, std=0.1)
    elif noise_type == "salt_pepper":
        return add_salt_and_pepper(img, amount=0.02)
    elif noise_type == "speckle":
        return add_speckle_noise(img, std=0.1)

# Custom Dataset class
class NoisyCIFAR10(Dataset):
    def __init__(self, clean_dataset):
        self.clean_dataset = clean_dataset

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.clean_dataset[idx]
        noisy_img = add_random_noise(clean_img)
        return noisy_img, clean_img  # (input (noisy) image, target (denoised) image)
