import torch
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List


def plot_samples_from_dataloader(dataloader: torch.utils.data.DataLoader) -> None:
    """
    Plot a sample of noisy and clean images from a DataLoader.

    This function visualizes the first 5 images of the first batch from a DataLoader,
    displaying the noisy images in the first row and the corresponding clean
    images in the second row.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing
            batches of noisy and clean images.

    Returns:
        None: The function displays the plot but does not return anything.
    """
    noisy_batch, clean_batch = next(iter(dataloader))

    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axs[0, i].imshow(noisy_batch[i].permute(1, 2, 0))
        axs[0, i].set_title("Noisy")
        axs[0, i].axis("off")

        axs[1, i].imshow(clean_batch[i].permute(1, 2, 0))
        axs[1, i].set_title("Clean")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_denoising_examples(noisy_imgs: torch.Tensor, clean_imgs: torch.Tensor, denoised_imgs: torch.Tensor, title: str = '', num_images: int = 5):
    """
    Plot examples of noisy, denoised, and clean images side-by-side for comparison.

    Args:
        noisy_imgs (torch.Tensor): Tensor of noisy images with shape [B, C, H, W].
        clean_imgs (torch.Tensor): Tensor of clean (ground truth) images with shape [B, C, H, W].
        denoised_imgs (torch.Tensor): Tensor of denoised images with shape [B, C, H, W].
        title (str, optional): Title of the entire plot. Defaults to an empty string.
        num_images (int, optional): Number of examples to plot. Defaults to 5.

    Returns:
        None: The function displays the plot but does not return anything.
    """
    noisy_imgs = noisy_imgs.cpu().permute(0, 2, 3, 1)
    clean_imgs = clean_imgs.cpu().permute(0, 2, 3, 1)
    denoised_imgs = denoised_imgs.cpu().permute(0, 2, 3, 1)

    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 2.5, 7))
    for i in range(num_images):
        axs[0, i].imshow(noisy_imgs[i])
        axs[0, i].set_title("Noisy")
        axs[0, i].axis("off")

        axs[1, i].imshow(denoised_imgs[i].clamp(0, 1))
        axs[1, i].set_title("Denoised")
        axs[1, i].axis("off")

        axs[2, i].imshow(clean_imgs[i])
        axs[2, i].set_title("Clean")
        axs[2, i].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_loss: List[float], val_loss: List[float]):
    """
    Plot the training and validation loss curves over epochs.

    This function creates a line plot to visualize how the training and
    validation loss values change across epochs during model training.

    Args:
        train_loss (List[float]): List of training loss values for each epoch.
        val_loss (List[float]): List of validation loss values for each epoch.

    Returns:
        None: The function displays the plot but does not return anything.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(train_loss, label='train')
    sns.lineplot(val_loss, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and val losses')
