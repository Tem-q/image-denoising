# **Image Denoising Using Classical Filters and Deep Learning Methods**

This project explores and compares various methods for image denoising, including classical image processing filters and modern deep learning architectures such as U-Net and Autoencoders.

---

## **Table of Contents**
1. [Overview](#overview)
1. [Dataset](#dataset)
1. [Results](#results)
1. [Project Structure](#folder-structure)
1. [License](#license)

---

## **Overview**
Noise in images is a common issue in computer vision. This project demonstrates how to remove noise using:
- **Classical methods:** Median filter, Gaussian filter, spatial filters.
- **Deep Learning models:** Simple CNN, U-Net and Autoencoders trained for denoising tasks.

The goal is to evaluate the effectiveness of these methods on a standardized dataset (CIFAR-10).

---

## **Dataset**
The project uses the [**CIFAR-10**](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
- Contains 60,000 images in 10 classes.
- Automatically downloaded to the `data/` folder during training.

> Note: The `data/` folder is ignored in the repository, but the script will download it automatically.

---

## **Results**



---

## **Project Structure**

```
image-denoising/
│
├── models/ # Directory for saved model weights
├── scripts/ # Directory for training and utility scripts
├── image_denoising.ipynb # Main Jupyter notebook with experiments and comparisons
```
---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.
