import torch
import torch.nn.functional as F


def median_filter(batch: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Implements median filtering for a batch of images.

    Args:
        batch (torch.Tensor): Tensor of images [B, C, H, W].
        kernel_size (int): Size of the median filter kernel.

    Returns:
        torch.Tensor: Tensor of filtered images [B, C, H, W].
    """
    # Ensure kernel size is odd
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    # Pad the batch to handle borders using reflection padding
    pad = kernel_size // 2
    padded_batch = F.pad(batch, pad=(pad, pad, pad, pad), mode='reflect')

    # Extract dimensions
    B, C, H, W = batch.shape

    # Container for the output results
    filtered_batch = torch.zeros_like(batch)

    # Sliding window approach
    for i in range(H):
        for j in range(W):
            # Extract local region (including padding)
            local_region = padded_batch[:, :, i:i + kernel_size, j:j + kernel_size]

            # Calculate median for each position in the channel
            filtered_batch[:, :, i, j] = local_region.reshape(B, C, -1).median(dim=-1).values

    return filtered_batch


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Generates a 2D Gaussian kernel.

    Args:
        kernel_size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: 2D Gaussian kernel.
    """
    coords = torch.arange(kernel_size) - kernel_size // 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize
    return kernel

def gaussian_filter(batch: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """
    Applies Gaussian filtering to a batch of images.

    Args:
        batch (torch.Tensor): Input tensor of shape [B, C, H, W].
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Filtered tensor of shape [B, C, H, W].
    """
    # Ensure kernel size is odd
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(batch.device)
    kernel = kernel.expand(batch.shape[1], 1, kernel_size, kernel_size)  # [C, 1, kH, kW]

    # Pad the batch to handle borders
    pad = kernel_size // 2
    padded_batch = F.pad(batch, pad=(pad, pad, pad, pad), mode='reflect')

    # Apply the Gaussian kernel using convolution
    filtered_batch = F.conv2d(
        padded_batch,
        kernel,
        groups=batch.shape[1]  # Apply the same kernel to each channel independently
    )

    return filtered_batch


def lowpass_filter(batch: torch.Tensor) -> torch.Tensor:
    """
    Apply a spatial filter to a batch of images using convolution.

    Args:
        batch (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor: Filtered tensor of shape [B, C, H, W].
    """

    kernel = torch.tensor([[1, 1, 1],
                           [1, 2, 1],
                           [1, 1, 1]], dtype=torch.float32, device=batch.device) / 10  # Averaging kernel (sum=1)

    # Expand kernel to apply the same filter to all channels
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    kernel = kernel.expand(batch.size(1), -1, -1, -1)  # [C, 1, kH, kW]

    # Apply convolution
    filtered_batch = F.conv2d(batch, kernel, padding=kernel.size(-1) // 2, groups=batch.size(1))

    return filtered_batch


def highpass_filter(batch: torch.Tensor) -> torch.Tensor:
    """
    Apply Sobel high-pass filter to a batch of images to enhance edges.

    Args:
        batch (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor: High-pass filtered tensor of shape [B, C, H, W].
    """
    # Define Sobel filters
    Gx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32, device=batch.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    Gy = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=torch.float32, device=batch.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    # Pad the batch to maintain dimensions after convolution
    batch_padded = F.pad(batch, (1, 1, 1, 1), mode='reflect')  # Padding for 3x3 filters

    # Initialize tensor to store results
    filtered_batch = torch.zeros_like(batch)

    # Apply Sobel filters to each channel independently
    for c in range(batch.size(1)):  # Iterate over channels
        channel = batch_padded[:, c:c+1, :, :]  # Extract single channel with shape [B, 1, H, W]
        grad_x = F.conv2d(channel, Gx, padding=0)  # Gradient in X direction
        grad_y = F.conv2d(channel, Gy, padding=0)  # Gradient in Y direction

        # Combine gradients to compute edge magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        filtered_batch[:, c:c+1, :, :] = grad_magnitude

    return filtered_batch
