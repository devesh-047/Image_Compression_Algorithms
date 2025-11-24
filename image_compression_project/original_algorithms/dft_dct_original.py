# -*- coding: utf-8 -*-
"""Image Compression - Block-based DCT/DFT with Quantization

Updated to use JPEG-like block processing with quantization matrices
instead of simple threshold-based compression.

Original file is located at
    https://colab.research.google.com/drive/1dNBEKBjNjHbZyp-1RAEe6GcilVCxR0mO
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct, idct, fft2, ifft2
import os

def create_sample_image(size=512):
    """
    Generates a synthetic test image (Zone Plate), converts it to an
    RGB format (three identical channels) for consistency.
    """
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img_gray = np.sin(R**2)
    # Normalize to 0-255 uint8
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)

    # Stack three identical channels to simulate a color image
    return np.stack([img_gray, img_gray, img_gray], axis=-1)

def load_external_image(file_path):
    """
    Loads an external image file and returns it as an RGB (H x W x 3)
    NumPy array (uint8, 0-255 range).
    """
    if not os.path.exists(file_path):
        print(f"Error: Image file not found at '{file_path}'. Using sample (color) image instead.")
        return create_sample_image(512)

    print(f"Loading image from: {file_path}")
    try:
        img = mpimg.imread(file_path)

        # Convert floating point images (0.0-1.0) to uint8 (0-255)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Handle alpha channel if present (4 channels -> 3 channels)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[..., :3]

        # If image is grayscale (2D), stack it to create 3 channels
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # Final check to ensure it's a 3-channel RGB image
        if len(img.shape) == 3 and img.shape[2] == 3:
            return img.astype(np.uint8)

        raise ValueError("Unsupported image format or channel count.")

    except Exception as e:
        print(f"An error occurred while loading or processing the image: {e}. Using sample (color) image instead.")
        return create_sample_image(512)

def to_grayscale(image_rgb):
    """
    Converts an RGB image array (H x W x 3) to a grayscale array (H x W)
    using the standard luminosity method (for histogram display).
    """
    # Ensure input is float64 for calculation and normalized to 0-1
    img_float = image_rgb.astype(np.float64) / 255.0

    # Check if already grayscale (2D)
    if len(img_float.shape) == 2:
        return (img_float * 255).astype(np.uint8)

    # Standard luminosity conversion for RGB
    # Grayscale = R*0.2989 + G*0.5870 + B*0.1140
    grayscale = np.dot(img_float[...,:3], [0.2989, 0.5870, 0.1140])
    return (grayscale * 255).astype(np.uint8)


def calculate_metrics(original, compressed):
    """
    Calculates MSE and PSNR. Works on both 2D (grayscale) and 3D (color) arrays.
    """
    # Convert to float64 for accurate calculation
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)

    # 1. MSE (calculated across all dimensions/channels)
    mse = np.mean((original - compressed) ** 2)

    # 2. PSNR
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 255.0
        # The PSNR calculation remains the same, using max_pixel=255
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return mse, psnr

def get_compression_ratio(quantized_coeffs_stack):
    """
    Calculates the theoretical compression ratio based on the number of
    zero coefficients vs total coefficients (proxy for RLE efficiency).
    CR = Total Coeffs / Non-Zero Coeffs
    This function is used for real-valued DCT coefficients.
    """
    total_elements = quantized_coeffs_stack.size
    non_zero_count = np.count_nonzero(quantized_coeffs_stack)

    if non_zero_count == 0: return 0.0
    return total_elements / non_zero_count


# ==============================================================================
# Block Compression with Quantization (JPEG-like)
# ==============================================================================

# Standard 8x8 Luminance Quantization Matrix (JPEG baseline)
JPEG_QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def dct_2d(image):
    """2D DCT-II transform"""
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct_2d(coeffs):
    """Inverse 2D DCT-II transform"""
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

def block_compression(channel, block_size, transform_func, inverse_transform_func, q_matrix):
    """
    Performs block-wise transform, quantization, and inverse steps.
    This simulates JPEG-like compression.
    
    Args:
        channel: Single channel image (H x W)
        block_size: Size of blocks (typically 8 for JPEG)
        transform_func: Transform function (dct_2d or fft2)
        inverse_transform_func: Inverse transform (idct_2d or ifft2)
        q_matrix: Quantization matrix
        
    Returns:
        reconstructed_channel: Reconstructed image
        quantized_coeffs_stack: All quantized coefficients (for CR calculation)
    """
    H, W = channel.shape
    reconstructed_channel = np.zeros_like(channel, dtype=np.float64)
    all_quantized_coeffs = []

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = channel[i:i + block_size, j:j + block_size].astype(np.float64)

            # Center data (shift by 128 for JPEG)
            block_centered = block - 128.0

            # Step 1: Transform
            transformed_coeffs = transform_func(block_centered)

            # Step 2: Quantization (Lossy compression step)
            if transform_func == fft2:
                # Quantize real and imaginary parts separately for DFT
                quantized_coeffs_real = np.round(np.real(transformed_coeffs) / q_matrix)
                quantized_coeffs_imag = np.round(np.imag(transformed_coeffs) / q_matrix)
                quantized_coeffs = quantized_coeffs_real + 1j * quantized_coeffs_imag
            else:  # DCT
                quantized_coeffs = np.round(transformed_coeffs / q_matrix)

            all_quantized_coeffs.append(quantized_coeffs.ravel())

            # Step 3: De-Quantization
            if transform_func == fft2:
                dequantized_coeffs = quantized_coeffs_real * q_matrix + 1j * quantized_coeffs_imag * q_matrix
            else:  # DCT
                dequantized_coeffs = quantized_coeffs * q_matrix

            # Step 4: Inverse Transform
            if inverse_transform_func == ifft2:
                # Take the real part for IDFT
                recon_block = np.real(inverse_transform_func(dequantized_coeffs))
            else:  # IDCT
                recon_block = inverse_transform_func(dequantized_coeffs)

            # De-center data (shift back by 128)
            recon_block += 128.0

            # Place reconstructed block back
            reconstructed_channel[i:i + block_size, j:j + block_size] = recon_block

    # Stack all quantized coefficients from all blocks
    quantized_coeffs_stack = np.concatenate(all_quantized_coeffs)

    # Clamp to 0-255 range and convert to uint8
    return np.clip(reconstructed_channel, 0, 255).astype(np.uint8), quantized_coeffs_stack


# ==============================================================================
# Color Image Wrapper Functions (Block-based Compression)
# ==============================================================================

def dct_compression_color(image_rgb, quality_scale=1.0):
    """
    Performs JPEG-like DCT compression on RGB image using block quantization.
    
    Args:
        image_rgb: RGB image (H x W x 3)
        quality_scale: Scaling factor for quantization matrix
                      > 1.0 = higher compression, lower quality
                      < 1.0 = lower compression, higher quality
                      
    Returns:
        reconstructed: Reconstructed RGB image
        all_coeffs: Concatenated quantized coefficients from all channels
    """
    BLOCK_SIZE = 8
    
    # Align image to 8x8 blocks
    H, W, C = image_rgb.shape
    H_new = H - (H % BLOCK_SIZE)
    W_new = W - (W % BLOCK_SIZE)
    image_aligned = image_rgb[:H_new, :W_new, :]
    
    # Create quantization matrix
    q_matrix = JPEG_QUANTIZATION_MATRIX * quality_scale
    
    dct_recon_channels = []
    dct_all_coeffs = []
    
    for i in range(3):
        recon_channel, coeffs = block_compression(
            image_aligned[:, :, i], BLOCK_SIZE, dct_2d, idct_2d, q_matrix
        )
        dct_recon_channels.append(recon_channel)
        dct_all_coeffs.append(coeffs)
    
    reconstructed = np.stack(dct_recon_channels, axis=-1)
    all_coeffs = np.concatenate(dct_all_coeffs)
    
    return reconstructed, all_coeffs


def dft_compression_color(image_rgb, q_scalar=50):
    """
    Performs block-based DFT compression on RGB image with quantization.
    
    Args:
        image_rgb: RGB image (H x W x 3)
        q_scalar: Scalar value for quantization matrix
                 Higher = higher compression, lower quality
                 
    Returns:
        reconstructed: Reconstructed RGB image
        all_coeffs: Concatenated quantized coefficients from all channels
    """
    BLOCK_SIZE = 8
    
    # Align image to 8x8 blocks
    H, W, C = image_rgb.shape
    H_new = H - (H % BLOCK_SIZE)
    W_new = W - (W % BLOCK_SIZE)
    image_aligned = image_rgb[:H_new, :W_new, :]
    
    # Create quantization matrix (uniform for DFT)
    q_matrix = np.full((BLOCK_SIZE, BLOCK_SIZE), q_scalar)
    
    dft_recon_channels = []
    dft_all_coeffs = []
    
    for i in range(3):
        recon_channel, coeffs = block_compression(
            image_aligned[:, :, i], BLOCK_SIZE, fft2, ifft2, q_matrix
        )
        dft_recon_channels.append(recon_channel)
        dft_all_coeffs.append(coeffs)
    
    reconstructed = np.stack(dft_recon_channels, axis=-1)
    all_coeffs = np.concatenate(dft_all_coeffs)
    
    return reconstructed, all_coeffs


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # 1. Load Image

    # --- OPTION 1: Load an external image (Modify 'input_image.jpg' to your file name) ---
    IMAGE_FILE_PATH = '/content/4.2.07.tiff'
    original_img_rgb = load_external_image(IMAGE_FILE_PATH)

    # Compression Level (Higher = More Compression = Lower Quality)
    # 90 means we discard the bottom 90% of coefficients (keeping top 10%)
    THRESHOLD = 90

    print(f"Running compression keeping only top {100-THRESHOLD}% of coefficients across 3 channels...\n")

    # --- DFT Process (Color) ---
    dft_recon, dft_coeffs = dft_compression_color(original_img_rgb, THRESHOLD)
    mse_dft, psnr_dft = calculate_metrics(original_img_rgb, dft_recon)
    cr_dft = get_compression_ratio(dft_coeffs, THRESHOLD)

    # --- DCT Process (Color) ---
    dct_recon, dct_coeffs = dct_compression_color(original_img_rgb, THRESHOLD)
    mse_dct, psnr_dct = calculate_metrics(original_img_rgb, dct_recon)
    cr_dct = get_compression_ratio(dct_coeffs, THRESHOLD)

    # --- Calculate Difference Maps (Error Maps) ---
    # Calculates the average absolute difference across RGB channels (axis=2)
    original_float = original_img_rgb.astype(np.float64)
    dft_diff_map = np.mean(np.abs(original_float - dft_recon.astype(np.float64)), axis=2)
    dct_diff_map = np.mean(np.abs(original_float - dct_recon.astype(np.float64)), axis=2)

    # --- Convert to Grayscale for Histograms and Console Output ---
    original_gray = to_grayscale(original_img_rgb)
    dft_gray = to_grayscale(dft_recon)
    dct_gray = to_grayscale(dct_recon)

    # --- Output Statistics ---
    print("--- DFT Results (Color) ---")
    print(f"MSE : {mse_dft:.2f}")
    print(f"PSNR: {psnr_dft:.2f} dB")
    print(f"CR  : {cr_dft:.2f}:1")
    print("-" * 20)
    print("--- DCT Results (Color) ---")
    print(f"MSE : {mse_dct:.2f}")
    print(f"PSNR: {psnr_dct:.2f} dB")
    print(f"CR  : {cr_dct:.2f}:1")

    # ==========================================
    # --- Visualization 1: Images ---
    # ==========================================
    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image (RGB)")
    plt.imshow(original_img_rgb)
    plt.axis('off')

    # DFT Compressed Image
    plt.subplot(1, 3, 2)
    plt.title(f"DFT Compressed\nMSE: {mse_dft:.2f} | PSNR: {psnr_dft:.2f}dB | CR: {cr_dft:.1f}:1")
    plt.imshow(dft_recon)
    plt.axis('off')

    # DCT Compressed Image
    plt.subplot(1, 3, 3)
    plt.title(f"DCT Compressed\nMSE: {mse_dct:.2f} | PSNR: {psnr_dct:.2f}dB | CR: {cr_dct:.1f}:1")
    plt.imshow(dct_recon)
    plt.axis('off')

    plt.tight_layout()


    # ==========================================
    # --- Visualization 2: Histograms ---
    # ==========================================
    plt.figure(figsize=(12, 5))

    # Original Histogram
    plt.subplot(1, 3, 1)
    plt.hist(original_gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.title("Original Histogram (Grayscale)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.xlim(0, 256)

    # DFT Histogram
    plt.subplot(1, 3, 2)
    plt.hist(dft_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title(f"DFT Histogram\nMSE: {mse_dft:.2f}")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.xlim(0, 256)

    # DCT Histogram
    plt.subplot(1, 3, 3)
    plt.hist(dct_gray.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title(f"DCT Histogram\nMSE: {mse_dct:.2f}")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.xlim(0, 256)

    plt.tight_layout()


    # ==========================================
    # --- Visualization 3: Difference Images (Error Maps) ---
    # ==========================================
    plt.figure(figsize=(10, 5))

    # DFT Difference Map
    plt.subplot(1, 2, 1)
    plt.title(f"DFT Difference Map (Avg Abs Error)\nMSE: {mse_dft:.2f}")
    # Using 'viridis' to highlight error intensity, setting vmax for good contrast.
    plt.imshow(dft_diff_map, cmap='viridis', vmin=0, vmax=30)
    plt.colorbar(label='Avg Absolute Error per Pixel (0-255)')
    plt.axis('off')

    # DCT Difference Map
    plt.subplot(1, 2, 2)
    plt.title(f"DCT Difference Map (Avg Abs Error)\nMSE: {mse_dct:.2f}")
    plt.imshow(dct_diff_map, cmap='viridis', vmin=0, vmax=30)
    plt.colorbar(label='Avg Absolute Error per Pixel (0-255)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()