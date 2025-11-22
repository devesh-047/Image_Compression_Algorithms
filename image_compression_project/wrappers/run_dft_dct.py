#!/usr/bin/env python3
"""
Thin CLI wrapper for DFT/DCT lossy compression algorithms.
This wrapper imports and calls the original implementation without modifying any algorithm logic.
"""
import argparse
import sys
import os
import pickle
import numpy as np
import matplotlib.image as mpimg

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from original_algorithms.dft_dct_original import (
    dft_compression_color,
    dct_compression_color,
    calculate_metrics,
    load_external_image
)


def compress_image(input_path, output_path, method='dct', threshold=90):
    """Compress image using original DFT/DCT implementation."""
    # Load image using original loader
    original_img_rgb = load_external_image(input_path)
    
    # Call original compression function based on method
    if method == 'dft':
        compressed_img, coeffs = dft_compression_color(original_img_rgb, threshold)
    elif method == 'dct':
        compressed_img, coeffs = dct_compression_color(original_img_rgb, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save compressed data
    compressed_data = {
        'coeffs': coeffs,
        'method': method,
        'threshold': threshold,
        'shape': original_img_rgb.shape
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    # Calculate metrics
    mse, psnr = calculate_metrics(original_img_rgb, compressed_img)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'compressed_image': compressed_img
    }


def decompress_image(input_path, output_path):
    """
    Decompress image using original DFT/DCT implementation.
    Note: For lossy compression, decompression reconstructs from saved coefficients.
    """
    # Load compressed data
    with open(input_path, 'rb') as f:
        compressed_data = pickle.load(f)
    
    coeffs = compressed_data['coeffs']
    method = compressed_data['method']
    
    # Import reconstruction functions
    from original_algorithms.dft_dct_original import idct_2d
    
    # Reconstruct image from coefficients
    if method == 'dct':
        # Reconstruct each channel
        reconstructed_channels = []
        for i in range(3):
            channel_coeffs = coeffs[:, :, i]
            reconstructed_channel = idct_2d(channel_coeffs)
            reconstructed_channels.append(reconstructed_channel)
        reconstructed = np.stack(reconstructed_channels, axis=-1)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    elif method == 'dft':
        # For DFT, reconstruction is already done during compression
        # This is a limitation of the original implementation
        print("Warning: DFT decompression from coefficients requires additional implementation")
        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(coeffs), axes=(0, 1)))
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Save reconstructed image
    mpimg.imsave(output_path, reconstructed)
    
    return reconstructed


def main():
    parser = argparse.ArgumentParser(description='DFT/DCT lossy image compression wrapper')
    parser.add_argument('command', choices=['compress', 'decompress'], 
                       help='Operation to perform')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--method', choices=['dft', 'dct'], default='dct',
                       help='Compression method (default: dct)')
    parser.add_argument('--threshold', type=int, default=90,
                       help='Compression threshold percentage (default: 90)')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        result = compress_image(args.input, args.output, args.method, args.threshold)
        print(f"Compression complete using {args.method.upper()}!")
        print(f"MSE: {result['mse']:.2f}")
        print(f"PSNR: {result['psnr']:.2f} dB")
        
        # Also save the compressed image as PNG for visual inspection
        preview_path = output_path.replace('.pkl', '_preview.png')
        mpimg.imsave(preview_path, result['compressed_image'])
        print(f"Preview saved to: {preview_path}")
    elif args.command == 'decompress':
        decompress_image(args.input, args.output)
        print(f"Decompression complete!")


if __name__ == '__main__':
    main()
