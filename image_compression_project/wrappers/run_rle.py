#!/usr/bin/env python3
"""
Thin CLI wrapper for RLE compression algorithm.
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

from original_algorithms.rle_original import (
    rle_encode,
    rle_decode,
    calculate_rle_compression_ratio
)


def compress_image(input_path, output_path):
    """Compress image using original RLE implementation."""
    # Load image
    img = mpimg.imread(input_path)
    
    # Convert float images to uint8 for better compression
    if img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)
    
    # Call original encode function
    encoded, shape, dtype = rle_encode(img)
    
    # Save compressed data
    compressed_data = {
        'encoded': encoded,
        'shape': shape,
        'dtype': str(dtype)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    # Calculate metrics
    ratio, orig_bits, encoded_bits, metadata_bits, total_bits, _, _, _ = calculate_rle_compression_ratio(img)
    
    return {
        'compression_ratio': ratio,
        'orig_bits': orig_bits,
        'encoded_bits': encoded_bits,
        'metadata_bits': metadata_bits,
        'total_bits': total_bits
    }


def decompress_image(input_path, output_path):
    """Decompress image using original RLE implementation."""
    # Load compressed data
    with open(input_path, 'rb') as f:
        compressed_data = pickle.load(f)
    
    encoded = compressed_data['encoded']
    shape = compressed_data['shape']
    dtype = np.dtype(compressed_data['dtype'])
    
    # Call original decode function
    decoded = rle_decode(encoded, shape, dtype)
    
    # Save decompressed image
    if len(decoded.shape) == 3:  # RGB
        mpimg.imsave(output_path, decoded)
    else:  # Grayscale
        mpimg.imsave(output_path, decoded, cmap='gray')
    
    return decoded


def main():
    parser = argparse.ArgumentParser(description='RLE image compression wrapper')
    parser.add_argument('command', choices=['compress', 'decompress'], 
                       help='Operation to perform')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        metrics = compress_image(args.input, args.output)
        print(f"Compression complete!")
        print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
        print(f"Original bits: {metrics['orig_bits']}")
        print(f"Compressed bits: {metrics['total_bits']}")
    elif args.command == 'decompress':
        decompress_image(args.input, args.output)
        print(f"Decompression complete!")


if __name__ == '__main__':
    main()
