#!/usr/bin/env python3
"""
Thin CLI wrapper for LZW compression algorithm.
This wrapper imports and calls the original implementation without modifying any algorithm logic.
"""
import argparse
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from original_algorithms.lzw_original import (
    lzw_encode,
    lzw_decode
)


def compress_image(input_path, output_path, max_dict_size=4096):
    """Compress image using original LZW implementation."""
    # Load image
    original_image_float = plt.imread(input_path)
    original_image_uint8 = (original_image_float * 255).astype(np.uint8)
    
    # Call original encode function
    encoded_codes = lzw_encode(original_image_uint8, max_dict_size=max_dict_size)
    
    # Save compressed data
    compressed_data = {
        'encoded': encoded_codes,
        'shape': original_image_uint8.shape,
        'max_dict_size': max_dict_size
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    # Calculate metrics
    orig_bits = original_image_uint8.size * 8
    encoded_bits = len(encoded_codes) * np.array(encoded_codes).itemsize * 8
    metadata_bits = len(original_image_uint8.shape) * 32
    total_bits = encoded_bits + metadata_bits
    compression_ratio = orig_bits / total_bits if total_bits > 0 else 0
    
    return {
        'compression_ratio': compression_ratio,
        'orig_bits': orig_bits,
        'encoded_bits': encoded_bits,
        'metadata_bits': metadata_bits,
        'total_bits': total_bits
    }


def decompress_image(input_path, output_path):
    """Decompress image using original LZW implementation."""
    # Load compressed data
    with open(input_path, 'rb') as f:
        compressed_data = pickle.load(f)
    
    encoded_codes = compressed_data['encoded']
    shape = compressed_data['shape']
    
    # Call original decode function
    decoded_image_uint8 = lzw_decode(encoded_codes, shape)
    
    # Save decompressed image
    plt.imsave(output_path, decoded_image_uint8)
    
    return decoded_image_uint8


def main():
    parser = argparse.ArgumentParser(description='LZW image compression wrapper')
    parser.add_argument('command', choices=['compress', 'decompress'], 
                       help='Operation to perform')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--max-dict-size', type=int, default=4096,
                       help='Maximum dictionary size for LZW (default: 4096)')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        metrics = compress_image(args.input, args.output, args.max_dict_size)
        print(f"Compression complete!")
        print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
        print(f"Original bits: {metrics['orig_bits']}")
        print(f"Compressed bits: {metrics['total_bits']}")
    elif args.command == 'decompress':
        decompress_image(args.input, args.output)
        print(f"Decompression complete!")


if __name__ == '__main__':
    main()
