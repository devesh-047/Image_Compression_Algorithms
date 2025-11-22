#!/usr/bin/env python3
"""
Thin CLI wrapper for Huffman compression algorithm.
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

from original_algorithms.huffman_original import (
    huffman_encode,
    huffman_decode,
    calculate_huffman_compression_ratio
)


def compress_image(input_path, output_path, base=2):
    """Compress image using original Huffman implementation."""
    # Load image
    img = mpimg.imread(input_path)
    if img.dtype in [np.float32, np.float64]:
        img = np.round(img, 4)
    
    # Call original encode function
    encoded, codebook = huffman_encode(img, base=base)
    
    # Save compressed data
    compressed_data = {
        'encoded': encoded,
        'codebook': codebook,
        'shape': img.shape,
        'dtype': str(img.dtype),
        'base': base
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    # Calculate and return metrics
    ratio, orig_bits, encoded_bits, dict_bits, total_bits, _, _ = calculate_huffman_compression_ratio(img, base)
    
    return {
        'compression_ratio': ratio,
        'orig_bits': orig_bits,
        'encoded_bits': encoded_bits,
        'dict_bits': dict_bits,
        'total_bits': total_bits
    }


def decompress_image(input_path, output_path):
    """Decompress image using original Huffman implementation."""
    # Load compressed data
    with open(input_path, 'rb') as f:
        compressed_data = pickle.load(f)
    
    encoded = compressed_data['encoded']
    codebook = compressed_data['codebook']
    shape = compressed_data['shape']
    dtype = np.dtype(compressed_data['dtype'])
    
    # Call original decode function
    decoded = huffman_decode(encoded, codebook, shape, dtype)
    
    # Save decompressed image
    if np.issubdtype(decoded.dtype, np.floating):
        mpimg.imsave(output_path, np.clip(decoded, 0, 1))
    else:
        mpimg.imsave(output_path, decoded)
    
    return decoded


def main():
    parser = argparse.ArgumentParser(description='Huffman image compression wrapper')
    parser.add_argument('command', choices=['compress', 'decompress'], 
                       help='Operation to perform')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--base', type=int, default=2,
                       help='N-ary base for Huffman encoding (default: 2)')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        metrics = compress_image(args.input, args.output, args.base)
        print(f"Compression complete!")
        print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
        print(f"Original bits: {metrics['orig_bits']}")
        print(f"Compressed bits: {metrics['total_bits']}")
    elif args.command == 'decompress':
        decompress_image(args.input, args.output)
        print(f"Decompression complete!")


if __name__ == '__main__':
    main()
