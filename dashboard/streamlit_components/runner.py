"""
Runner Module: Safe wrapper for calling original compression algorithms
Preserves original algorithm implementations while providing Streamlit-friendly interface
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import time

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "image_compression_project"))

# Import original algorithms (NEVER MODIFIED)
from original_algorithms import huffman_original, lzw_original, rle_original, dft_dct_original


class AlgorithmRunner:
    """Wrapper that calls original algorithms safely without modification"""
    
    def __init__(self):
        self.algorithms = {
            'Huffman': self._run_huffman,
            'LZW': self._run_lzw,
            'RLE': self._run_rle,
            'DCT': self._run_dct,
            'DFT': self._run_dft
        }
        
        self.algorithm_types = {
            'Huffman': 'lossless',
            'LZW': 'lossless',
            'RLE': 'lossless',
            'DCT': 'lossy',
            'DFT': 'lossy'
        }
    
    def get_available_algorithms(self):
        """Return list of available algorithms"""
        return list(self.algorithms.keys())
    
    def get_algorithm_type(self, algorithm_name):
        """Return 'lossless' or 'lossy'"""
        return self.algorithm_types.get(algorithm_name, 'unknown')
    
    def run_compression(self, image: np.ndarray, algorithm_name: str, 
                       params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """
        Run compression algorithm on image
        
        Args:
            image: numpy array (H, W) or (H, W, 3)
            algorithm_name: name of algorithm
            params: algorithm-specific parameters
            output_dir: directory to save compressed files
            
        Returns:
            dict with compressed_path, reconstructed_image, metadata
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Call algorithm-specific runner
        result = self.algorithms[algorithm_name](image, params, output_dir)
        
        elapsed = time.time() - start_time
        result['time_seconds'] = elapsed
        result['algorithm'] = algorithm_name
        result['type'] = self.algorithm_types[algorithm_name]
        
        return result
    
    def _run_huffman(self, img: np.ndarray, params: Dict, output_dir: Path) -> Dict:
        """Run Huffman compression (lossless)"""
        base = params.get('base', 2)
        
        if len(img.shape) == 3:
            # Color image - process per channel
            compressed_channels = []
            codebooks = []
            
            for i in range(3):
                channel = img[:, :, i]
                encoded, codebook = huffman_original.huffman_encode(channel, base=base)
                compressed_channels.append(encoded)
                codebooks.append(codebook)
            
            # Save compressed data
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'channels': compressed_channels,
                    'codebooks': codebooks,
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }, f)
            
            # Decompress
            reconstructed_channels = []
            for encoded, codebook in zip(compressed_channels, codebooks):
                channel_shape = (img.shape[0], img.shape[1])
                decoded = huffman_original.huffman_decode(encoded, codebook, channel_shape, img.dtype)
                reconstructed_channels.append(decoded)
            
            reconstructed = np.stack(reconstructed_channels, axis=2)
            
        else:
            # Grayscale
            encoded, codebook = huffman_original.huffman_encode(img, base=base)
            
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'encoded': encoded,
                    'codebook': codebook,
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }, f)
            
            reconstructed = huffman_original.huffman_decode(encoded, codebook, img.shape, img.dtype)
        
        return {
            'compressed_path': compressed_path,
            'reconstructed': reconstructed,
            'params': {'base': base}
        }
    
    def _run_lzw(self, img: np.ndarray, params: Dict, output_dir: Path) -> Dict:
        """Run LZW compression (lossless)"""
        if len(img.shape) == 3:
            # Color image
            compressed_channels = []
            
            for i in range(3):
                channel = img[:, :, i]
                encoded = lzw_original.lzw_encode(channel.flatten())
                compressed_channels.append(encoded)
            
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'channels': compressed_channels,
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }, f)
            
            # Decompress
            reconstructed_channels = []
            for encoded in compressed_channels:
                channel_shape = (img.shape[0], img.shape[1])
                decoded = lzw_original.lzw_decode(encoded, channel_shape)
                reconstructed_channels.append(decoded)
            
            reconstructed = np.stack(reconstructed_channels, axis=2)
            
        else:
            # Grayscale
            encoded = lzw_original.lzw_encode(img.flatten())
            
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'encoded': encoded,
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }, f)
            
            reconstructed = lzw_original.lzw_decode(encoded, img.shape)
        
        return {
            'compressed_path': compressed_path,
            'reconstructed': reconstructed,
            'params': {}
        }
    
    def _run_rle(self, img: np.ndarray, params: Dict, output_dir: Path) -> Dict:
        """Run RLE compression (lossless)"""
        if len(img.shape) == 3:
            # Color image
            compressed_channels = []
            
            for i in range(3):
                channel = img[:, :, i]
                encoded, shape, dtype = rle_original.rle_encode(channel)
                compressed_channels.append((encoded, shape, dtype))
            
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'channels': compressed_channels,
                    'original_shape': img.shape
                }, f)
            
            # Decompress
            reconstructed_channels = []
            for encoded, shape, dtype in compressed_channels:
                decoded = rle_original.rle_decode(encoded, shape, dtype)
                reconstructed_channels.append(decoded)
            
            reconstructed = np.stack(reconstructed_channels, axis=2)
            
        else:
            # Grayscale
            encoded, shape, dtype = rle_original.rle_encode(img)
            
            compressed_path = output_dir / "compressed.pkl"
            with open(compressed_path, 'wb') as f:
                pickle.dump({
                    'encoded': encoded,
                    'shape': shape,
                    'dtype': dtype
                }, f)
            
            reconstructed = rle_original.rle_decode(encoded, shape, dtype)
        
        return {
            'compressed_path': compressed_path,
            'reconstructed': reconstructed,
            'params': {}
        }
    
    def _run_dct(self, img: np.ndarray, params: Dict, output_dir: Path) -> Dict:
        """Run DCT compression (lossy)"""
        threshold_percent = params.get('threshold_percent', 90)
        
        if len(img.shape) == 3:
            compressed, coeffs = dft_dct_original.dct_compression_color(img, threshold_percent)
        else:
            # Convert grayscale to 3-channel for algorithm
            img_3d = np.stack([img, img, img], axis=2)
            compressed, coeffs = dft_dct_original.dct_compression_color(img_3d, threshold_percent)
            compressed = compressed[:, :, 0]
        
        # Save compressed data
        compressed_path = output_dir / "compressed.pkl"
        with open(compressed_path, 'wb') as f:
            pickle.dump({
                'compressed': compressed,
                'coeffs': coeffs,
                'threshold_percent': threshold_percent,
                'original_shape': img.shape
            }, f)
        
        return {
            'compressed_path': compressed_path,
            'reconstructed': compressed,
            'params': {'threshold_percent': threshold_percent}
        }
    
    def _run_dft(self, img: np.ndarray, params: Dict, output_dir: Path) -> Dict:
        """Run DFT compression (lossy)"""
        threshold_percent = params.get('threshold_percent', 90)
        
        if len(img.shape) == 3:
            compressed, coeffs = dft_dct_original.dft_compression_color(img, threshold_percent)
        else:
            # Convert grayscale to 3-channel for algorithm
            img_3d = np.stack([img, img, img], axis=2)
            compressed, coeffs = dft_dct_original.dft_compression_color(img_3d, threshold_percent)
            compressed = compressed[:, :, 0]
        
        # Save compressed data
        compressed_path = output_dir / "compressed.pkl"
        with open(compressed_path, 'wb') as f:
            pickle.dump({
                'compressed': compressed,
                'coeffs': coeffs,
                'threshold_percent': threshold_percent,
                'original_shape': img.shape
            }, f)
        
        return {
            'compressed_path': compressed_path,
            'reconstructed': compressed,
            'params': {'threshold_percent': threshold_percent}
        }


# Global instance
runner = AlgorithmRunner()
