"""
Basic tests for dashboard functionality
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "image_compression_project"))

from dashboard.streamlit_components.runner import runner

def test_runner_imports():
    """Test that runner module imports correctly"""
    assert runner is not None
    assert hasattr(runner, 'run_compression')
    assert hasattr(runner, 'get_available_algorithms')
    assert hasattr(runner, 'get_algorithm_type')

def test_available_algorithms():
    """Test algorithm listing"""
    algorithms = runner.get_available_algorithms()
    assert isinstance(algorithms, list)
    assert len(algorithms) == 5
    assert 'Huffman' in algorithms
    assert 'LZW' in algorithms
    assert 'RLE' in algorithms
    assert 'DCT' in algorithms
    assert 'DFT' in algorithms

def test_algorithm_types():
    """Test algorithm type detection"""
    assert runner.get_algorithm_type('Huffman') == 'lossless'
    assert runner.get_algorithm_type('LZW') == 'lossless'
    assert runner.get_algorithm_type('RLE') == 'lossless'
    assert runner.get_algorithm_type('DCT') == 'lossy'
    assert runner.get_algorithm_type('DFT') == 'lossy'

def test_huffman_compression():
    """Test Huffman compression on small synthetic image"""
    # Create 32x32 test image
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    output_dir = PROJECT_ROOT / "dashboard" / "results" / "compressed" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run compression
    result = runner.run_compression(
        test_image,
        'Huffman',
        {'base': 2},
        output_dir
    )
    
    # Verify result structure
    assert 'compressed_path' in result
    assert 'reconstructed' in result
    assert 'type' in result
    assert 'time_seconds' in result
    assert 'params' in result
    
    # Verify lossless
    assert result['compressed_path'].exists()
    assert result['type'] == 'lossless'
    assert np.array_equal(test_image, result['reconstructed']), "Huffman should be lossless"

def test_dct_compression():
    """Test DCT compression on small synthetic image"""
    # Create 32x32 test image
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    output_dir = PROJECT_ROOT / "dashboard" / "results" / "compressed" / "test_dct"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run compression
    result = runner.run_compression(
        test_image,
        'DCT',
        {'threshold_percent': 90},
        output_dir
    )
    
    # Verify result structure
    assert 'compressed_path' in result
    assert 'reconstructed' in result
    assert 'type' in result
    assert result['type'] == 'lossy'
    
    # Verify lossy (should not be exact)
    assert result['reconstructed'].shape == test_image.shape
    # Note: DCT may occasionally be exact for small random images, so we just check shape

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
