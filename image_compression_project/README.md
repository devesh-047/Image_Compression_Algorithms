# Image Compression Algorithms - Converted Python Project

This repository contains implementations of various image compression algorithms, converted from Jupyter notebooks into a uniform Python project structure while **preserving all original algorithm implementations verbatim**.

## 🚨 Important: Original Implementations Preserved

**All algorithm code has been extracted verbatim from the original Jupyter notebooks without any modifications to the core logic.** The files in `original_algorithms/` are exact copies of the algorithm code cells from the notebooks, ensuring complete preservation of the original implementations.

## 📁 Project Structure

```
Image_Compression_Algorithms/
├── image_compression_project/          # Main project package
│   ├── original_algorithms/            # Verbatim algorithm implementations
│   │   ├── huffman_original.py        # Huffman encoding (N-ary support)
│   │   ├── lzw_original.py            # LZW compression
│   │   ├── rle_original.py            # Run-Length Encoding
│   │   └── dft_dct_original.py        # DFT/DCT lossy compression
│   ├── wrappers/                       # Thin CLI wrappers
│   │   ├── run_huffman.py             # Huffman CLI wrapper
│   │   ├── run_lzw.py                 # LZW CLI wrapper
│   │   ├── run_rle.py                 # RLE CLI wrapper
│   │   └── run_dft_dct.py             # DFT/DCT CLI wrapper
│   ├── tests/                          # Verification and unit tests
│   │   ├── verify_equivalence.py      # Original vs wrapper equivalence tests
│   │   └── test_algorithms.py         # Comprehensive unit tests
│   ├── outputs/                        # Output directories
│   │   ├── compressed/                # Compressed files
│   │   └── decompressed/              # Decompressed files
│   ├── requirements.txt                # Project dependencies
│   └── setup.py                        # Package installation script
├── repo_backup_before_conversion/      # Complete backup of original files
│   ├── lossless/                      # Original notebooks and data
│   ├── lossy/                         # Original lossy algorithms
│   ├── samples/                       # Sample images
│   └── README.md                      # Original README
├── lossless/                          # Original lossless notebooks (preserved)
├── lossy/                             # Original lossy implementation (preserved)
├── samples/                           # Sample images for testing
├── conversion_log.txt                 # Detailed conversion log
├── results_check.txt                  # Verification results
└── README.md                          # This file
```

## 🔧 Installation

### Option 1: Install in Development Mode (Recommended)

```powershell
cd image_compression_project
pip install -e .
```

### Option 2: Install Dependencies Only

```powershell
cd image_compression_project
pip install -r requirements.txt
```

## 🚀 Usage

### Command-Line Interface (After Installation)

The project provides CLI commands for each algorithm:

#### Huffman Compression
```powershell
# Compress
compress-huffman compress input.png output.pkl --base 2

# Decompress
compress-huffman decompress output.pkl reconstructed.png
```

#### LZW Compression
```powershell
# Compress
compress-lzw compress input.png output.pkl --max-dict-size 4096

# Decompress
compress-lzw decompress output.pkl reconstructed.png
```

#### RLE Compression
```powershell
# Compress
compress-rle compress input.png output.pkl

# Decompress
compress-rle decompress output.pkl reconstructed.png
```

#### DFT/DCT Lossy Compression
```powershell
# Compress with DCT
compress-dft-dct compress input.png output.pkl --method dct --threshold 90

# Compress with DFT
compress-dft-dct compress input.png output.pkl --method dft --threshold 90

# Decompress
compress-dft-dct decompress output.pkl reconstructed.png
```

### Using Wrappers Directly

```powershell
cd image_compression_project

# Huffman
python wrappers\run_huffman.py compress ..\samples\kodak\kodim01.png outputs\compressed\huffman.pkl --base 2

# LZW
python wrappers\run_lzw.py compress ..\samples\kodak\kodim01.png outputs\compressed\lzw.pkl

# RLE
python wrappers\run_rle.py compress ..\samples\kodak\kodim01.png outputs\compressed\rle.pkl

# DCT
python wrappers\run_dft_dct.py compress ..\samples\kodak\kodim01.png outputs\compressed\dct.pkl --method dct --threshold 90
```

### Using Original Algorithms Directly in Python

```python
import sys
sys.path.append('image_compression_project')

import numpy as np
import matplotlib.image as mpimg
from original_algorithms import huffman_original, lzw_original, rle_original

# Load image
img = mpimg.imread('samples/kodak/kodim01.png')

# Huffman compression
encoded, codebook = huffman_original.huffman_encode(img, base=2)
decoded = huffman_original.huffman_decode(encoded, codebook, img.shape, img.dtype)

# LZW compression
img_uint8 = (img * 255).astype(np.uint8)
lzw_encoded = lzw_original.lzw_encode(img_uint8, max_dict_size=4096)
lzw_decoded = lzw_original.lzw_decode(lzw_encoded.copy(), img_uint8.shape)

# RLE compression
rle_encoded, shape, dtype = rle_original.rle_encode(img_uint8)
rle_decoded = rle_original.rle_decode(rle_encoded, shape, dtype)
```

## 🧪 Testing and Verification

### Run All Tests

```powershell
cd image_compression_project
python -m pytest tests/ -v
```

### Run Equivalence Verification

This verifies that wrapper implementations produce identical results to original algorithms:

```powershell
cd image_compression_project
python tests\verify_equivalence.py
```

**Important:** All tests must pass before any cleanup operations. If tests fail, a `failure_report.txt` will be generated in the project root.

### Test Coverage

```powershell
cd image_compression_project
python -m pytest tests/ --cov=original_algorithms --cov=wrappers --cov-report=html
```

## 📊 Algorithms Implemented

### Lossless Compression

1. **Huffman Coding** (`huffman_original.py`)
   - N-ary Huffman encoding (base 2 to 62)
   - Variable-length encoding based on symbol frequency
   - Bit-exact reconstruction guaranteed

2. **LZW (Lempel-Ziv-Welch)** (`lzw_original.py`)
   - Dictionary-based compression
   - Configurable dictionary size (512-4096)
   - Works best on images with repetitive patterns

3. **RLE (Run-Length Encoding)** (`rle_original.py`)
   - Encodes runs of identical values
   - Efficient for images with large uniform regions
   - Simple but effective for specific image types

### Lossy Compression

4. **DFT/DCT** (`dft_dct_original.py`)
   - Discrete Fourier Transform (DFT)
   - Discrete Cosine Transform (DCT)
   - Configurable threshold for quality/size trade-off
   - Metrics: MSE, PSNR

## 📈 Verification Results

All verification tests check for:

- **Lossless algorithms**: Bit-exact equivalence (`np.array_equal`)
- **Lossy algorithms**: Numerical metrics within tolerance (PSNR, MSE)
- **Compressed files**: Byte-wise hash comparison
- **Wrapper vs Original**: Identical behavior verification

See `results_check.txt` for detailed verification results.

## 📝 Conversion Log

The `conversion_log.txt` file contains:
- Complete list of files extracted from notebooks
- Mapping of notebook cells to Python functions
- Any shims or compatibility layers created
- Verification steps performed
- Commands to reproduce verification locally

## 🔄 Reproducing Verification

To verify the conversion locally:

```powershell
# 1. Install dependencies
cd image_compression_project
pip install -r requirements.txt

# 2. Run unit tests
python -m pytest tests/test_algorithms.py -v

# 3. Run equivalence verification
python tests/verify_equivalence.py

# 4. Check all tests passed
echo $LASTEXITCODE  # Should be 0
```

## 🛠️ Development

### Running Individual Algorithm Tests

```powershell
# Test Huffman only
python -m pytest tests/test_algorithms.py::TestHuffmanCompression -v

# Test LZW only
python -m pytest tests/test_algorithms.py::TestLZWCompression -v

# Test RLE only
python -m pytest tests/test_algorithms.py::TestRLECompression -v

# Test lossy algorithms
python -m pytest tests/test_algorithms.py::TestLossyCompression -v
```

### Adding New Tests

Tests are located in `image_compression_project/tests/`. Follow the existing patterns:
- Use `pytest` fixtures for shared data
- Test both edge cases and typical inputs
- Verify bit-exact reconstruction for lossless algorithms
- Use metric tolerances for lossy algorithms

## 🔍 Code Quality Guarantees

1. **Original Algorithm Preservation**: All algorithm code in `original_algorithms/` is verbatim from notebooks
2. **No Logic Modifications**: Zero changes to algorithm logic, variable names, or magic constants
3. **Wrapper Isolation**: Wrappers only add I/O handling and CLI, never modify algorithms
4. **Test Coverage**: Comprehensive tests ensure correctness and equivalence
5. **Version Control**: Complete backup in `repo_backup_before_conversion/`

## 📚 Documentation Files

- `conversion_log.txt`: Complete conversion process documentation
- `cleanup_log.txt`: Log of any cleanup operations performed
- `results_check.txt`: Detailed test and verification results
- `failure_report.txt`: Generated if any verification fails (should not exist if all tests pass)

## 🐛 Troubleshooting

### Import Errors

If you encounter import errors:
```powershell
# Ensure you're in the correct directory
cd image_compression_project

# Install in development mode
pip install -e .
```

### Test Failures

If tests fail:
1. Check `failure_report.txt` for details
2. Verify sample images are present in `samples/kodak/`
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check Python version (requires Python 3.7+)

### Image Path Issues

Update image paths in wrapper scripts if samples are in a different location:
```python
# In wrapper scripts, modify the sample image path
samples_dir = project_root / "samples" / "kodak"
```

## 📄 License

This project preserves the original licensing of the source repository.

## 👥 Contributing

When contributing:
1. **Never modify** files in `original_algorithms/` - they must remain verbatim
2. Wrappers can be enhanced with better CLI/error handling
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Update documentation accordingly

## 🙏 Acknowledgments

This project is a conversion of the original Jupyter notebook-based implementations into a structured Python project, with all original algorithm code preserved exactly as written.

---

**Last Updated**: November 22, 2025  
**Conversion Status**: ✅ Complete and Verified
