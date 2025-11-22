"""
Setup script for Image Compression Algorithms project.
This package contains original algorithm implementations extracted from Jupyter notebooks.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="image-compression-algorithms",
    version="1.0.0",
    author="Repository Maintainer",
    description="Image compression algorithms (Huffman, LZW, RLE, DFT/DCT) with original implementations preserved",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devesh-047/Image_Compression_Algorithms",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "Pillow>=8.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "compress-huffman=image_compression_project.wrappers.run_huffman:main",
            "compress-lzw=image_compression_project.wrappers.run_lzw:main",
            "compress-rle=image_compression_project.wrappers.run_rle:main",
            "compress-dft-dct=image_compression_project.wrappers.run_dft_dct:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="image compression huffman lzw rle dct dft",
)
