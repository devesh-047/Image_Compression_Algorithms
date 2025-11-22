# Image Compression Dashboard

Interactive Streamlit dashboard for comparing image compression algorithms.

## Features

- **Multi-algorithm comparison**: Huffman, LZW, RLE (lossless) and DCT, DFT (lossy)
- **Multiple image upload**: Process multiple images simultaneously
- **Real-time metrics**: Compression ratio, PSNR, SSIM with lossless detection
- **Visual comparison**: Side-by-side image display
- **Trade-off analysis**: Interactive scatter plots (CR vs PSNR/SSIM)
- **Export capabilities**: CSV download, session save

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:
```bash
streamlit run app.py
```

Or from the project root:
```bash
streamlit run dashboard/app.py
```

## Workflow

1. **Upload Images**: Click "Browse files" in the sidebar (supports PNG, JPG, TIFF)
2. **Select Algorithms**: Check the algorithms you want to run
3. **Adjust Parameters**:
   - Huffman: Base (2-8)
   - DCT/DFT: Threshold percentage (50-100%)
4. **Run Compression**: Click "▶️ Run Compression"
5. **View Results**: Compare original and reconstructed images, metrics table
6. **Generate Plots**: Click "📊 Generate Plots" for trade-off analysis
7. **Export**: Download CSV or save session to JSON

## Metrics Explained

### Compression Ratio (CR)
- Formula: `original_size / compressed_size`
- Higher is better (more compression)
- Example: 8.00:1 means file is 8× smaller

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in decibels (dB)
- **∞ (inf)** = perfect reconstruction (lossless)
- **>40 dB** = excellent quality
- **30-40 dB** = good quality
- **<30 dB** = noticeable degradation

### SSIM (Structural Similarity Index)
- Range: 0 to 1
- **1.0** = perfect reconstruction (lossless)
- **>0.95** = excellent quality
- **0.8-0.95** = good quality
- **<0.8** = visible artifacts

## Lossless Detection

The dashboard automatically detects perfect reconstruction:
- When `original == reconstructed` (pixel-exact):
  - PSNR is set to **∞ (inf)**
  - SSIM is set to **1.0**
- This identifies true lossless algorithms (Huffman, LZW, RLE when successful)

## Algorithm Types

### Lossless (Perfect Reconstruction)
- **Huffman**: N-ary entropy coding (adjustable base)
- **LZW**: Dictionary-based compression
- **RLE**: Run-length encoding

### Lossy (Quality vs Size Trade-off)
- **DCT**: Discrete Cosine Transform (JPEG-like)
- **DFT**: Discrete Fourier Transform
- Both support threshold parameter (% of coefficients to keep)

## Output Structure

```
dashboard/
├── results/
│   ├── compressed/         # Compressed files (.pkl)
│   │   ├── Huffman/
│   │   ├── LZW/
│   │   ├── RLE/
│   │   ├── DCT/
│   │   └── DFT/
│   ├── reconstructed/      # Decompressed images
│   ├── metrics/            # Session JSON files
│   └── plots/              # Generated visualizations
```

## Technical Notes

- **Original algorithms**: Located in `image_compression_project/original_algorithms/`
- **Wrapper module**: `streamlit_components/runner.py` (safe calls without modification)
- **Image formats**: Converts all inputs to NumPy arrays (H×W×3 for RGB)
- **Compressed format**: Pickle (.pkl) for Python object serialization

## Troubleshooting

### Module not found errors
Make sure you run from the dashboard directory:
```bash
cd dashboard
streamlit run app.py
```

### scikit-image not installed
The app falls back to manual PSNR/SSIM calculation if scikit-image is unavailable.

### Large images slow processing
Consider resizing images or running fewer algorithms simultaneously.

## Example Session

1. Upload `kodim01.png` from samples/kodak
2. Select: Huffman, DCT
3. Set Huffman base=2, DCT threshold=90%
4. Run compression
5. View results:
   - Huffman: 8:1 CR, PSNR=∞, SSIM=1.0 (lossless)
   - DCT: 1.1:1 CR, PSNR=32 dB, SSIM=0.85 (lossy)
6. Generate trade-off plots
7. Download CSV with all metrics

## Future Enhancements

- ZIP export with all outputs
- Difference heatmaps (per-pixel error visualization)
- DICOM format support
- Batch processing mode
- Custom parameter presets
- Comparison with baseline algorithms

## Credits

Part of the Image Compression Algorithms research project.
