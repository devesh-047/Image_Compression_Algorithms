# Image_Compression_Algorithms
Implementation and Comparison of image compression algorithms

## 🚀 Interactive Dashboard

This project includes a **Streamlit dashboard** for interactive compression algorithm comparison!

### Quick Start

```powershell
# 1. Navigate to project root
cd Image_Compression_Algorithms

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dashboard dependencies (one-time)
pip install streamlit pandas

# 4. Launch dashboard
cd dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### Dashboard Features

- **📁 Multi-image upload** - PNG, JPG, TIFF formats supported
- **🔄 Real-time compression** - All 5 algorithms (Huffman, LZW, RLE, DCT, DFT)
- **📊 Metrics with lossless detection** - Automatic PSNR=∞, SSIM=1.0 for perfect reconstruction
- **🖼️ Visual comparison** - Side-by-side original vs compressed images
- **📈 Trade-off analysis** - Interactive plots (CR vs PSNR, CR vs SSIM)
- **💾 Export capabilities** - CSV download, JSON session save

### Usage Example

1. Upload images from `samples/kodak/` (e.g., `kodim01.png`)
2. Select algorithms: ✅ Huffman (lossless), ✅ DCT (lossy)
3. Adjust parameters: Huffman base=2, DCT threshold=90%
4. Click "▶️ Run Compression"
5. View results and metrics

**Expected Results** (kodim01.png):
- Huffman: 8:1 compression, PSNR=∞, SSIM=1.0 (lossless)
- DCT: 1.1:1 compression, ~32 dB PSNR, ~0.85 SSIM (lossy)

📖 **Full documentation**: See `dashboard/README.md` and `dashboard/QUICKSTART.md`

---

## 🎉 Project Status: Converted to Uniform Python Package

This repository has been **successfully converted** from Jupyter notebooks to a structured Python project while **preserving all original algorithm implementations verbatim**.

### 📁 New Structure

- **`image_compression_project/`** - Main Python package with:
  - `original_algorithms/` - Verbatim algorithm implementations from notebooks
  - `wrappers/` - Thin CLI wrappers for easy command-line access
  - `tests/` - Comprehensive verification and unit tests
  - Full documentation and setup files

- **`repo_backup_before_conversion/`** - Complete backup of original files
- **Original notebooks preserved** in `lossless/` directory
- **Documentation files**: `conversion_log.txt`, `results_check.txt`, `cleanup_log.txt`

### 🚀 Quick Start

```powershell
# Navigate to the package
cd image_compression_project

# Install dependencies
pip install -r requirements.txt

# Run verification tests
python tests/verify_basic.py

# See detailed documentation
cat README.md
```

### ✅ Verification Status

**All tests PASSED** ✓
- 6/6 basic algorithm tests passed
- Lossless algorithms: Bit-exact reconstruction verified
- Original code preserved without modifications
- Wrapper equivalence confirmed

### 📖 Documentation

- **[Project README](image_compression_project/README.md)** - Complete usage guide
- **[Conversion Log](conversion_log.txt)** - Detailed conversion process
- **[Verification Results](results_check.txt)** - Test results and quality metrics
- **[Cleanup Log](cleanup_log.txt)** - File preservation record

---

## Original README Below

