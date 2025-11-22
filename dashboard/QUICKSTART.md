# 🚀 Quick Start Guide: Streamlit Dashboard

## ✅ Verification Complete

**Tests Passed**: 5/5 ✓
```
test_runner_imports PASSED
test_available_algorithms PASSED
test_algorithm_types PASSED
test_huffman_compression PASSED
test_dct_compression PASSED
```

**Status**: Dashboard is ready to use!

---

## 🎯 Launch Dashboard (3 Steps)

### Step 1: Install Dependencies (if not already)
```powershell
cd "c:\Users\DEVESH PALO\projects\Image_Compression_Algorithms\dashboard"
pip install -r requirements.txt
```

### Step 2: Launch Streamlit
```powershell
streamlit run app.py
```

### Step 3: Use the Dashboard
- Browser will open automatically at `http://localhost:8501`
- If not, manually open: http://localhost:8501

---

## 🖼️ Quick Demo Workflow

### 1. Upload Images
- Click **"Browse files"** in the sidebar
- Navigate to: `c:\Users\DEVESH PALO\projects\Image_Compression_Algorithms\samples\kodak`
- Select one or more images (e.g., `kodim01.png`, `kodim02.png`)
- Click **Open**

### 2. Select Algorithms
Check the boxes for algorithms you want to test:
- ☑️ **Huffman** (lossless) - Recommended
- ☐ LZW (lossless) - Not recommended for photos
- ☐ RLE (lossless) - Not recommended for photos
- ☑️ **DCT** (lossy) - Recommended
- ☐ DFT (lossy)

### 3. Adjust Parameters
- **Huffman Base**: Keep at 2 (default)
- **DCT/DFT Threshold %**: Try 90 (higher = better quality, lower compression)

### 4. Run Compression
- Click **"▶️ Run Compression"** (blue button)
- Wait for progress spinner (~2-5 seconds per image)
- See success message: "✅ Completed X compression runs!"

### 5. View Results
**Image Comparison**: Side-by-side original and reconstructed images

**Metrics Table** (example for kodim01.png):
| Algorithm | Type     | CR    | PSNR (dB) | SSIM   | Time (s) | Size (KB) |
|-----------|----------|-------|-----------|--------|----------|-----------|
| Huffman   | lossless | 8.00  | ∞ (inf)   | 1.0000 | 2.15     | 115.3     |
| DCT       | lossy    | 1.11  | 32.45     | 0.8523 | 0.98     | 831.2     |

### 6. Generate Trade-off Plots
- Click **"📊 Generate Plots"**
- View two scatter plots:
  - **Compression Ratio vs PSNR**: Higher CR + higher PSNR = better
  - **Compression Ratio vs SSIM**: Higher CR + higher SSIM = better

### 7. Export Results
- Click **"📥 Download CSV"** to get `compression_results.csv`
- Click **"💾 Save Session"** to save JSON to `dashboard/results/metrics/`

---

## 🎨 Understanding the Results

### Lossless Algorithms (Perfect Reconstruction)
**Huffman** is the star performer:
- **PSNR**: ∞ (inf) = pixel-perfect reconstruction
- **SSIM**: 1.0000 = perfect structural similarity
- **CR**: ~8.00:1 on natural images
- **Use case**: Medical imaging, archival, when quality is critical

**LZW/RLE** typically fail on photos:
- **CR**: <1:1 (file expansion!)
- **Reason**: Natural images have high entropy, no repetition patterns
- **Use case**: Binary images, logos, text documents

### Lossy Algorithms (Quality vs Size Trade-off)
**DCT/DFT** offer controllable compression:
- **PSNR**: 30-35 dB = good quality
- **SSIM**: 0.8-0.9 = minor perceptual differences
- **CR**: 1.1-1.5:1 depending on threshold
- **Use case**: Web images, bandwidth-limited applications

### Key Metrics Explained

**Compression Ratio (CR)**
- Formula: `original_size ÷ compressed_size`
- Example: 8.00:1 means file is 8× smaller
- Higher = more compression

**PSNR (Peak Signal-to-Noise Ratio)**
- Measured in decibels (dB)
- **∞ (inf)** = lossless (perfect)
- **>40 dB** = excellent quality
- **30-40 dB** = good quality
- **<30 dB** = visible artifacts

**SSIM (Structural Similarity Index)**
- Range: 0 to 1
- **1.0** = lossless (perfect)
- **>0.95** = excellent
- **0.8-0.95** = good
- **<0.8** = noticeable degradation

---

## 💡 Tips & Recommendations

### For Best Results
1. **Use Huffman for lossless**: Best CR with perfect quality
2. **Use DCT for lossy**: Good balance of quality and size
3. **Skip LZW/RLE on photos**: They cause file expansion
4. **Adjust DCT threshold**: 
   - 95% = high quality, low compression
   - 85% = lower quality, better compression
5. **Test on multiple images**: Results vary by content

### Performance Tips
- **Small images** (<512×512): Very fast (<1s per algorithm)
- **Large images** (>2048×2048): Slower (2-5s per algorithm)
- **Multiple images**: Run in batches, download CSV for analysis

### Common Use Cases

**Medical Imaging** (lossless required):
- Select: ✅ Huffman only
- Expected: 8:1 CR, PSNR=∞, SSIM=1.0

**Web Gallery** (lossy acceptable):
- Select: ✅ DCT
- Threshold: 90%
- Expected: 1.1:1 CR, 32 dB PSNR, 0.85 SSIM

**Algorithm Comparison** (research):
- Select: ✅ All algorithms
- Compare trade-off plots
- Export CSV for further analysis

---

## 📊 Example Session Output

After running Huffman + DCT on 3 Kodak images:

**Images Processed**: kodim01.png, kodim02.png, kodim03.png
**Algorithms Run**: Huffman, DCT
**Total Runs**: 6 (3 images × 2 algorithms)

**Average Results**:
| Metric | Huffman | DCT (90%) |
|--------|---------|-----------|
| CR     | 8.12:1  | 1.11:1    |
| PSNR   | ∞       | 32.8 dB   |
| SSIM   | 1.0     | 0.86      |
| Time   | 2.2s    | 1.0s      |

**Files Generated**:
- `dashboard/results/compressed/Huffman/kodim01/compressed_image.pkl`
- `dashboard/results/compressed/DCT/kodim01/compressed_image.pkl`
- (... 4 more compressed files)
- `dashboard/results/metrics/session_20250130_123456.json`
- `compression_results.csv` (downloaded)

---

## 🐛 Troubleshooting

### Dashboard Won't Launch
**Error**: `streamlit: command not found`
**Fix**: `pip install streamlit>=1.28.0`

**Error**: `No module named 'streamlit'`
**Fix**: Make sure virtual environment is activated:
```powershell
cd "c:\Users\DEVESH PALO\projects\Image_Compression_Algorithms"
.\venv\Scripts\Activate.ps1
cd dashboard
streamlit run app.py
```

### Import Errors
**Error**: `ModuleNotFoundError: No module named 'streamlit_components'`
**Fix**: Run from correct directory:
```powershell
cd "c:\Users\DEVESH PALO\projects\Image_Compression_Algorithms\dashboard"
streamlit run app.py
```

### Slow Performance
**Symptom**: Compression takes >10 seconds
**Cause**: Very large images or old hardware
**Fix**: 
1. Resize images before upload
2. Run fewer algorithms simultaneously
3. Test on smaller images first (e.g., 512×512)

### Metrics Show Inf/NaN
**Symptom**: PSNR shows "nan" or unexpected values
**Cause**: Algorithm failed or image corrupted
**Fix**: Check console for error messages, try different image

---

## 🎓 Advanced Usage

### Comparing Datasets
1. Upload all 24 Kodak images
2. Select Huffman + DCT
3. Run compression (takes ~2-3 minutes)
4. Download CSV
5. Analyze in Excel/Python:
   - Average CR per algorithm
   - Best/worst performing images
   - Quality distribution

### Custom Parameters
**Huffman Base** (2-8):
- Base 2: Standard binary Huffman
- Base 4: Quaternary Huffman (slower, similar CR)
- Base 8: Octal Huffman (much slower, minimal CR gain)

**DCT/DFT Threshold** (50-100%):
- 100%: Keep all coefficients (minimal compression)
- 90%: Good balance (recommended)
- 70%: Aggressive compression (visible artifacts)
- 50%: Maximum compression (poor quality)

### Exporting for Research
1. Run all algorithms on dataset
2. Download CSV: `compression_results.csv`
3. Generate plots: Click "📊 Generate Plots"
4. Save session: Click "💾 Save Session"
5. Analyze:
   - Import CSV to pandas: `pd.read_csv('compression_results.csv')`
   - Group by algorithm: `df.groupby('algorithm').mean()`
   - Plot custom graphs using matplotlib

---

## 📂 Output Files Reference

### Compressed Files
```
dashboard/results/compressed/
├── Huffman/
│   └── kodim01/
│       └── compressed_image.pkl     # Huffman codes + tree
├── DCT/
│   └── kodim01/
│       └── compressed_image.pkl     # DCT coefficients
└── ...
```

### Session Files
```
dashboard/results/metrics/
└── session_20250130_123456.json     # Timestamp, metrics, parameters
```

### CSV Export
```
compression_results.csv
Columns: image_name, algorithm, type, compression_ratio, psnr, ssim, 
         time_seconds, original_size_bytes, compressed_size_bytes
```

---

## 🔗 Related Resources

**Full Documentation**: See `dashboard/README.md`
**Implementation Details**: See `dashboard/IMPLEMENTATION_SUMMARY.md`
**Original Algorithms**: See `image_compression_project/original_algorithms/`
**Research Report**: See `results/final_report.md`
**Experimental Data**: See `results/combined_results.csv` (120 experiments)

---

## 🎉 You're Ready!

**Command to launch**:
```powershell
cd "c:\Users\DEVESH PALO\projects\Image_Compression_Algorithms\dashboard"
streamlit run app.py
```

**First test**:
1. Upload `samples/kodak/kodim01.png`
2. Select: ✅ Huffman, ✅ DCT
3. Click "▶️ Run Compression"
4. Expect: Huffman 8:1 CR (PSNR=∞), DCT 1.1:1 CR (~32 dB)

**Questions?** Check `dashboard/README.md` or run tests: `pytest dashboard/tests/ -v`

---

**Have fun exploring image compression! 🚀**
