"""
Streamlit Dashboard for Image Compression Algorithms
Provides interactive UI for running and comparing compression methods
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
from datetime import datetime
import io

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "image_compression_project"))

from streamlit_components.runner import runner
from original_algorithms import huffman_original

def to_grayscale(image_rgb):
    """
    Converts an RGB image array (H x W x 3) to a grayscale array (H x W)
    using the standard luminosity method (for histogram display).
    From reference script: image_compression.py
    """
    # Ensure input is float64 for calculation and normalized to 0-1
    img_float = image_rgb.astype(np.float64) / 255.0

    # Check if already grayscale (2D)
    if len(img_float.shape) == 2:
        return (img_float * 255).astype(np.uint8)

    # Standard luminosity conversion for RGB
    # Grayscale = R*0.2989 + G*0.5870 + B*0.1140
    grayscale = np.dot(img_float[...,:3], [0.2989, 0.5870, 0.1140])
    return (grayscale * 255).astype(np.uint8)

# Page config
st.set_page_config(
    page_title="Image Compression Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}
if 'show_plots' not in st.session_state:
    st.session_state.show_plots = False
if 'download_csv' not in st.session_state:
    st.session_state.download_csv = False

def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray, compressed_size: int, original_size: int, 
                     coeffs=None, is_complex=False):
    """Calculate compression metrics with lossless detection
    
    For lossy algorithms (DCT/DFT), uses theoretical compression ratio based on quantized coefficients.
    For lossless algorithms, uses actual file size.
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        compressed_size: Compressed file size in bytes
        original_size: Original image size in bytes
        coeffs: Quantized coefficients array (for lossy algorithms)
        is_complex: Whether coefficients are complex (for DFT)
    """
    
    # Compression ratio calculation
    if coeffs is not None:
        # Lossy algorithm: use quantized coefficient count
        # CR = total_coefficients / non_zero_coefficients
        if is_complex:
            # For DFT: count real and imaginary parts separately
            total_complex_coeffs = coeffs.size
            total_scalar_elements = total_complex_coeffs * 2
            non_zero_real = np.count_nonzero(np.real(coeffs))
            non_zero_imag = np.count_nonzero(np.imag(coeffs))
            non_zero_scalar_count = non_zero_real + non_zero_imag
            compression_ratio = total_scalar_elements / non_zero_scalar_count if non_zero_scalar_count > 0 else 0
        else:
            # For DCT: count real coefficients
            total_elements = coeffs.size
            non_zero_count = np.count_nonzero(coeffs)
            compression_ratio = total_elements / non_zero_count if non_zero_count > 0 else 0
    else:
        # Lossless algorithm: use file size ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    # Check for exact equality (lossless)
    is_lossless = np.array_equal(original, reconstructed)
    
    # Calculate MSE (from reference script)
    original_float = original.astype(np.float64)
    reconstructed_float = reconstructed.astype(np.float64)
    mse = np.mean((original_float - reconstructed_float) ** 2)
    
    if is_lossless:
        psnr = float('inf')
        ssim = 1.0
    else:
        # Calculate PSNR
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
            
            # Calculate SSIM
            if len(original.shape) == 3:
                ssim = structural_similarity(original, reconstructed, channel_axis=2, data_range=255)
            else:
                ssim = structural_similarity(original, reconstructed, data_range=255)
        except Exception as e:
            # Fallback calculation
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            ssim = 0.0 if mse > 0 else 1.0
    
    return {
        'compression_ratio': compression_ratio,
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'is_lossless': is_lossless,
        'original_size': original_size,
        'compressed_size': compressed_size
    }

def main():
    st.title("🖼️ Image Compression Dashboard")
    st.markdown("**Compare lossless and lossy compression algorithms interactively**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=['png', 'jpg', 'jpeg', 'tiff'],
            accept_multiple_files=True,
            help="Upload one or more images to compress"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_images:
                    img = Image.open(uploaded_file)
                    img_array = np.array(img)
                    # Calculate raw image data size (uncompressed)
                    raw_size = img_array.nbytes  # height × width × channels × bytes_per_pixel
                    st.session_state.uploaded_images[uploaded_file.name] = {
                        'pil': img,
                        'array': img_array,
                        'size': raw_size
                    }
        
        st.divider()
        
        # Algorithm selection
        st.subheader("Select Algorithms")
        algorithms = runner.get_available_algorithms()
        
        selected_algorithms = {}
        for algo in algorithms:
            algo_type = runner.get_algorithm_type(algo)
            selected = st.checkbox(
                f"{algo} ({algo_type})",
                value=(algo == 'Huffman'),
                key=f"select_{algo}"
            )
            selected_algorithms[algo] = selected
        
        st.divider()
        
        # Parameters
        st.subheader("Parameters")
        
        params = {}
        if selected_algorithms.get('Huffman'):
            params['Huffman'] = {
                'base': st.slider("Huffman Base", 2, 62, 2, help="N-ary Huffman base")
            }
        
        if selected_algorithms.get('DCT'):
            quality_scale = st.slider(
                "DCT Quality Scale",
                0.5, 10.0, 5.0, 0.5,
                help="JPEG-like quantization scale: Higher = more compression, lower quality"
            )
            params['DCT'] = {'quality_scale': quality_scale}
        
        if selected_algorithms.get('DFT'):
            q_scalar = st.slider(
                "DFT Quantization Scalar",
                10, 200, 95, 5,
                help="Block quantization scalar: Higher = more compression, lower quality"
            )
            params['DFT'] = {'q_scalar': q_scalar}
        
        if selected_algorithms.get('LZW'):
            lzw_max_dict = st.slider(
                "LZW max dictionary size",
                min_value=256,
                max_value=65536,
                value=4096,
                step=256,
                help="Maximum dictionary size for LZW compression"
            )
            params['LZW'] = {'max_dict_size': lzw_max_dict}
        else:
            params.setdefault('LZW', {})
        
        params.setdefault('RLE', {})
        
        st.divider()
        
        # Action buttons
        run_button = st.button("▶️ Run Compression", type="primary", width='stretch')
        
        if st.session_state.results:
            st.divider()
            st.subheader("📊 Analysis & Export")
            
            if st.button("📊 Generate Trade-off Plots", width='stretch'):
                st.session_state.show_plots = True
                st.rerun()
            
            if st.button("📥 Download Results CSV", width='stretch'):
                st.session_state.download_csv = True
                st.rerun()
            
            if st.button("💾 Save Session", width='stretch'):
                save_session()
    
    # Main area
    if not st.session_state.uploaded_images:
        st.info("👈 Upload images in the sidebar to get started")
        st.markdown("""
        ### Features
        - **Lossless**: Huffman, LZW, RLE (perfect reconstruction)
        - **Lossy**: DCT, DFT (quality vs size trade-off)
        - **Metrics**: Compression ratio, PSNR, SSIM
        - **Visualization**: Side-by-side comparison, trade-off graphs
        """)
        return
    
    # Run compression
    if run_button:
        st.session_state.results = []
        
        with st.spinner("Running compression algorithms..."):
            for img_name, img_data in st.session_state.uploaded_images.items():
                img_array = img_data['array']
                original_size = img_data['size']
                
                for algo_name, is_selected in selected_algorithms.items():
                    if not is_selected:
                        continue
                    
                    try:
                        # Run algorithm
                        output_dir = PROJECT_ROOT / "dashboard" / "results" / "compressed" / algo_name / img_name.split('.')[0]
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        result = runner.run_compression(
                            img_array,
                            algo_name,
                            params.get(algo_name, {}),
                            output_dir
                        )
                        
                        # Calculate metrics
                        # Use the pickle file size as compressed size (includes all overhead)
                        compressed_size = result['compressed_path'].stat().st_size
                        
                        # For lossy algorithms, pass coeffs for theoretical ratio calculation
                        coeffs = result.get('coeffs', None)
                        is_complex = result.get('is_complex', False)
                        
                        metrics = calculate_metrics(
                            img_array,
                            result['reconstructed'],
                            compressed_size,
                            original_size,
                            coeffs=coeffs,
                            is_complex=is_complex
                        )
                        
                        # Store result
                        st.session_state.results.append({
                            'image_name': img_name,
                            'algorithm': algo_name,
                            'type': result['type'],
                            'original': img_array,
                            'reconstructed': result['reconstructed'],
                            'time_seconds': result['time_seconds'],
                            **metrics,
                            **result['params']
                        })
                        
                    except Exception as e:
                        st.error(f"Error running {algo_name} on {img_name}: {str(e)}")
        
        st.success(f"✅ Completed {len(st.session_state.results)} compression runs!")
    
    # Display results
    if st.session_state.results:
        st.header("📊 Results")
        
        # Group by image
        results_by_image = {}
        for result in st.session_state.results:
            img_name = result['image_name']
            if img_name not in results_by_image:
                results_by_image[img_name] = []
            results_by_image[img_name].append(result)
        
        # Display each image's results
        for img_name, results in results_by_image.items():
            st.subheader(f"Image: {img_name}")
            
            # Show images
            cols = st.columns(len(results) + 1)
            
            with cols[0]:
                st.image(results[0]['original'], caption="Original", width='stretch')
            
            for idx, result in enumerate(results):
                with cols[idx + 1]:
                    st.image(result['reconstructed'], caption=f"{result['algorithm']}", width='stretch')
            
            # Add histogram and difference map visualizations
            with st.expander("📊 View Histograms & Difference Maps"):
                # Histograms
                st.subheader("Pixel Intensity Histograms (Grayscale)")
                
                # Create histogram figure
                num_algos = len(results)
                fig_hist, axes_hist = plt.subplots(1, num_algos + 1, figsize=(4 * (num_algos + 1), 4))
                if num_algos == 0:
                    axes_hist = [axes_hist]
                
                # Original histogram
                original_gray = to_grayscale(results[0]['original'])
                axes_hist[0].hist(original_gray.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
                axes_hist[0].set_title("Original Histogram")
                axes_hist[0].set_xlabel("Pixel Intensity (0-255)")
                axes_hist[0].set_ylabel("Frequency")
                axes_hist[0].set_xlim(0, 256)
                
                # Reconstructed histograms
                for idx, result in enumerate(results):
                    recon_gray = to_grayscale(result['reconstructed'])
                    axes_hist[idx + 1].hist(recon_gray.ravel(), bins=256, range=[0, 256], 
                                           color='blue' if idx == 0 else 'red', alpha=0.7)
                    axes_hist[idx + 1].set_title(f"{result['algorithm']} Histogram")
                    axes_hist[idx + 1].set_xlabel("Pixel Intensity (0-255)")
                    axes_hist[idx + 1].set_ylabel("Frequency")
                    axes_hist[idx + 1].set_xlim(0, 256)
                
                plt.tight_layout()
                st.pyplot(fig_hist)
                plt.close()
                
                # Difference Maps (Error Maps)
                st.subheader("Difference Maps (Average Absolute Error)")
                
                num_algos_diff = len([r for r in results if r['type'] == 'lossy'])
                if num_algos_diff > 0:
                    fig_diff, axes_diff = plt.subplots(1, num_algos_diff, figsize=(6 * num_algos_diff, 5))
                    if num_algos_diff == 1:
                        axes_diff = [axes_diff]
                    
                    diff_idx = 0
                    for result in results:
                        if result['type'] == 'lossy':
                            # Calculate difference map (avg across RGB channels)
                            original_float = results[0]['original'].astype(np.float64)
                            recon_float = result['reconstructed'].astype(np.float64)
                            diff_map = np.mean(np.abs(original_float - recon_float), axis=2)
                            
                            # Display difference map
                            im = axes_diff[diff_idx].imshow(diff_map, cmap='viridis', vmin=0, vmax=30)
                            axes_diff[diff_idx].set_title(f"{result['algorithm']} Difference Map\nMSE: {result.get('mse', 0):.2f}")
                            axes_diff[diff_idx].axis('off')
                            plt.colorbar(im, ax=axes_diff[diff_idx], label='Avg Absolute Error (0-255)')
                            diff_idx += 1
                    
                    plt.tight_layout()
                    st.pyplot(fig_diff)
                    plt.close()
                else:
                    st.info("Difference maps are only available for lossy compression algorithms (DCT, DFT)")
            
            # Metrics table
            metrics_data = []
            for result in results:
                psnr_str = "∞ (inf)" if result['psnr'] == float('inf') else f"{result['psnr']:.2f}"
                metrics_data.append({
                    'Algorithm': result['algorithm'],
                    'Type': result['type'],
                    'CR': f"{result['compression_ratio']:.2f}:1",
                    'PSNR (dB)': psnr_str,
                    'SSIM': f"{result['ssim']:.4f}",
                    'Time (s)': f"{result['time_seconds']:.2f}",
                    'Size (KB)': f"{result['compressed_size']/1024:.1f}"
                })
            
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            st.divider()
        
        # Trade-off plots
        if st.session_state.get('show_plots', False):
            col_header1, col_header2 = st.columns([3, 1])
            with col_header1:
                st.header("📈 Trade-off Analysis")
            with col_header2:
                if st.button("❌ Hide Plots"):
                    st.session_state.show_plots = False
                    st.rerun()
            
            # Prepare data
            plot_data = pd.DataFrame([{
                'Algorithm': r['algorithm'],
                'CR': r['compression_ratio'],
                'PSNR': r['psnr'] if r['psnr'] != float('inf') else 100,  # Cap for plotting
                'SSIM': r['ssim']
            } for r in st.session_state.results])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                for algo in plot_data['Algorithm'].unique():
                    data = plot_data[plot_data['Algorithm'] == algo]
                    ax.scatter(data['CR'], data['PSNR'], label=algo, s=100, alpha=0.7)
                ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
                ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
                ax.set_title('Compression Ratio vs PSNR', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                for algo in plot_data['Algorithm'].unique():
                    data = plot_data[plot_data['Algorithm'] == algo]
                    ax.scatter(data['CR'], data['SSIM'], label=algo, s=100, alpha=0.7)
                ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
                ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
                ax.set_title('Compression Ratio vs SSIM', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # CSV download
        if st.session_state.get('download_csv', False) and st.session_state.results:
            st.divider()
            col_csv1, col_csv2 = st.columns([3, 1])
            with col_csv1:
                st.subheader("📥 Download Results")
            with col_csv2:
                if st.button("❌ Hide Download"):
                    st.session_state.download_csv = False
                    st.rerun()
            
            csv_data = pd.DataFrame([{
                'image_name': r['image_name'],
                'algorithm': r['algorithm'],
                'type': r['type'],
                'compression_ratio': r['compression_ratio'],
                'psnr': r['psnr'],
                'ssim': r['ssim'],
                'time_seconds': r['time_seconds'],
                'original_size_bytes': r['original_size'],
                'compressed_size_bytes': r['compressed_size']
            } for r in st.session_state.results])
            
            # Show preview
            st.dataframe(csv_data, width='stretch', hide_index=True)
            
            csv = csv_data.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "compression_results.csv",
                "text/csv",
                key='download-csv',
                width='stretch'
            )

def save_session():
    """Save current session to JSON"""
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(st.session_state.uploaded_images),
        'num_results': len(st.session_state.results),
        'results': [{
            'image_name': r['image_name'],
            'algorithm': r['algorithm'],
            'compression_ratio': r['compression_ratio'],
            'psnr': r['psnr'] if r['psnr'] != float('inf') else 'inf',
            'ssim': r['ssim']
        } for r in st.session_state.results]
    }
    
    output_file = PROJECT_ROOT / "dashboard" / "results" / "metrics" / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    st.success(f"✅ Session saved to {output_file.name}")

if __name__ == "__main__":
    main()
