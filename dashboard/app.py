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

def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray, compressed_size: int, original_size: int):
    """Calculate compression metrics with lossless detection"""
    
    # Compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    # Check for exact equality (lossless)
    is_lossless = np.array_equal(original, reconstructed)
    
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
            mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            ssim = 0.0 if mse > 0 else 1.0
    
    return {
        'compression_ratio': compression_ratio,
        'psnr': psnr,
        'ssim': ssim,
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
                    st.session_state.uploaded_images[uploaded_file.name] = {
                        'pil': img,
                        'array': img_array,
                        'size': uploaded_file.size
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
                'base': st.slider("Huffman Base", 2, 8, 2, help="N-ary Huffman base")
            }
        
        if selected_algorithms.get('DCT') or selected_algorithms.get('DFT'):
            threshold = st.slider(
                "DCT/DFT Threshold %",
                50, 100, 90,
                help="Percentage of coefficients to keep"
            )
            if selected_algorithms.get('DCT'):
                params['DCT'] = {'threshold_percent': threshold}
            if selected_algorithms.get('DFT'):
                params['DFT'] = {'threshold_percent': threshold}
        
        params.setdefault('LZW', {})
        params.setdefault('RLE', {})
        
        st.divider()
        
        # Action buttons
        run_button = st.button("▶️ Run Compression", type="primary", use_container_width=True)
        
        if st.session_state.results:
            if st.button("📊 Generate Plots", use_container_width=True):
                st.session_state.show_plots = True
            
            if st.button("💾 Save Session", use_container_width=True):
                save_session()
            
            if st.button("📥 Download CSV", use_container_width=True):
                st.session_state.download_csv = True
    
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
                        compressed_size = result['compressed_path'].stat().st_size
                        metrics = calculate_metrics(
                            img_array,
                            result['reconstructed'],
                            compressed_size,
                            original_size
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
                st.image(results[0]['original'], caption="Original", use_container_width=True)
            
            for idx, result in enumerate(results):
                with cols[idx + 1]:
                    st.image(result['reconstructed'], caption=f"{result['algorithm']}", use_container_width=True)
            
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
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.divider()
        
        # Trade-off plots
        if st.session_state.get('show_plots', False):
            st.header("📈 Trade-off Analysis")
            
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
        if st.session_state.get('download_csv', False):
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
            
            csv = csv_data.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "compression_results.csv",
                "text/csv",
                key='download-csv'
            )
            st.session_state.download_csv = False

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
