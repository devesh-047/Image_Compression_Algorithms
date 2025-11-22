# Launch Script for Image Compression Dashboard
# Run this script to start the Streamlit dashboard

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Image Compression Dashboard" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (Test-Path "app.py") {
    Write-Host "✓ Found app.py" -ForegroundColor Green
} else {
    Write-Host "✗ Error: app.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the dashboard directory" -ForegroundColor Yellow
    exit 1
}

# Check if streamlit is installed
try {
    $streamlitVersion = & python -c "import streamlit; print(streamlit.__version__)" 2>$null
    Write-Host "✓ Streamlit installed (version $streamlitVersion)" -ForegroundColor Green
} catch {
    Write-Host "✗ Streamlit not installed!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Launching dashboard..." -ForegroundColor Cyan
Write-Host "Dashboard will open in your browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Launch Streamlit
streamlit run app.py
