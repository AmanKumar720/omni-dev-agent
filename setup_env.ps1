# Setup script for Omni-Dev Agent development environment
# Run this script to activate the virtual environment and install dependencies

Write-Host "Setting up Omni-Dev Agent development environment..." -ForegroundColor Green

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
    
    Write-Host "Virtual environment activated!" -ForegroundColor Green
    Write-Host "You can now run: python -m src.main" -ForegroundColor Cyan
} else {
    Write-Host "Virtual environment not found. Please run: python -m venv venv" -ForegroundColor Red
}
