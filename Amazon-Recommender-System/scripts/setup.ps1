# Setup script for Windows
Write-Host "Setting up Amazon Recommender System..." -ForegroundColor Green

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "Found: $pythonVersion"
} catch {
    Write-Host "Python 3 required" -ForegroundColor Red
    exit 1
}

# Check Java
try {
    $javaVersion = java -version 2>&1
    Write-Host "Found Java"
} catch {
    Write-Host "Java 8+ required for Spark" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate and install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "output\models"
New-Item -ItemType Directory -Force -Path "output\results"

Write-Host "Setup completed!" -ForegroundColor Green
Write-Host "Activate: .\venv\Scripts\Activate.ps1"
Write-Host "Run: python scripts\run_app.py --interactive"