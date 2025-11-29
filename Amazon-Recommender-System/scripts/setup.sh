#!/bin/bash

# Amazon Recommender System Setup Script for Mac/Linux

echo "Setting up Amazon Recommender System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Java is installed (required for Spark)
if ! command -v java &> /dev/null; then
    echo "Java is required for Apache Spark but not installed. Please install Java 8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p output/models
mkdir -p output/results
mkdir -p output/plots

# Set environment variables
echo "Setting up environment variables..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Setup completed successfully!"
echo ""
echo "To activate the virtual environment: source venv/bin/activate"
echo "To run the application: python scripts/run_app.py"
echo "To run tests: pytest tests/"