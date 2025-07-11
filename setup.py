#!/usr/bin/env python3
"""
Setup script for Phi-2 Fine-tuning Project

This script automates the installation and initial setup process.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["books", "models", "outputs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/ directory")

def download_sample_books():
    """Download sample books from Project Gutenberg."""
    print("\n📚 Downloading sample books...")
    
    # Popular books from Project Gutenberg
    books = {
        "1342": "Pride and Prejudice",
        "11": "Alice in Wonderland",
        "1661": "The Count of Monte Cristo",
        "84": "Frankenstein",
        "98": "A Tale of Two Cities"
    }
    
    downloaded = 0
    for book_id, title in books.items():
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                filename = f"books/gutenberg_{book_id}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"✅ Downloaded: {title}")
                downloaded += 1
            else:
                print(f"❌ Failed to download: {title}")
                
        except Exception as e:
            print(f"❌ Error downloading {title}: {e}")
    
    print(f"\n📊 Downloaded {downloaded}/{len(books)} sample books")
    return downloaded > 0

def check_gpu():
    """Check GPU availability."""
    print("\n🖥️ Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet. GPU check will be available after installation.")
        return False

def create_config_file():
    """Create a configuration file with default settings."""
    print("\n⚙️ Creating configuration file...")
    
    config = {
        "model_name": "microsoft/phi-2",
        "training": {
            "num_epochs": 3,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "max_length": 2048
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        },
        "generation": {
            "default_temperature": 0.8,
            "default_top_p": 0.9,
            "default_max_length": 500
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created config.json with default settings")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Add your own .txt book files to the 'books/' directory")
    print("2. Run training: python phi2_finetuning.py --mode train")
    print("3. Launch UI: python phi2_finetuning.py --mode ui")
    print("4. Test the model: python example_usage.py")
    
    print("\n📖 Available commands:")
    print("• python phi2_finetuning.py --mode train          # Train the model")
    print("• python phi2_finetuning.py --mode ui            # Launch web interface")
    print("• python phi2_finetuning.py --mode inference     # Command-line interface")
    print("• python example_usage.py                        # Run examples")
    
    print("\n📚 Sample books downloaded to 'books/' directory")
    print("💡 You can add more .txt files to improve training")
    
    print("\n🔧 Configuration:")
    print("• Edit config.json to modify training parameters")
    print("• Modify phi2_finetuning.py for advanced customization")
    
    print("\n🚀 Happy writing!")

def main():
    """Main setup function."""
    print("🚀 Phi-2 Fine-tuning Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        print("Please check your internet connection and try again.")
        return
    
    # Create directories
    create_directories()
    
    # Download sample books
    download_sample_books()
    
    # Create config file
    create_config_file()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()