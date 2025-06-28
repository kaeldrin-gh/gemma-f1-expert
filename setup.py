#!/usr/bin/env python3
"""
Quick setup script for the Gemma F1 Expert project.

This script helps new users get started by:
1. Checking system requirements
2. Installing dependencies 
3. Collecting initial F1 data
4. Providing next steps

Usage:
    python setup.py
"""

import sys
import subprocess
import platform
from pathlib import Path
import requests


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ required")
        print(f"Current version: {platform.python_version()}")
        return False
    
    print(f"✅ Python {platform.python_version()} - Compatible")
    return True


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Available: {gpu_name} ({memory:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU detected - training will be slow")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_internet():
    """Check internet connectivity for data collection."""
    try:
        response = requests.get("https://api.jolpi.ca/ergast/f1/current.json", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity - Can fetch F1 data")
            return True
        else:
            print("⚠️  Jolpica API not accessible")
            return False
    except Exception:
        print("⚠️  No internet connection - data collection will fail")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "logs"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")


def collect_initial_data():
    """Collect initial F1 data."""
    print("🏎️  Collecting initial F1 data...")
    
    try:
        # Run data collection scripts
        subprocess.run([sys.executable, "data/fetch_jolpica.py"], check=True, timeout=300)
        print("✅ Jolpica data collected")
        
        subprocess.run([sys.executable, "data/scrape_press.py"], check=True, timeout=180)
        print("✅ Press data collected")
        
        subprocess.run([sys.executable, "data/build_dataset.py"], check=True, timeout=60)
        print("✅ Dataset created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data collection failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Data collection timed out")
        return False


def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("\n📋 Next Steps:")
    print("1. Train the model:")
    print("   python src/train_lora.py")
    print("\n2. Test the model:")
    print("   python src/generate.py \"Who won the 2023 Monaco GP?\"")
    print("\n3. Start interactive chat:")
    print("   python src/generate.py --interactive")
    print("\n4. Launch web interface:")
    print("   streamlit run src/webapp.py")
    print("\n5. Run evaluation:")
    print("   python src/evaluate.py")
    print("\n6. Run tests:")
    print("   pytest tests/ -v")
    
    print("\n💡 Tips:")
    print("- Training takes ~30-45 minutes on an 8GB GPU")
    print("- Use VS Code tasks for easier workflow")
    print("- Check the Jupyter notebook for Colab training")
    
    print("\n📚 Documentation:")
    print("- README.md - Full project documentation")
    print("- notebooks/00_train.ipynb - Colab training guide")
    print("- License: Apache 2.0")


def main():
    """Main setup function."""
    print("🏎️  Gemma F1 Expert Setup")
    print("=" * 30)
    
    # System checks
    print("\n🔍 System Checks:")
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n❌ Setup failed - incompatible Python version")
        sys.exit(1)
    
    # Check internet
    internet_ok = check_internet()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    deps_ok = install_dependencies()
    
    if not deps_ok:
        print("\n❌ Setup failed - could not install dependencies")
        sys.exit(1)
    
    # Check GPU after installing PyTorch
    print("\n🖥️  Hardware Check:")
    check_gpu()
    
    # Collect data if internet is available
    if internet_ok:
        print("\n📊 Data Collection:")
        data_ok = collect_initial_data()
        
        if not data_ok:
            print("⚠️  Data collection failed - you can try manually later")
    else:
        print("⚠️  Skipping data collection - no internet connection")
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup complete! Ready to train your F1 expert! 🏁")


if __name__ == "__main__":
    main()
