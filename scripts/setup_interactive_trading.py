#!/usr/bin/env python3
"""
Setup Script for Interactive Trading Module
==========================================

Installs additional dependencies required for the interactive trading module.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages for interactive trading"""
    print("🎯 Setting up Interactive Trading Module Dependencies")
    print("=" * 60)
    
    # Required packages
    packages = [
        "pygame>=2.5.0",  # For sound notifications
        "plyer>=2.1.0"    # For desktop notifications
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("✅ All dependencies installed successfully!")
        print("\n🚀 You can now run the interactive trading module:")
        print("   python main.py interactive-trading")
    else:
        print("⚠️  Some packages failed to install. The module may work with limited functionality.")
        print("   Sound notifications require pygame")
        print("   Desktop notifications require plyer")
    
    print("\n📝 Note: If you encounter issues with pygame on Windows, try:")
    print("   pip install pygame --upgrade")
    print("   or")
    print("   conda install pygame")

if __name__ == "__main__":
    main()
