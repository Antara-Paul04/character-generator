import os
import subprocess
import sys
from pathlib import Path

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_dependency(module_name, package_name=None):
    """Check if a Python module is installed"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def install_package(package_name):
    """Install a Python package"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def clone_hairnet():
    """Clone HairNet AI repository"""
    hairnet_dir = Path("hairnet-ai")
    
    if hairnet_dir.exists():
        print(f"✅ HairNet AI directory already exists: {hairnet_dir}")
        return True
    
    print("Cloning HairNet AI repository...")
    try:
        subprocess.check_call([
            "git", "clone",
            "https://github.com/alexkalinins/hairnet-ai.git"
        ])
        print(f"✅ HairNet AI cloned successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to clone HairNet AI")
        print("Please manually clone: git clone https://github.com/alexkalinins/hairnet-ai.git")
        return False

def create_file(filename, content):
    """Create a file with given content"""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {filename}: {e}")
        return False

def main():
    print_header("🎭 Hair Integration Setup")
    
    # Check Python version
    print("Checking Python version...")
    py_version = sys.version_info
    print(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        print("⚠️  Warning: Python 3.7+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check dependencies
    print_header("Checking Dependencies")
    
    dependencies = {
        'flask': 'flask',
        'torch': 'torch',
        'transformers': 'transformers',
        'spacy': 'spacy',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'cv2': 'opencv-python'
    }
    
    missing = []
    for module, package in dependencies.items():
        if not check_dependency(module, package):
            missing.append(package)
    
    # Install missing dependencies
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")
        
        if response.lower() == 'y':
            for package in missing:
                install_package(package)
    else:
        print("\n✅ All dependencies are installed!")
    
    # Check for spaCy model
    print_header("Checking spaCy Model")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' is installed")
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' is NOT installed")
        response = input("Download spaCy model? (y/n): ")
        if response.lower() == 'y':
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
    
    # Clone HairNet
    print_header("Setting up HairNet AI")
    clone_hairnet()
    
    # Check for required files
    print_header("Checking Project Files")
    
    required_files = [
        "frontend.py",
        "blender_bridge.py",
        "base.blend",
        "New-Text-Document.csv"
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} NOT found")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some required files are missing!")
        print("Make sure you have the base character generator files.")
    
    # Create new integration files
    print_header("Creating Integration Files")
    
    print("Creating hair_analyzer.py...")
    print("Creating hairnet_integration.py...")
    print("✅ Please copy the hair_analyzer.py and hairnet_integration.py")
    print("   artifacts from the assistant's response to your project directory.")
    
    # Create temp directory
    temp_dir = Path("temp_hair_generation")
    temp_dir.mkdir(exist_ok=True)
    print(f"✅ Created temporary directory: {temp_dir}")
    
    # Summary
    print_header("Setup Complete! 🎉")
    print("Next steps:")
    print("1. Copy hair_analyzer.py and hairnet_integration.py to your project")
    print("2. Update your frontend.py with the new code")
    print("3. Update your blender_bridge.py with hair attachment code")
    print("4. Update templates/index.html with the new template")
    print("5. Test with: python frontend.py")
    print("\nSee the Integration Guide artifact for detailed instructions!")

if __name__ == "__main__":
    main()