"""
Quick Start Guide for Campus Navigation App
Run this script to test the setup and verify everything is working
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("Checking dependencies...")
    
    dependencies = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'faiss': 'faiss',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            logger.error(f"  ✗ {package_name} NOT FOUND")
            missing.append(package_name)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.info(f"Install them with: pip install {' '.join(missing)}")
        return False
    
    logger.info("All dependencies installed!")
    return True

def check_dataset():
    """Check if dataset directory structure is correct"""
    logger.info("\nChecking dataset structure...")
    
    base_dir = Path(__file__).parent.absolute()
    dataset_dir = base_dir / "dataset"
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    required_dirs = ["train", "test"]
    for split in required_dirs:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.error(f"Missing directory: {split_dir}")
            return False
        
        locations = [d.name for d in split_dir.iterdir() if d.is_dir()]
        image_count = 0
        
        for loc_dir in split_dir.iterdir():
            if loc_dir.is_dir():
                images = list(loc_dir.glob("*.jpg")) + \
                        list(loc_dir.glob("*.jpeg")) + \
                        list(loc_dir.glob("*.png"))
                image_count += len(images)
                logger.info(f"  {split}/{loc_dir.name}: {len(images)} images")
        
        logger.info(f"  Total in {split}: {image_count} images")
    
    logger.info("Dataset structure OK!")
    return True

def check_modules():
    """Check if all local modules can be imported"""
    logger.info("\nChecking local modules...")
    
    modules = [
        'config',
        'data_loader',
        'embedding_model',
        'faiss_index',
        'pathfinding',
    ]
    
    base_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(base_dir))
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {module_name}.py")
        except ImportError as e:
            logger.error(f"  ✗ {module_name}.py: {e}")
            all_ok = False
    
    if all_ok:
        logger.info("All modules OK!")
    
    return all_ok

def check_index():
    """Check if FAISS index exists"""
    logger.info("\nChecking FAISS index...")
    
    base_dir = Path(__file__).parent.absolute()
    index_file = base_dir / "campus.index"
    labels_file = base_dir / "labels.npy"
    
    if index_file.exists() and labels_file.exists():
        logger.info(f"  ✓ Index found: {index_file.name}")
        logger.info(f"  ✓ Labels found: {labels_file.name}")
        logger.info("Index OK!")
        return True
    else:
        logger.warning("Index not found!")
        logger.info("This is expected on first run.")
        logger.info(f"\nTo build the index, run:")
        logger.info(f"  python train_index.py")
        return False

def main():
    """Run all checks"""
    logger.info("=" * 60)
    logger.info("CAMPUS NAVIGATION APP - SETUP CHECK")
    logger.info("=" * 60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Local Modules", check_modules),
        ("FAISS Index", check_index),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"\nError during {check_name} check: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name}: {status}")
    
    all_passed = all(result for _, result in results[:-1])  # Exclude index check
    index_ok = results[-1][1]
    
    logger.info("\n" + "=" * 60)
    
    if all_passed:
        logger.info("✓ Setup looks good!")
        
        if not index_ok:
            logger.info("\nNext steps:")
            logger.info("1. Build the FAISS index:")
            logger.info("   python train_index.py")
            logger.info("\n2. Test the app:")
            logger.info("   python app.py")
        else:
            logger.info("\nYou're ready to go! Try:")
            logger.info("   python app.py")
    else:
        logger.error("✗ Setup has issues. Please fix the problems above.")
    
    logger.info("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
