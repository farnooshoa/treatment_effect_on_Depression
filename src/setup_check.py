"""
Setup Helper Script
This script helps you set up the project and find the data file.
"""

import os
import sys

def check_environment():
    """Check if all dependencies are installed"""
    print("=" * 80)
    print("CHECKING ENVIRONMENT")
    print("=" * 80)
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:<20} installed")
        except ImportError:
            print(f"✗ {package:<20} MISSING")
            missing.append(package)
    
    if missing:
        print("\n" + "=" * 80)
        print("MISSING PACKAGES")
        print("=" * 80)
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All packages installed!")
    return True


def find_data_file():
    """Find the data file"""
    print("\n" + "=" * 80)
    print("LOOKING FOR DATA FILE")
    print("=" * 80)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations
    possible_locations = [
        os.path.join(script_dir, 'data_generated.csv'),
        os.path.join(script_dir, '..', 'data', 'data_generated.csv'),
        os.path.join(script_dir, 'data', 'data_generated.csv'),
        os.path.join(script_dir, '..', 'data_generated.csv'),
    ]
    
    print(f"\nScript directory: {script_dir}")
    print("\nSearching in:")
    
    found = None
    for i, path in enumerate(possible_locations, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        status = "✓ FOUND" if exists else "✗ not found"
        print(f"  {i}. {abs_path}")
        print(f"     {status}")
        
        if exists and found is None:
            found = abs_path
    
    if found:
        print(f"\n✓ Data file found at: {found}")
        print(f"  Size: {os.path.getsize(found):,} bytes")
        return True
    else:
        print("\n" + "=" * 80)
        print("DATA FILE NOT FOUND")
        print("=" * 80)
        print("\nPlease place 'data_generated.csv' in one of these locations:")
        print(f"  1. {os.path.join(script_dir, 'data_generated.csv')}")
        print(f"  2. {os.path.abspath(os.path.join(script_dir, '..', 'data', 'data_generated.csv'))}")
        print(f"  3. {os.path.join(script_dir, 'data', 'data_generated.csv')}")
        return False


def check_project_structure():
    """Check project structure"""
    print("\n" + "=" * 80)
    print("PROJECT STRUCTURE")
    print("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    expected_files = {
        'treatment_effect_model.py': 'Main implementation',
        'optimizations.py': 'Optimization proposals',
        'run_baseline.py': 'Easy runner script',
    }
    
    print(f"\nChecking src/ directory: {script_dir}")
    print()
    
    all_present = True
    for filename, description in expected_files.items():
        filepath = os.path.join(script_dir, filename)
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"{status} {filename:<30} {description}")
        if not exists:
            all_present = False
    
    return all_present


def main():
    """Main setup check"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  TREATMENT EFFECT ESTIMATION - SETUP CHECK".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    # Check 1: Environment
    env_ok = check_environment()
    
    # Check 2: Project structure
    structure_ok = check_project_structure()
    
    # Check 3: Data file
    data_ok = find_data_file()
    
    # Summary
    print("\n" + "=" * 80)
    print("SETUP SUMMARY")
    print("=" * 80)
    
    status_env = "✓" if env_ok else "✗"
    status_structure = "✓" if structure_ok else "✗"
    status_data = "✓" if data_ok else "✗"
    
    print(f"\n{status_env} Dependencies installed")
    print(f"{status_structure} Project files present")
    print(f"{status_data} Data file found")
    
    if env_ok and structure_ok and data_ok:
        print("\n" + "=" * 80)
        print("✓ ALL CHECKS PASSED - READY TO RUN!")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python treatment_effect_model.py")
        print("  python run_baseline.py")
        print()
        return True
    else:
        print("\n" + "=" * 80)
        print("✗ SETUP INCOMPLETE")
        print("=" * 80)
        print("\nPlease fix the issues above before running the code.")
        print()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
