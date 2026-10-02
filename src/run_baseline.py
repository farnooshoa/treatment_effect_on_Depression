#!/usr/bin/env python3
"""
QUICK START SCRIPT

This script provides an easy way to run the baseline implementation
with appropriate defaults and progress tracking.
"""

import sys
import os
from datetime import datetime

def print_header():
    """Print welcome header"""
    print("="*80)
    print(" "*20 + "TREATMENT EFFECT ESTIMATION")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required = ['torch', 'numpy', 'pandas', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return False
    
    print("\nAll dependencies satisfied!\n")
    return True

def check_data_file():
    """Check if data file exists"""
    data_path = 'data\data_generated.csv'
    if os.path.exists(data_path):
        print(f"✓ Data file found: {data_path}\n")
        return True
    else:
        print(f"✗ Data file not found: {data_path}")
        print("Please ensure the CSV file is uploaded.\n")
        return False

def run_baseline():
    """Run the baseline implementation"""
    print("="*80)
    print("RUNNING BASELINE IMPLEMENTATION")
    print("="*80)
    print("\nThis will train a three-stage model:")
    print("  1. Representation learning (stability, propensity, balance)")
    print("  2. Outcome prediction (factual outcomes and treatment effects)")
    print("  3. Diffusion model (counterfactual generation)")
    print("\nEach stage uses early stopping with patience=15.")
    print("Expected total time: 30-60 minutes on CPU, 10-20 minutes on GPU.\n")
    
    input("Press Enter to continue...")
    print()
    
    # Import and run
    try:
        from treatment_effect_model import main
        print("\nStarting training...\n")
        model, results = main()
        
        print("\n" + "="*80)
        print("BASELINE TRAINING COMPLETE!")
        print("="*80)
        print("\nResults Summary:")
        print(f"  Factual MSE:  {results['factual_mse']:.4f}")
        print(f"  Factual RMSE: {results['factual_rmse']:.4f}")
        print(f"  Factual MAE:  {results['factual_mae']:.4f}")
        print(f"  Factual R²:   {results['factual_r2']:.4f}")
        print("\nFiles saved:")
        print("  - baseline_results.csv")
        print("  - baseline_model.pt")
        print("\n" + "="*80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def offer_ablation():
    """Ask if user wants to run ablation study"""
    print("\n" + "="*80)
    print("ABLATION STUDY")
    print("="*80)
    print("\nWould you like to run the ablation study to compare optimizations?")
    print("This will test 6 different configurations:")
    print("  1. Baseline")
    print("  2. + Attention mechanism")
    print("  3. + Contrastive learning")
    print("  4. + Adversarial balancing")
    print("  5. + Residual denoiser")
    print("  6. All optimizations combined")
    print("\nNote: This takes significantly longer (2-3 hours on CPU).")
    
    response = input("\nRun ablation study? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            from optimizations import run_ablation_study
            import torch
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            filepath = '/mnt/user-data/uploads/data_generated.csv'
            
            print("\nStarting ablation study...\n")
            results = run_ablation_study(filepath, device=device)
            
            print("\n" + "="*80)
            print("ABLATION STUDY COMPLETE!")
            print("="*80)
            print("\nResults saved to: ablation_results.csv")
            print("Please review the file for detailed comparisons.\n")
            
        except Exception as e:
            print(f"\nERROR during ablation study: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping ablation study.")

def main():
    """Main execution flow"""
    print_header()
    
    # Check requirements
    if not check_dependencies():
        return
    
    if not check_data_file():
        return
    
    # Run baseline
    success = run_baseline()
    
    if success:
        # Offer ablation study
        offer_ablation()
    
    print("\n" + "="*80)
    print("SESSION COMPLETE")
    print("="*80)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThank you for reviewing this implementation!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
