# Quick Start Guide

> Get started with the treatment effect estimation implementation in 5 minutes

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/farnooshoa/treatment_effect_on_Depression.git
cd treatment_effect_on_Depression
```

### 2. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Option 1: Quick Start (Recommended)

```bash
cd src
python run_baseline.py
```

This will:
- ✓ Check dependencies
- ✓ Train all three stages
- ✓ Evaluate on test set
- ✓ Save results
- ✓ Optionally run ablation study

### Option 2: Run Baseline Only

```bash
cd src
python treatment_effect_model.py
```

Output files:
- `baseline_results.csv` - Performance metrics
- `baseline_model.pt` - Saved model weights

### Option 3: Run Ablation Study

```bash
cd src
python optimizations.py
```

Output file:
- `ablation_results.csv` - Comparison of 6 configurations

---

## What You'll See

### Stage 1: Representation Learning
```
STAGE 1: Representation Learning
================================================================================
Epoch 1/100 | Train Loss: 2.3456 | Val Loss: 2.1234 | Stability: 0.5432
Epoch 2/100 | Train Loss: 2.1234 | Val Loss: 2.0123 | Stability: 0.4987
...
Stage 1 complete. Best validation loss: 1.8765
```

### Stage 2: Outcome Prediction
```
STAGE 2: Outcome and ITE Prediction
================================================================================
Epoch 1/100 | Train Loss: 8.7654 | Val Loss: 8.9876 | Outcome: 8.5432
Epoch 2/100 | Train Loss: 8.5432 | Val Loss: 8.7654 | Outcome: 8.3210
...
Stage 2 complete. Best validation loss: 8.3456
```

### Stage 3: Diffusion Model
```
STAGE 3: Diffusion Model for Counterfactuals
================================================================================
Epoch 1/100 | Train Loss: 0.4567 | Val Loss: 0.4789
Epoch 2/100 | Train Loss: 0.4321 | Val Loss: 0.4543
...
Stage 3 complete. Best validation loss: 0.4123
```

### Final Results
```
TEST SET EVALUATION
================================================================================
Factual Outcome Prediction:
  MSE:  10.2345
  RMSE: 3.1992
  MAE:  2.4567
  R²:   0.6543

Results saved to baseline_results.csv
Model saved to baseline_model.pt
```

---

## File Structure

```
treatment_effect_on_Depression/
├── README.md                       # Main documentation
├── quick_start.md                  # This file
├── requirements.txt                # Dependencies
│
├── src/
│   ├── treatment_effect_model.py  # Main implementation (1000+ lines)
│   ├── optimizations.py            # 5 optimizations + ablation (500+ lines)
│   └── run_baseline.py             # Easy execution script
│
├── docs/
│   ├── theoretical_foundations.md  # Mathematical derivations
│   └── executive_summary.md        # High-level overview
│
├── data/
│   └── data_generated.csv          # Dataset
│
└── results/
    ├── baseline_results.csv        # Performance metrics
    ├── ablation_results.csv        # Ablation study results
    └── baseline_model.pt           # Saved model
```

---

## Expected Runtime

| Task | CPU | GPU |
|------|-----|-----|
| Baseline Training | 30-60 min | 10-20 min |
| Ablation Study | 2-3 hours | 30-60 min |

---

## Key Features

✅ **Complete Training Objective**: All 7 loss components derived mathematically

✅ **Three-Stage Training**: Proper encoder freezing in stages 2 & 3

✅ **Patient-Level Splitting**: No data leakage across visits

✅ **Doubly Robust ITE**: With propensity clipping and winsorization

✅ **Conditional Diffusion**: 100-step reverse sampling for counterfactuals

✅ **5 Optimizations**: Attention, contrastive, adversarial, learned schedule, residual

✅ **Comprehensive Docs**: Theory, code, and usage all documented

---

## Troubleshooting

### Issue: "Module not found"
```bash
# Install missing packages
pip install torch numpy pandas scikit-learn matplotlib
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size in the code
# Edit treatment_effect_model.py, line ~XXX:
batch_size = 64  # Change to 32 or 16
```

### Issue: "File not found: data_generated.csv"
```bash
# Make sure you're in the right directory
cd treatment_effect_on_Depression
ls data/  # Should see data_generated.csv
```

### Issue: Training is slow
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, code will use CPU (slower but works)
```

---

## Next Steps

1. **Run the baseline** to see it works
2. **Read the documentation** in `docs/`
3. **Try the ablation study** to compare optimizations
4. **Modify the code** to experiment

---

## Need Help?

- **Full Documentation**: See `README.md`
- **Theory**: See `docs/theoretical_foundations.md`
- **Overview**: See `docs/executive_summary.md`
---

**Ready to go? Run this:**

```bash
cd src && python run_baseline.py
```

Good luck! 🚀
