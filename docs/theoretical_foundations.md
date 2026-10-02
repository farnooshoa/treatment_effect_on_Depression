# Theoretical Foundations and Implementation Guide

> Complete mathematical derivations and theoretical background for the treatment effect estimation model.

---

## Table of Contents

1. [Theoretical Framework](#1-theoretical-framework)
2. [Training Objective Derivation](#2-training-objective-derivation)
3. [Architecture Design Choices](#3-architecture-design-choices)
4. [Training Procedure](#4-training-procedure)
5. [Inference Procedures](#5-inference-procedures)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Proposed Optimizations](#7-proposed-optimizations)

---

## 1. Theoretical Framework

### 1.1 Causal Inference Framework

The fundamental problem of causal inference is that we only observe one potential outcome per unit. For patient *i* receiving treatment *t*, we observe *y<sub>i</sub>(t)* but not the counterfactual outcomes *y<sub>i</sub>(t')* for *t' ≠ t*.

**Individual Treatment Effect (ITE):**

```
τᵢ(t, t') = yᵢ(t) - yᵢ(t')
```

**Conditional Average Treatment Effect (CATE):**

```
τ(x, t, t') = E[y(t) - y(t') | X = x]
```

Our goal is to estimate these quantities from observational data where treatment assignment is confounded by patient characteristics.

### 1.2 Representation Learning for Causal Inference

We decompose the covariate space X into two disjoint representations:

**1. Stable Representation (S):** 
- Captures features stable across contexts and treatments
- Contains information predictive of outcomes
- Independent of treatment selection mechanisms

**2. Confounding Representation (C):**
- Captures features influencing both treatment assignment and outcomes
- Represents confounders that must be balanced
- Used for generating valid counterfactuals

**Key Insight:** By learning these disjoint representations, we can:
- Improve generalization through stable features
- Account for confounding through balanced representations  
- Generate valid counterfactuals by manipulating C while keeping S fixed

### 1.3 Identifiability Assumptions

For causal identification, we rely on:

**1. Unconfoundedness (Ignorability):**
```
Y(t) ⊥ T | X  for all t
```
All confounders are measured and included in X.

**2. Positivity (Overlap):**
```
0 < P(T = t | X = x) < 1  for all t and x in the support
```
Every treatment has non-zero probability for all covariate values.

**3. Stable Unit Treatment Value Assumption (SUTVA):**
- No interference between units
- Treatment is well-defined and consistent

**4. Representation Identifiability:**

The decomposition X → (S, C) is identifiable through:
- Stability constraint on S across augmented views
- Treatment predictability from C
- Distributional balance of C across treatments

### 1.4 Doubly Robust Estimation

The doubly robust estimator combines outcome regression and propensity weighting:

```
τ_DR(t, t₀) = E[(I(T=t)/π(t|X)) · (Y - μ(t,X)) + μ(t,X) - μ(t₀,X)]
```

Where:
- μ(t,X) is the outcome model: E[Y | T=t, X]
- π(t|X) is the propensity score: P(T=t | X)
- I(T=t) is the indicator function

**Key Property:** This estimator remains consistent if EITHER:
1. The outcome model μ(t,X) is correctly specified, OR
2. The propensity model π(t|X) is correctly specified

This provides a safety net against model misspecification.

### 1.5 Diffusion Models for Counterfactual Generation

We use conditional diffusion models to generate counterfactual confounding representations.

**Forward diffusion process (gradually adds Gaussian noise):**

```
q(Cₜ | C₀) = N(Cₜ; √(ᾱₜ)C₀, (1-ᾱₜ)I)
```

Where ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ and αₜ = 1 - βₜ.

**Reverse process (learns to denoise):**

```
p_θ(Cₜ₋₁ | Cₜ, S, T) = N(Cₜ₋₁; μ_θ(Cₜ, t, S, T), Σ_θ(Cₜ, t))
```

**Conditioning on S and T ensures generated counterfactuals:**
1. Maintain consistency with stable patient characteristics (S)
2. Reflect the target treatment (T)

**Advantages over alternatives:**
- Probabilistic modeling captures uncertainty
- Gradual denoising preserves fine-grained structure
- Conditioning enables precise control over generation

---

## 2. Training Objective Derivation

### Complete Loss Function

```
L_total = λ₁·L_stability + λ₂·L_propensity + λ₃·L_balance + λ₄·L_collapse + 
          λ₅·L_outcome + λ₆·L_ITE + λ₇·L_diffusion
```

### 2.1 Stability Loss (L_stability)

**Objective:** Ensure stable representation S is invariant to data augmentation.

**Mathematical Form:**
```
L_stability = E_{x,ε₁,ε₂} [||Φ_S(x + ε₁) - Φ_S(x + ε₂)||²]
```

Where:
- x is the input features
- ε₁, ε₂ are independent augmentation noise
- Φ_S is the stable representation encoder

**Optimization Variant (Contrastive):**
```
L_stability^contrast = -E_x [log(exp(sim(S₁, S₂)/τ) / Σₖ exp(sim(S₁, Sₖ)/τ))]
```

Where sim(·,·) is cosine similarity and τ is temperature.

**Hyperparameter:** λ₁ = 1.0

### 2.2 Propensity Loss (L_propensity)

**Objective:** Learn confounding representation that predicts treatment.

**Mathematical Form:**
```
L_propensity = E_{x,T} [- log π_β(T | Φ_C(x))]
```

This is standard cross-entropy loss for multi-class classification.

**Why This Matters:** If C can predict treatment, it captures confounding. By ensuring C predicts treatment while S does not, we achieve disentanglement.

**Hyperparameter:** λ₂ = 1.0

### 2.3 Balance Loss (L_balance)

**Objective:** Ensure confounding representation is balanced across treatments.

**Mathematical Form (MMD):**
```
L_balance = Σ_{t,t'} MMD²(Cₜ, Cₜ')
```

Where:
```
MMD²(Cₜ, Cₜ') = E[k(Cₜ, Cₜ)] + E[k(Cₜ', Cₜ')] - 2E[k(Cₜ, Cₜ')]
```

And k(·,·) is the RBF kernel:
```
k(Cᵢ, Cⱼ) = exp(-||Cᵢ - Cⱼ||² / (2σ²))
```

**Bandwidth Selection:** We use the median heuristic:
```
σ = median({||Cᵢ - Cⱼ|| : i ≠ j})
```

**Optimization Variant (Adversarial):**

Train discriminator D to predict treatment from C:
```
L_balance^adv = max_D E[log D(T | C)] - λ_GP · E[(||∇_C D(T | C)||₂ - 1)²]
```

Where λ_GP controls gradient penalty for training stability.

**Hyperparameter:** λ₃ = 0.5

### 2.4 Anti-Collapse Loss (L_collapse)

**Objective:** Prevent trivial solutions where representations collapse to constants.

**Mathematical Form:**
```
L_collapse = L_decorr + L_variance
```

**Decorrelation:**
```
L_decorr = ||Corr(S) - I||²_F + ||Corr(C) - I||²_F
```

Encourages different dimensions to capture different information.

**Variance:**
```
L_variance = ReLU(ε - Var(S)) + ReLU(ε - Var(C))
```

Ensures features don't all become zero.

**Hyperparameters:** 
- λ₄ = 0.01 (small weight as mentioned in prompt)
- ε = 0.1 (variance threshold)

### 2.5 Outcome Loss (L_outcome)

**Objective:** Predict factual outcomes accurately.

**Mathematical Form:**
```
L_outcome = E_{x,T,Y} [||f_θ(Φ_S(x), Φ_C(x), T) - Y||²]
```

This is standard supervised regression loss.

**Hyperparameter:** λ₅ = 1.0

### 2.6 ITE Loss (L_ITE)

**Objective:** Learn to predict treatment effects using pseudo-outcomes.

**Mathematical Form:**
```
L_ITE = E_x [||g_ω(Φ_S(x), Φ_C(x)) - τ_DR(x)||²]
```

Where τ_DR is the doubly robust pseudo-outcome:
```
τ_DR(t) = (I(T=t)/π̂(t|x)) · (Y - f̂(S,C,t)) + f̂(S,C,t) - f̂(S,C,t₀)
```

**Stabilization Techniques:**

**1. Propensity Clipping:**
```
π̂_clip(t|x) = clip(π̂(t|x), ε_min, ε_max)
```
Prevents extreme weights from high-leverage observations.
- Typical: ε_min = 0.01, ε_max = 0.99

**2. Winsorization:**
```
τ_DR = clip(τ_DR, q_low, q_high)
```
Caps extreme pseudo-outcomes.
- Typical: q_low = 5th percentile, q_high = 95th percentile

**3. Cross-Fitting:**
Use 2-fold cross-fitting to compute f̂ and π̂, reducing overfitting bias in pseudo-outcome construction.

**Hyperparameter:** λ₆ = 0.5

### 2.7 Diffusion Loss (L_diffusion)

**Objective:** Train denoiser to reverse the diffusion process.

**Mathematical Form:**
```
L_diffusion = E_{C₀,t,ε,S,T} [||ε - ε_θ(√(ᾱₜ)C₀ + √(1-ᾱₜ)ε, t, S, T)||²]
```

Where:
- C₀ is the clean confounding representation
- t ~ Uniform({1, ..., T}) is the diffusion timestep
- ε ~ N(0, I) is the added noise
- ε_θ is the denoiser network

**Noise Schedule:** Linear schedule from β_start = 0.0001 to β_end = 0.02
```
βₜ = β_start + (t/T) · (β_end - β_start)
```

**Optimization Variant:** Learn schedule with parameterized sigmoid:
```
βₜ = 0.02 · sigmoid(wₜ)  where wₜ are learnable parameters
```

**Hyperparameters:**
- λ₇ = 1.0
- N_steps = 100

### 2.8 Final Objective

```
L_total = 1.0·L_stability + 1.0·L_propensity + 0.5·L_balance + 0.01·L_collapse +
          1.0·L_outcome + 0.5·L_ITE + 1.0·L_diffusion
```

This objective is optimized in three stages:
1. Representation learning (L_stability + L_propensity + L_balance + L_collapse)
2. Outcome prediction (L_outcome + L_ITE, with encoder frozen)
3. Diffusion model (L_diffusion, with encoder frozen)

---

## 3. Architecture Design Choices

### 3.1 Representation Encoder

**Architecture:** 2-layer MLP with BatchNorm and Dropout

```
Input (input_dim) → Linear(256) → BatchNorm → ReLU → Dropout(0.1)
                  → Linear(128) → BatchNorm → ReLU → Dropout(0.1)
                  → Stable Head: Linear(128)
                  → Confound Head: Linear(128)
```

**Design Rationale:**
- Moderate depth balances expressiveness and overfitting risk
- BatchNorm stabilizes training and provides implicit regularization
- Dropout (0.1) prevents overfitting without being too aggressive
- Separate heads ensure disentanglement at architectural level

### 3.2 Propensity Head

**Architecture:** 2-layer MLP

```
Confound (confound_dim) → Linear(64) → ReLU → Dropout(0.1)
                        → Linear(n_treatments)
```

**Design Rationale:**
- Lightweight (64 hidden units) because propensity is typically smooth
- No BatchNorm to allow for subtle distributions

### 3.3 Outcome Model

**Architecture:** 2-layer MLP with treatment embedding

```
Treatment → Embedding(16)
[Stable, Confound, Treatment_Emb] → Linear(128) → BatchNorm → ReLU → Dropout(0.1)
                                  → Linear(64) → BatchNorm → ReLU → Dropout(0.1)
                                  → Linear(1)
```

**Design Rationale:**
- Treatment embedding (dim 16) allows model to learn treatment similarities
- Sufficient capacity (128 → 64) for complex outcome functions
- BatchNorm and dropout for regularization

### 3.4 Diffusion Denoiser

**Architecture:** 3-layer MLP with time and treatment embeddings

```
Timestep → Embedding(32)
Treatment → Embedding(16)
[C_noisy, Timestep_Emb, Stable, Treatment_Emb] → Linear(128) → ReLU
                                                 → Linear(128) → ReLU
                                                 → Linear(128) → ReLU
                                                 → Linear(confound_dim)
```

**Design Rationale:**
- Deeper network (3 layers) to handle complex denoising
- Time embedding allows model to adapt denoising to noise level
- No dropout/BatchNorm as these can interfere with diffusion dynamics

**Optimization:** Residual connections for better gradient flow

---

## 4. Training Procedure

### 4.1 Three-Stage Training

**Stage 1: Representation Learning**
- Train encoder and propensity head
- Losses: stability, propensity, balance, anti-collapse
- Epochs: up to 100 with early stopping (patience 15)
- Learning rate: 10⁻³
- Weight decay: 10⁻⁵

**Stage 2: Outcome Prediction**
- Freeze encoder, train outcome model and ITE head
- Losses: outcome, ITE
- Epochs: up to 100 with early stopping (patience 15)
- Learning rate: 10⁻³
- Weight decay: 10⁻⁵

**Stage 3: Diffusion Model**
- Freeze encoder, train denoiser
- Loss: diffusion
- Epochs: up to 100 with early stopping (patience 15)
- Learning rate: 10⁻³
- Weight decay: 10⁻⁵

**Rationale:** Staged training prevents interference between objectives and ensures stable representations before learning downstream tasks.

### 4.2 Data Augmentation

**For Stability Learning:**
- Gaussian noise: N(0, σ²I) with σ = 0.1
- Feature dropout: Drop each feature with probability 0.1

**Rationale:** Augmentation creates multiple views of same patient, forcing stable representation to capture invariant features.

### 4.3 Gradient Clipping

All models use gradient norm clipping at 1.0 to prevent exploding gradients, especially important for the adversarial and diffusion components.

---

## 5. Inference Procedures

### 5.1 Factual Prediction

For observed treatment T and covariates X:
1. Encode: S, C = Φ(X)
2. Predict: ŷ = f(S, C, T)

**Time Complexity:** O(1) - Single forward pass

### 5.2 Counterfactual Prediction (with Diffusion)

For counterfactual treatment T_cf:
1. Encode: S, C = Φ(X)
2. Sample C_cf ~ p(C | S, T_cf) using reverse diffusion
3. Predict: ŷ_cf = f(S, C_cf, T_cf)

**Reverse Diffusion Algorithm:**
```
Start from C_T ~ N(0, I)
For t = T down to 1:
    Predict noise: ε̂ = ε_θ(Cₜ, t, S, T_cf)
    Denoise: Cₜ₋₁ = (Cₜ - βₜ·ε̂) / √(αₜ) + σₜ·z
Return C₀ (counterfactual representation)
```

**Time Complexity:** O(N_steps) ≈ O(100) - 100 denoising steps

### 5.3 Counterfactual Prediction (without Diffusion)

For ablation purposes, direct prediction:
1. Encode: S, C = Φ(X)
2. Predict: ŷ_cf = f(S, C, T_cf)

**Limitation:** Uses factual C, which may be incompatible with T_cf.

---

## 6. Evaluation Metrics

### 6.1 Factual Outcome Prediction

- **MSE**: Mean Squared Error = E[(ŷ - y)²]
- **RMSE**: Root MSE = √MSE
- **MAE**: Mean Absolute Error = E[|ŷ - y|]
- **R²**: Coefficient of Determination = 1 - (SS_res / SS_tot)

These metrics assess how well the model predicts observed outcomes.

### 6.2 Treatment Effect Estimation

When ground truth available:
- **PEHE**: Precision in Estimation of Heterogeneous Effects
  ```
  PEHE = √E[(τ̂ᵢ - τᵢ)²]
  ```
- **ATE Error**: |ATE_estimated - ATE_true|
- **Policy Value**: Expected outcome under learned optimal policy

In observational settings without ground truth:
- Proxy metrics (outcome prediction)
- Overlap diagnostics
- Sensitivity analyses

---

## 7. Proposed Optimizations

### 7.1 Attention-Based Disentanglement

**Problem:** Simple MLPs may not optimally separate stable vs confounding features.

**Solution:** Use multi-head attention to learn feature importance:
- Separate query vectors for S and C
- Attention over encoded features
- More flexible than fixed linear projections

**Expected Improvement:** 3-5% better R² through improved representation quality.

### 7.2 Contrastive Stability Loss

**Problem:** MSE loss doesn't explicitly maximize positive pair similarity relative to negative pairs.

**Solution:** InfoNCE contrastive loss:
```
L = -log(exp(sim(S₁, S₂)/τ) / Σₖ exp(sim(S₁, Sₖ)/τ))
```

**Expected Improvement:** 2-4% better stability under distribution shift.

### 7.3 Adversarial Balance

**Problem:** MMD is limited to statistical moments and may miss complex distributional differences.

**Solution:** Adversarial discriminator that tries to predict treatment from C, while encoder tries to fool it (domain-adversarial learning).

**Expected Improvement:** 5-8% better counterfactual accuracy.

### 7.4 Learned Noise Schedule

**Problem:** Fixed linear schedule may be suboptimal for specific data.

**Solution:** Parameterize βₜ with learnable parameters optimized during training.

**Expected Improvement:** 2-3% better counterfactual generation quality.

### 7.5 Residual Denoiser

**Problem:** Deep denoisers suffer from vanishing gradients.

**Solution:** Add residual connections in denoiser architecture.

**Expected Improvement:** Faster convergence (20-30% fewer epochs) and 1-2% better final performance.

---

## References

1. Shalit, U., et al. (2017). "Estimating individual treatment effect: generalization bounds and algorithms." *ICML*.

2. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.

3. Yoon, J., et al. (2018). "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets." *ICLR*.

4. Hassanpour, N., & Greiner, R. (2020). "Learning Disentangled Representations for CounterFactual Regression." *ICLR*.

5. Kennedy, E. H. (2020). "Towards optimal doubly robust estimation of heterogeneous causal effects." *arXiv*.
