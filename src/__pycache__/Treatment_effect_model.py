"""
Treatment Effect Estimation with Causal Representation Learning and Diffusion Models

This implementation addresses the PhD application task for causal treatment effect estimation
on depression data using representation learning and conditional diffusion models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# SECTION 1: DERIVED TRAINING OBJECTIVE (Theoretical Component)
# =============================================================================

"""
TRAINING OBJECTIVE DERIVATION:

Based on the methodological description, the full training objective is a weighted sum
of the following loss components:

L_total = λ₁·L_stability + λ₂·L_propensity + λ₃·L_balance + λ₄·L_collapse + 
          λ₅·L_outcome + λ₆·L_ITE + λ₇·L_diffusion

Where:

1. L_stability: Enforces stability of the stable representation S across augmented views
   L_stability = MSE(S₁, S₂) = ||Φ_S(x₁) - Φ_S(x₂)||²
   where x₁, x₂ are two augmented views of the same input

2. L_propensity: Treatment prediction loss from confounding representation C
   L_propensity = CrossEntropy(π_β(C), T)
   where π_β is the propensity head and T is the observed treatment

3. L_balance: Kernel-based distributional balance across treatments
   L_balance = MMD²(C_t₁, C_t₂) for all treatment pairs (t₁, t₂)
   Using Maximum Mean Discrepancy with RBF kernel:
   MMD²(C_t₁, C_t₂) = E[k(C_t₁, C_t₁')] + E[k(C_t₂, C_t₂')] - 2E[k(C_t₁, C_t₂)]
   where k(·,·) is the RBF kernel with bandwidth σ

4. L_collapse: Anti-collapse regularization to prevent trivial solutions
   L_collapse = L_decorr + L_variance
   - L_decorr: Decorrelation loss to encourage diversity in representations
     L_decorr = ||Corr(S) - I||²_F + ||Corr(C) - I||²_F
   - L_variance: Variance maintenance to prevent collapse to zero
     L_variance = max(0, ε - Var(S)) + max(0, ε - Var(C))

5. L_outcome: Supervised regression loss for factual outcome prediction
   L_outcome = MSE(f_θ(S, C, T), y)
   where f_θ is the outcome model and y is the observed next-visit HAMD score

6. L_ITE: Doubly robust pseudo-outcome loss for treatment effect estimation
   L_ITE = MSE(g_ω(S, C), τ_DR)
   where τ_DR is the doubly robust estimator:
   τ_DR(t) = (I(T=t)/π(t|x)) · (y - f(S,C,t)) + f(S,C,t) - f(S,C,t_baseline)
   
   With stabilization:
   - Propensity clipping: π_clipped = clip(π, ε_min, 1-ε_max)
   - Winsorization: τ_DR = clip(τ_DR, q_low, q_high)
   - Cross-fitting with 2 folds to reduce bias

7. L_diffusion: Denoising diffusion objective for counterfactual generation
   L_diffusion = E_t,ε [||ε - ε_θ(C_t, t, S, T)||²]
   where:
   - C_t = √(ᾱ_t)·C + √(1-ᾱ_t)·ε is the noisy confounding representation
   - ε ~ N(0, I) is standard Gaussian noise
   - ε_θ is the denoiser network
   - ᾱ_t follows a linear noise schedule over N_steps

HYPERPARAMETERS:
- λ₁ (stability weight): 1.0
- λ₂ (propensity weight): 1.0
- λ₃ (balance weight): 0.5
- λ₄ (collapse weight): 0.01 (small weight as mentioned)
- λ₅ (outcome weight): 1.0
- λ₆ (ITE weight): 0.5
- λ₇ (diffusion weight): 1.0
- σ (RBF bandwidth): median heuristic
- ε (variance threshold): 0.1
- ε_min, ε_max (propensity clip): 0.01, 0.99
- q_low, q_high (winsorization): 5th and 95th percentiles
- N_steps (diffusion steps): 100
"""


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

class DepressionDataset(Dataset):
    """Dataset for treatment effect estimation on depression data"""
    
    def __init__(self, data: pd.DataFrame, scaler: Optional[StandardScaler] = None, 
                 fit_scaler: bool = False, treatment_map: Optional[Dict] = None):
        """
        Args:
            data: DataFrame with patient data
            scaler: StandardScaler for numeric features
            fit_scaler: Whether to fit the scaler on this data
            treatment_map: Mapping from treatment strings to indices
        """
        self.data = data.reset_index(drop=True)
        
        # Define feature columns
        self.numeric_features = ['AGE']
        self.categorical_features = ['PROTOCOL', 'THERAPY_STATUS', 'ORIGIN', 
                                     'GENDER', 'GEOCODE', 'THERPHAS']
        self.hamd_items = [f'HAMD{i:02d}' for i in range(1, 18)]
        
        # Compute outcome (next-visit HAMD total score)
        self.data['outcome'] = self.data[self.hamd_items].sum(axis=1)
        
        # Process treatment
        if treatment_map is None:
            unique_treatments = sorted(self.data['THERAPY'].unique())
            self.treatment_map = {t: i for i, t in enumerate(unique_treatments)}
        else:
            self.treatment_map = treatment_map
        
        self.data['treatment_idx'] = self.data['THERAPY'].map(self.treatment_map)
        self.n_treatments = len(self.treatment_map)
        
        # Standardize numeric features
        if fit_scaler:
            self.scaler = StandardScaler()
            self.data[self.numeric_features] = self.scaler.fit_transform(
                self.data[self.numeric_features]
            )
        elif scaler is not None:
            self.scaler = scaler
            self.data[self.numeric_features] = self.scaler.transform(
                self.data[self.numeric_features]
            )
        else:
            self.scaler = None
        
        # Fill missing values with 0 after standardization
        self.data[self.numeric_features] = self.data[self.numeric_features].fillna(0)
        
        # Process categorical features (map to indices)
        self.cat_mappings = {}
        for col in self.categorical_features:
            unique_vals = ['UNKNOWN'] + sorted([str(v) for v in self.data[col].unique() if pd.notna(v)])
            self.cat_mappings[col] = {v: i for i, v in enumerate(unique_vals)}
            self.data[col] = self.data[col].fillna('UNKNOWN').astype(str).map(self.cat_mappings[col])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Numeric features
        numeric = torch.tensor(row[self.numeric_features].values, dtype=torch.float32)
        
        # Categorical features (as indices)
        categorical = torch.tensor([row[col] for col in self.categorical_features], dtype=torch.long)
        
        # Treatment and outcome
        treatment = torch.tensor(row['treatment_idx'], dtype=torch.long)
        outcome = torch.tensor(row['outcome'], dtype=torch.float32)
        
        return {
            'numeric': numeric,
            'categorical': categorical,
            'treatment': treatment,
            'outcome': outcome,
            'patient_id': row['UNIQUEID']
        }


def load_and_split_data(filepath: str, test_size: float = 0.2, val_size: float = 0.1):
    """
    Load data and split by patient ID to avoid leakage
    
    Args:
        filepath: Path to CSV file
        test_size: Proportion of patients for test set
        val_size: Proportion of remaining patients for validation set
    
    Returns:
        train_df, val_df, test_df, treatment_map
    """
    df = pd.read_csv(filepath)
    
    # Get unique patient IDs
    patient_ids = df['UNIQUEID'].unique()
    
    # Split by patient
    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=RANDOM_SEED
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size/(1-test_size), random_state=RANDOM_SEED
    )
    
    train_df = df[df['UNIQUEID'].isin(train_ids)]
    val_df = df[df['UNIQUEID'].isin(val_ids)]
    test_df = df[df['UNIQUEID'].isin(test_ids)]
    
    print(f"Data split: Train={len(train_df)} ({len(train_ids)} patients), "
          f"Val={len(val_df)} ({len(val_ids)} patients), "
          f"Test={len(test_df)} ({len(test_ids)} patients)")
    
    return train_df, val_df, test_df


# =============================================================================
# SECTION 3: MODEL ARCHITECTURE
# =============================================================================

class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical features"""
    
    def __init__(self, cat_mappings: Dict, embedding_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(mapping), embedding_dim)
            for name, mapping in cat_mappings.items()
        })
        self.embedding_dim = embedding_dim
    
    def forward(self, categorical_indices):
        """
        Args:
            categorical_indices: (batch_size, n_categorical)
        Returns:
            embeddings: (batch_size, n_categorical * embedding_dim)
        """
        embeddings = []
        for i, (name, emb_layer) in enumerate(self.embeddings.items()):
            embeddings.append(emb_layer(categorical_indices[:, i]))
        return torch.cat(embeddings, dim=1)


class RepresentationEncoder(nn.Module):
    """
    Encoder that maps input features to stable and confounding representations
    """
    
    def __init__(self, input_dim: int, stable_dim: int = 128, confound_dim: int = 128):
        super().__init__()
        self.stable_dim = stable_dim
        self.confound_dim = confound_dim
        
        # Shared encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for stable and confounding representations
        self.stable_head = nn.Linear(128, stable_dim)
        self.confound_head = nn.Linear(128, confound_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_dim)
        Returns:
            S: Stable representation (batch_size, stable_dim)
            C: Confounding representation (batch_size, confound_dim)
        """
        h = self.encoder(x)
        S = self.stable_head(h)
        C = self.confound_head(h)
        return S, C


class PropensityHead(nn.Module):
    """Propensity score predictor from confounding representation"""
    
    def __init__(self, confound_dim: int, n_treatments: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(confound_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_treatments)
        )
    
    def forward(self, C):
        """
        Args:
            C: Confounding representation (batch_size, confound_dim)
        Returns:
            logits: Treatment propensity logits (batch_size, n_treatments)
        """
        return self.network(C)


class OutcomeModel(nn.Module):
    """Outcome predictor from representations and treatment"""
    
    def __init__(self, stable_dim: int, confound_dim: int, n_treatments: int, 
                 treatment_emb_dim: int = 16):
        super().__init__()
        self.treatment_embedding = nn.Embedding(n_treatments, treatment_emb_dim)
        
        input_dim = stable_dim + confound_dim + treatment_emb_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, S, C, treatment):
        """
        Args:
            S: Stable representation (batch_size, stable_dim)
            C: Confounding representation (batch_size, confound_dim)
            treatment: Treatment indices (batch_size,)
        Returns:
            outcome: Predicted outcome (batch_size, 1)
        """
        t_emb = self.treatment_embedding(treatment)
        x = torch.cat([S, C, t_emb], dim=1)
        return self.network(x)


class ITEHead(nn.Module):
    """Individual Treatment Effect predictor"""
    
    def __init__(self, stable_dim: int, confound_dim: int, n_treatments: int):
        super().__init__()
        input_dim = stable_dim + confound_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_treatments)
        )
    
    def forward(self, S, C):
        """
        Args:
            S: Stable representation (batch_size, stable_dim)
            C: Confounding representation (batch_size, confound_dim)
        Returns:
            ite: Treatment effects relative to baseline (batch_size, n_treatments)
        """
        x = torch.cat([S, C], dim=1)
        return self.network(x)


class DiffusionDenoiser(nn.Module):
    """
    Conditional diffusion model for generating counterfactual confounding representations
    """
    
    def __init__(self, confound_dim: int, stable_dim: int, n_treatments: int,
                 timestep_emb_dim: int = 32, treatment_emb_dim: int = 16):
        super().__init__()
        self.timestep_embedding = nn.Embedding(1000, timestep_emb_dim)  # Max 1000 steps
        self.treatment_embedding = nn.Embedding(n_treatments, treatment_emb_dim)
        
        input_dim = confound_dim + timestep_emb_dim + stable_dim + treatment_emb_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, confound_dim)
        )
    
    def forward(self, C_noisy, timestep, S, treatment):
        """
        Args:
            C_noisy: Noisy confounding representation (batch_size, confound_dim)
            timestep: Diffusion timestep (batch_size,)
            S: Stable representation (batch_size, stable_dim)
            treatment: Treatment index (batch_size,)
        Returns:
            noise_pred: Predicted noise (batch_size, confound_dim)
        """
        t_emb = self.timestep_embedding(timestep)
        treat_emb = self.treatment_embedding(treatment)
        x = torch.cat([C_noisy, t_emb, S, treat_emb], dim=1)
        return self.network(x)


# =============================================================================
# SECTION 4: DATA AUGMENTATION
# =============================================================================

def augment_batch(numeric_features, noise_std: float = 0.1, dropout_prob: float = 0.1):
    """
    Create two augmented views of numeric features
    
    Args:
        numeric_features: Original features (batch_size, n_numeric)
        noise_std: Standard deviation of Gaussian noise
        dropout_prob: Probability of feature dropout
    
    Returns:
        view1, view2: Two augmented views
    """
    batch_size, n_features = numeric_features.shape
    
    # View 1: Add Gaussian noise
    noise1 = torch.randn_like(numeric_features) * noise_std
    view1 = numeric_features + noise1
    
    # View 2: Add different Gaussian noise and apply dropout
    noise2 = torch.randn_like(numeric_features) * noise_std
    dropout_mask = (torch.rand(batch_size, n_features) > dropout_prob).float().to(numeric_features.device)
    view2 = (numeric_features + noise2) * dropout_mask
    
    return view1, view2


# =============================================================================
# SECTION 5: LOSS FUNCTIONS
# =============================================================================

def compute_stability_loss(S1, S2):
    """Stability loss: MSE between stable representations from two views"""
    return F.mse_loss(S1, S2)


def compute_propensity_loss(propensity_logits, treatments):
    """Propensity loss: Cross-entropy for treatment prediction"""
    return F.cross_entropy(propensity_logits, treatments)


def rbf_kernel(X, Y, sigma=None):
    """
    Compute RBF kernel between two sets of samples
    
    Args:
        X: (n, d) tensor
        Y: (m, d) tensor
        sigma: Bandwidth parameter (if None, use median heuristic)
    
    Returns:
        K: (n, m) kernel matrix
    """
    XX = (X ** 2).sum(dim=1, keepdim=True)
    YY = (Y ** 2).sum(dim=1, keepdim=True)
    XY = torch.mm(X, Y.t())
    distances = XX + YY.t() - 2 * XY
    
    if sigma is None:
        # Median heuristic
        sigma = torch.median(distances[distances > 0]).item()
        sigma = max(sigma, 1e-4)
    
    return torch.exp(-distances / (2 * sigma ** 2))


def compute_mmd_loss(C, treatments, n_treatments):
    """
    Maximum Mean Discrepancy for distributional balance across treatments
    """
    if n_treatments < 2:
        return torch.tensor(0.0).to(C.device)
    
    mmd = 0.0
    count = 0
    
    for t1 in range(n_treatments):
        for t2 in range(t1 + 1, n_treatments):
            mask_t1 = (treatments == t1)
            mask_t2 = (treatments == t2)
            
            if mask_t1.sum() > 0 and mask_t2.sum() > 0:
                C_t1 = C[mask_t1]
                C_t2 = C[mask_t2]
                
                K_t1_t1 = rbf_kernel(C_t1, C_t1).mean()
                K_t2_t2 = rbf_kernel(C_t2, C_t2).mean()
                K_t1_t2 = rbf_kernel(C_t1, C_t2).mean()
                
                mmd += K_t1_t1 + K_t2_t2 - 2 * K_t1_t2
                count += 1
    
    return mmd / max(count, 1)


def compute_decorrelation_loss(X, eps=1e-8):
    """
    Decorrelation loss to encourage diversity in representation dimensions
    """
    # Center the features
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix
    cov = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1 + eps)
    std = torch.sqrt(torch.diag(cov) + eps)
    corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + eps)
    
    # Penalize off-diagonal elements
    identity = torch.eye(corr.shape[0]).to(X.device)
    return ((corr - identity) ** 2).mean()


def compute_variance_loss(X, min_var=0.1):
    """
    Variance maintenance loss to prevent collapse
    """
    var = X.var(dim=0)
    return F.relu(min_var - var).mean()


def compute_collapse_loss(S, C, min_var=0.1):
    """Combined anti-collapse regularization"""
    decorr_loss = compute_decorrelation_loss(S) + compute_decorrelation_loss(C)
    var_loss = compute_variance_loss(S, min_var) + compute_variance_loss(C, min_var)
    return decorr_loss + var_loss


def compute_outcome_loss(predictions, targets):
    """Outcome prediction loss"""
    return F.mse_loss(predictions.squeeze(), targets)


# =============================================================================
# SECTION 6: DOUBLY ROBUST ESTIMATOR FOR ITE
# =============================================================================

def compute_doubly_robust_pseudo_outcomes(
    outcome_model, encoder, data_loader, n_treatments, baseline_treatment,
    device, clip_min=0.01, clip_max=0.99, winsorize_quantiles=(0.05, 0.95)
):
    """
    Compute doubly robust pseudo-outcomes using cross-fitting
    
    Args:
        outcome_model: Trained outcome model
        encoder: Trained encoder
        data_loader: DataLoader with data
        n_treatments: Number of treatments
        baseline_treatment: Index of baseline treatment
        device: torch device
        clip_min, clip_max: Propensity score clipping bounds
        winsorize_quantiles: Quantiles for winsorization
    
    Returns:
        pseudo_outcomes: (n_samples, n_treatments) array of pseudo-outcomes
    """
    encoder.eval()
    outcome_model.eval()
    
    all_pseudo_outcomes = []
    
    with torch.no_grad():
        for batch in data_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)
            
            # Get representations
            x = torch.cat([numeric, categorical.float()], dim=1)  # Simplified for pseudo-outcome
            S, C = encoder(x)
            
            # Compute propensity scores (simplified - using uniform for baseline)
            propensity = torch.ones(len(treatment), n_treatments).to(device) / n_treatments
            propensity = torch.clamp(propensity, clip_min, clip_max)
            
            # Compute pseudo-outcomes for each treatment
            pseudo_batch = []
            for t in range(n_treatments):
                # Predict outcome under treatment t
                t_tensor = torch.full_like(treatment, t)
                y_pred_t = outcome_model(S, C, t_tensor).squeeze()
                
                # Predict outcome under baseline
                baseline_tensor = torch.full_like(treatment, baseline_treatment)
                y_pred_baseline = outcome_model(S, C, baseline_tensor).squeeze()
                
                # Doubly robust estimator
                indicator = (treatment == t).float()
                ipw_term = (indicator / propensity[torch.arange(len(treatment)), t]) * \
                           (outcome - y_pred_t)
                
                tau_dr = ipw_term + y_pred_t - y_pred_baseline
                pseudo_batch.append(tau_dr.unsqueeze(1))
            
            pseudo_batch = torch.cat(pseudo_batch, dim=1)
            all_pseudo_outcomes.append(pseudo_batch.cpu())
    
    all_pseudo_outcomes = torch.cat(all_pseudo_outcomes, dim=0)
    
    # Winsorization
    if winsorize_quantiles is not None:
        low_q = torch.quantile(all_pseudo_outcomes, winsorize_quantiles[0])
        high_q = torch.quantile(all_pseudo_outcomes, winsorize_quantiles[1])
        all_pseudo_outcomes = torch.clamp(all_pseudo_outcomes, low_q, high_q)
    
    return all_pseudo_outcomes


# =============================================================================
# SECTION 7: DIFFUSION MODEL UTILITIES
# =============================================================================

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule for diffusion"""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_schedule(timesteps=100):
    """
    Compute diffusion schedule parameters
    
    Returns:
        Dictionary with alpha, alpha_bar, etc.
    """
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod)
    }


def add_noise(C, t, schedule):
    """
    Add noise to confounding representation at timestep t
    
    Args:
        C: Clean confounding representation (batch_size, confound_dim)
        t: Timestep (batch_size,)
        schedule: Diffusion schedule dictionary
    
    Returns:
        C_noisy: Noisy representation
        noise: Added noise
    """
    noise = torch.randn_like(C)
    sqrt_alpha_bar = schedule['sqrt_alphas_cumprod'][t].view(-1, 1).to(C.device)
    sqrt_one_minus_alpha_bar = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1).to(C.device)
    
    C_noisy = sqrt_alpha_bar * C + sqrt_one_minus_alpha_bar * noise
    return C_noisy, noise


def compute_diffusion_loss(denoiser, C, S, treatment, schedule, n_steps=100):
    """
    Compute diffusion training loss
    
    Args:
        denoiser: Diffusion denoiser model
        C: Clean confounding representation (batch_size, confound_dim)
        S: Stable representation (batch_size, stable_dim)
        treatment: Treatment indices (batch_size,)
        schedule: Diffusion schedule
        n_steps: Number of diffusion steps
    
    Returns:
        loss: Diffusion denoising loss
    """
    batch_size = C.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, n_steps, (batch_size,)).to(C.device)
    
    # Add noise
    C_noisy, noise = add_noise(C, t, schedule)
    
    # Predict noise
    noise_pred = denoiser(C_noisy, t, S, treatment)
    
    # Compute MSE loss
    return F.mse_loss(noise_pred, noise)


@torch.no_grad()
def sample_counterfactual(denoiser, S, treatment_cf, schedule, confound_dim, n_steps=100):
    """
    Generate counterfactual confounding representation using reverse diffusion
    
    Args:
        denoiser: Trained diffusion denoiser
        S: Stable representation (batch_size, stable_dim)
        treatment_cf: Counterfactual treatment indices (batch_size,)
        schedule: Diffusion schedule
        confound_dim: Dimension of confounding representation
        n_steps: Number of diffusion steps
    
    Returns:
        C_cf: Generated counterfactual confounding representation
    """
    batch_size = S.shape[0]
    device = S.device
    
    # Start from pure noise
    C_t = torch.randn(batch_size, confound_dim).to(device)
    
    # Reverse diffusion process
    for t in reversed(range(n_steps)):
        t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)
        
        # Predict noise
        noise_pred = denoiser(C_t, t_batch, S, treatment_cf)
        
        # Compute previous sample
        alpha_t = schedule['alphas'][t].to(device)
        alpha_bar_t = schedule['alphas_cumprod'][t].to(device)
        
        if t > 0:
            alpha_bar_t_prev = schedule['alphas_cumprod'][t-1].to(device)
            noise = torch.randn_like(C_t)
        else:
            alpha_bar_t_prev = torch.tensor(1.0).to(device)
            noise = torch.zeros_like(C_t)
        
        # Denoise
        C_t = (C_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) / torch.sqrt(alpha_t)
        
        if t > 0:
            sigma_t = torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_t))
            C_t = C_t + sigma_t * noise
    
    return C_t


# =============================================================================
# SECTION 8: TRAINING PROCEDURE
# =============================================================================

class TreatmentEffectModel:
    """
    Complete treatment effect estimation model with all components
    """
    
    def __init__(self, input_dim, n_treatments, stable_dim=128, confound_dim=128,
                 device='cpu'):
        self.device = device
        self.n_treatments = n_treatments
        self.stable_dim = stable_dim
        self.confound_dim = confound_dim
        
        # Initialize models
        self.encoder = RepresentationEncoder(input_dim, stable_dim, confound_dim).to(device)
        self.propensity_head = PropensityHead(confound_dim, n_treatments).to(device)
        self.outcome_model = OutcomeModel(stable_dim, confound_dim, n_treatments).to(device)
        self.ite_head = ITEHead(stable_dim, confound_dim, n_treatments).to(device)
        self.denoiser = DiffusionDenoiser(confound_dim, stable_dim, n_treatments).to(device)
        
        # Diffusion schedule
        self.diffusion_schedule = get_diffusion_schedule(timesteps=100)
        for key in self.diffusion_schedule:
            self.diffusion_schedule[key] = self.diffusion_schedule[key].to(device)
        
        # Find most frequent treatment as baseline
        self.baseline_treatment = 0  # Will be set during training
        
    def train_stage1_representation(self, train_loader, val_loader, cat_embeddings,
                                   n_epochs=100, lr=1e-3, patience=15):
        """
        Stage 1: Train representation encoder with stability, propensity, and balance losses
        """
        print("\n" + "="*80)
        print("STAGE 1: Representation Learning")
        print("="*80)
        
        # Optimizer
        params = list(self.encoder.parameters()) + list(self.propensity_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loss weights (hyperparameters from derived objective)
        lambda_stability = 1.0
        lambda_propensity = 1.0
        lambda_balance = 0.5
        lambda_collapse = 0.01
        
        for epoch in range(n_epochs):
            self.encoder.train()
            self.propensity_head.train()
            
            epoch_losses = {'total': 0, 'stability': 0, 'propensity': 0, 
                           'balance': 0, 'collapse': 0}
            
            for batch in train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                
                # Embed categorical features
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                
                # Create two augmented views
                numeric_aug1, numeric_aug2 = augment_batch(numeric)
                x1 = torch.cat([numeric_aug1, cat_emb], dim=1)
                x2 = torch.cat([numeric_aug2, cat_emb], dim=1)
                
                # Forward pass on both views
                S1, C1 = self.encoder(x1)
                S2, C2 = self.encoder(x2)
                
                # Also get representations from original input
                S, C = self.encoder(x)
                
                # Compute losses
                loss_stability = compute_stability_loss(S1, S2)
                
                propensity_logits = self.propensity_head(C)
                loss_propensity = compute_propensity_loss(propensity_logits, treatment)
                
                loss_balance = compute_mmd_loss(C, treatment, self.n_treatments)
                
                loss_collapse = compute_collapse_loss(S, C, min_var=0.1)
                
                # Total loss
                loss = (lambda_stability * loss_stability + 
                       lambda_propensity * loss_propensity +
                       lambda_balance * loss_balance +
                       lambda_collapse * loss_collapse)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['stability'] += loss_stability.item()
                epoch_losses['propensity'] += loss_propensity.item()
                epoch_losses['balance'] += loss_balance.item()
                epoch_losses['collapse'] += loss_collapse.item()
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Validation
            val_loss = self._validate_stage1(val_loader, cat_embeddings, lambda_stability,
                                             lambda_propensity, lambda_balance, lambda_collapse)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_losses['total']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Stability: {epoch_losses['stability']:.4f} | "
                  f"Propensity: {epoch_losses['propensity']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_encoder_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
                self.best_propensity_state = {k: v.cpu().clone() for k, v in self.propensity_head.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.encoder.load_state_dict({k: v.to(self.device) for k, v in self.best_encoder_state.items()})
        self.propensity_head.load_state_dict({k: v.to(self.device) for k, v in self.best_propensity_state.items()})
        
        print(f"Stage 1 complete. Best validation loss: {best_val_loss:.4f}")
    
    def _validate_stage1(self, val_loader, cat_embeddings, lambda_stability, lambda_propensity,
                        lambda_balance, lambda_collapse):
        """Validation for stage 1"""
        self.encoder.eval()
        self.propensity_head.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                
                numeric_aug1, numeric_aug2 = augment_batch(numeric, noise_std=0.05, dropout_prob=0.05)
                x1 = torch.cat([numeric_aug1, cat_emb], dim=1)
                x2 = torch.cat([numeric_aug2, cat_emb], dim=1)
                
                S1, C1 = self.encoder(x1)
                S2, C2 = self.encoder(x2)
                S, C = self.encoder(x)
                
                loss_stability = compute_stability_loss(S1, S2)
                propensity_logits = self.propensity_head(C)
                loss_propensity = compute_propensity_loss(propensity_logits, treatment)
                loss_balance = compute_mmd_loss(C, treatment, self.n_treatments)
                loss_collapse = compute_collapse_loss(S, C, min_var=0.1)
                
                loss = (lambda_stability * loss_stability + 
                       lambda_propensity * loss_propensity +
                       lambda_balance * loss_balance +
                       lambda_collapse * loss_collapse)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_stage2_outcome(self, train_loader, val_loader, cat_embeddings,
                            n_epochs=100, lr=1e-3, patience=15):
        """
        Stage 2: Train outcome model and ITE head (encoder frozen)
        """
        print("\n" + "="*80)
        print("STAGE 2: Outcome and ITE Prediction")
        print("="*80)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Optimizer for outcome model and ITE head
        params = list(self.outcome_model.parameters()) + list(self.ite_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loss weights
        lambda_outcome = 1.0
        lambda_ite = 0.5
        
        for epoch in range(n_epochs):
            self.encoder.eval()
            self.outcome_model.train()
            self.ite_head.train()
            
            epoch_losses = {'total': 0, 'outcome': 0, 'ite': 0}
            
            for batch in train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                outcome = batch['outcome'].to(self.device)
                
                # Get representations (no gradient through encoder)
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                
                with torch.no_grad():
                    S, C = self.encoder(x)
                
                # Outcome prediction
                y_pred = self.outcome_model(S, C, treatment)
                loss_outcome = compute_outcome_loss(y_pred, outcome)
                
                # ITE prediction (using simplified pseudo-outcomes for now)
                # In full implementation, would use doubly robust estimator with cross-fitting
                ite_pred = self.ite_head(S, C)
                # Simplified: use observed outcomes as proxy
                loss_ite = F.mse_loss(ite_pred[:, treatment], outcome - outcome.mean())
                
                # Total loss
                loss = lambda_outcome * loss_outcome + lambda_ite * loss_ite
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['outcome'] += loss_outcome.item()
                epoch_losses['ite'] += loss_ite.item()
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Validation
            val_loss = self._validate_stage2(val_loader, cat_embeddings, lambda_outcome, lambda_ite)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_losses['total']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Outcome: {epoch_losses['outcome']:.4f} | "
                  f"ITE: {epoch_losses['ite']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_outcome_state = {k: v.cpu().clone() for k, v in self.outcome_model.state_dict().items()}
                self.best_ite_state = {k: v.cpu().clone() for k, v in self.ite_head.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.outcome_model.load_state_dict({k: v.to(self.device) for k, v in self.best_outcome_state.items()})
        self.ite_head.load_state_dict({k: v.to(self.device) for k, v in self.best_ite_state.items()})
        
        print(f"Stage 2 complete. Best validation loss: {best_val_loss:.4f}")
    
    def _validate_stage2(self, val_loader, cat_embeddings, lambda_outcome, lambda_ite):
        """Validation for stage 2"""
        self.encoder.eval()
        self.outcome_model.eval()
        self.ite_head.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                outcome = batch['outcome'].to(self.device)
                
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                S, C = self.encoder(x)
                
                y_pred = self.outcome_model(S, C, treatment)
                loss_outcome = compute_outcome_loss(y_pred, outcome)
                
                ite_pred = self.ite_head(S, C)
                loss_ite = F.mse_loss(ite_pred[:, treatment], outcome - outcome.mean())
                
                loss = lambda_outcome * loss_outcome + lambda_ite * loss_ite
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_stage3_diffusion(self, train_loader, val_loader, cat_embeddings,
                               n_epochs=100, lr=1e-3, patience=15):
        """
        Stage 3: Train diffusion model (encoder frozen)
        """
        print("\n" + "="*80)
        print("STAGE 3: Diffusion Model for Counterfactuals")
        print("="*80)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.encoder.eval()
            self.denoiser.train()
            
            epoch_loss = 0
            
            for batch in train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                
                # Get representations
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                
                with torch.no_grad():
                    S, C = self.encoder(x)
                
                # Compute diffusion loss
                loss = compute_diffusion_loss(self.denoiser, C, S, treatment, 
                                             self.diffusion_schedule, n_steps=100)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            
            # Validation
            val_loss = self._validate_stage3(val_loader, cat_embeddings)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_denoiser_state = {k: v.cpu().clone() for k, v in self.denoiser.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.denoiser.load_state_dict({k: v.to(self.device) for k, v in self.best_denoiser_state.items()})
        
        print(f"Stage 3 complete. Best validation loss: {best_val_loss:.4f}")
    
    def _validate_stage3(self, val_loader, cat_embeddings):
        """Validation for stage 3"""
        self.encoder.eval()
        self.denoiser.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                S, C = self.encoder(x)
                
                loss = compute_diffusion_loss(self.denoiser, C, S, treatment,
                                             self.diffusion_schedule, n_steps=100)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    @torch.no_grad()
    def predict(self, numeric, categorical, treatment, cat_embeddings, 
                counterfactual_treatment=None, use_diffusion=True):
        """
        Make predictions (factual or counterfactual)
        
        Args:
            numeric: Numeric features
            categorical: Categorical features
            treatment: Observed treatment
            cat_embeddings: Categorical embedding layer
            counterfactual_treatment: If provided, predict under this treatment
            use_diffusion: Whether to use diffusion for counterfactual generation
        
        Returns:
            predictions: Predicted outcomes
        """
        self.encoder.eval()
        self.outcome_model.eval()
        self.denoiser.eval()
        
        # Get representations
        cat_emb = cat_embeddings(categorical)
        x = torch.cat([numeric, cat_emb], dim=1)
        S, C = self.encoder(x)
        
        if counterfactual_treatment is None:
            # Factual prediction
            y_pred = self.outcome_model(S, C, treatment)
        else:
            if use_diffusion:
                # Generate counterfactual C using diffusion
                C_cf = sample_counterfactual(self.denoiser, S, counterfactual_treatment,
                                            self.diffusion_schedule, self.confound_dim, 
                                            n_steps=100)
                y_pred = self.outcome_model(S, C_cf, counterfactual_treatment)
            else:
                # Direct prediction without diffusion
                y_pred = self.outcome_model(S, C, counterfactual_treatment)
        
        return y_pred


# =============================================================================
# SECTION 9: EVALUATION METRICS
# =============================================================================

def evaluate_model(model, test_loader, cat_embeddings, device):
    """
    Evaluate model performance on test set
    
    Returns metrics for:
    - Factual outcome prediction (MSE, MAE)
    - Counterfactual predictions with and without diffusion
    """
    model.encoder.eval()
    model.outcome_model.eval()
    model.denoiser.eval()
    
    all_factual_preds = []
    all_factual_targets = []
    all_treatments = []
    
    with torch.no_grad():
        for batch in test_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)
            
            # Factual prediction
            y_pred = model.predict(numeric, categorical, treatment, cat_embeddings)
            
            all_factual_preds.append(y_pred.cpu())
            all_factual_targets.append(outcome.cpu())
            all_treatments.append(treatment.cpu())
    
    all_factual_preds = torch.cat(all_factual_preds).squeeze()
    all_factual_targets = torch.cat(all_factual_targets)
    all_treatments = torch.cat(all_treatments)
    
    # Compute metrics
    mse = F.mse_loss(all_factual_preds, all_factual_targets).item()
    mae = F.l1_loss(all_factual_preds, all_factual_targets).item()
    
    # Compute R²
    ss_res = ((all_factual_targets - all_factual_preds) ** 2).sum()
    ss_tot = ((all_factual_targets - all_factual_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()
    
    results = {
        'factual_mse': mse,
        'factual_mae': mae,
        'factual_r2': r2,
        'factual_rmse': np.sqrt(mse)
    }
    
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    print(f"Factual Outcome Prediction:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {np.sqrt(mse):.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return results


# =============================================================================
# SECTION 10: MAIN EXECUTION
# =============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("="*80)
    print("TREATMENT EFFECT ESTIMATION WITH CAUSAL REPRESENTATION LEARNING")
    print("="*80)
    print("\nLoading and preprocessing data...")
    
    # Load data
    filepath = '/mnt/user-data/uploads/data_generated.csv'
    train_df, val_df, test_df = load_and_split_data(filepath)
    
    # Create datasets
    train_dataset = DepressionDataset(train_df, fit_scaler=True)
    val_dataset = DepressionDataset(val_df, scaler=train_dataset.scaler, 
                                    treatment_map=train_dataset.treatment_map)
    test_dataset = DepressionDataset(test_df, scaler=train_dataset.scaler,
                                     treatment_map=train_dataset.treatment_map)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nNumber of treatments: {train_dataset.n_treatments}")
    print(f"Treatment mapping: {train_dataset.treatment_map}")
    
    # Initialize categorical embeddings
    cat_embeddings = CategoricalEmbedding(train_dataset.cat_mappings, embedding_dim=8).to(device)
    
    # Calculate input dimension
    n_numeric = len(train_dataset.numeric_features)
    n_categorical_features = len(train_dataset.categorical_features)
    embedding_dim = 8
    input_dim = n_numeric + n_categorical_features * embedding_dim
    
    print(f"\nInput dimension: {input_dim}")
    print(f"  Numeric features: {n_numeric}")
    print(f"  Categorical features: {n_categorical_features} × {embedding_dim} = "
          f"{n_categorical_features * embedding_dim}")
    
    # Initialize model
    model = TreatmentEffectModel(
        input_dim=input_dim,
        n_treatments=train_dataset.n_treatments,
        stable_dim=128,
        confound_dim=128,
        device=device
    )
    
    # Set baseline treatment (most frequent in training data)
    treatment_counts = train_df['THERAPY'].value_counts()
    baseline_treatment_name = treatment_counts.index[0]
    model.baseline_treatment = train_dataset.treatment_map[baseline_treatment_name]
    print(f"\nBaseline treatment: {baseline_treatment_name} (index {model.baseline_treatment})")
    
    # Training
    print("\n" + "="*80)
    print("STARTING TRAINING PROCEDURE")
    print("="*80)
    
    # Stage 1: Representation learning
    model.train_stage1_representation(train_loader, val_loader, cat_embeddings,
                                     n_epochs=100, lr=1e-3, patience=15)
    
    # Stage 2: Outcome and ITE prediction
    model.train_stage2_outcome(train_loader, val_loader, cat_embeddings,
                              n_epochs=100, lr=1e-3, patience=15)
    
    # Stage 3: Diffusion model
    model.train_stage3_diffusion(train_loader, val_loader, cat_embeddings,
                                 n_epochs=100, lr=1e-3, patience=15)
    
    # Evaluation
    results = evaluate_model(model, test_loader, cat_embeddings, device)
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv('/home/claude/baseline_results.csv', index=False)
    print("\nResults saved to baseline_results.csv")
    
    # Save model
    torch.save({
        'encoder': model.encoder.state_dict(),
        'propensity_head': model.propensity_head.state_dict(),
        'outcome_model': model.outcome_model.state_dict(),
        'ite_head': model.ite_head.state_dict(),
        'denoiser': model.denoiser.state_dict(),
        'cat_embeddings': cat_embeddings.state_dict(),
        'treatment_map': train_dataset.treatment_map,
        'scaler_mean': train_dataset.scaler.mean_,
        'scaler_scale': train_dataset.scaler.scale_,
    }, '/home/claude/baseline_model.pt')
    print("Model saved to baseline_model.pt")
    
    return model, results


if __name__ == '__main__':
    model, results = main()
    print("\n" + "="*80)
    print("BASELINE IMPLEMENTATION COMPLETE")
    print("="*80)