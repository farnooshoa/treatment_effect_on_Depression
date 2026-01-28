"""
OPTIMIZATION PROPOSALS AND ABLATION STUDIES

This file contains proposed optimizations to the baseline model along with
ablation studies comparing the optimized variants against the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from treatment_effect_model import *
import copy


# =============================================================================
# OPTIMIZATION 1: Attention-Based Representation Disentanglement
# =============================================================================

class AttentionDisentanglement(nn.Module):
    """
    Use attention mechanism to better disentangle stable and confounding representations
    
    Motivation: The baseline uses simple MLPs to separate S and C, but attention
    can learn which features are more important for stability vs. confounding,
    leading to better representation quality.
    """
    
    def __init__(self, input_dim: int, stable_dim: int = 128, confound_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.stable_dim = stable_dim
        self.confound_dim = confound_dim
        
        # Shared encoder
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
        
        # Multi-head attention for feature importance
        self.attention = nn.MultiheadAttention(128, n_heads, batch_first=True)
        
        # Separate attention queries for stable and confounding
        self.stable_query = nn.Parameter(torch.randn(1, 1, 128))
        self.confound_query = nn.Parameter(torch.randn(1, 1, 128))
        
        # Projection heads
        self.stable_head = nn.Linear(128, stable_dim)
        self.confound_head = nn.Linear(128, confound_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encode
        h = self.encoder(x)
        h_expanded = h.unsqueeze(1)  # (batch, 1, 128)
        
        # Attend for stable representation
        stable_query = self.stable_query.expand(batch_size, -1, -1)
        stable_attn_out, _ = self.attention(stable_query, h_expanded, h_expanded)
        S = self.stable_head(stable_attn_out.squeeze(1))
        
        # Attend for confounding representation
        confound_query = self.confound_query.expand(batch_size, -1, -1)
        confound_attn_out, _ = self.attention(confound_query, h_expanded, h_expanded)
        C = self.confound_head(confound_attn_out.squeeze(1))
        
        return S, C


# =============================================================================
# OPTIMIZATION 2: Contrastive Learning for Better Stability
# =============================================================================

class ContrastiveStabilityLoss(nn.Module):
    """
    Use contrastive learning (InfoNCE) for stronger stability enforcement
    
    Motivation: MSE loss for stability is effective but doesn't explicitly
    maximize agreement while minimizing agreement with negative samples.
    Contrastive loss can lead to more robust stable representations.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, S1, S2):
        """
        Args:
            S1, S2: Stable representations from two views (batch_size, stable_dim)
        
        Returns:
            Contrastive loss
        """
        batch_size = S1.shape[0]
        
        # Normalize representations
        S1_norm = F.normalize(S1, dim=1)
        S2_norm = F.normalize(S2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(S1_norm, S2_norm.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(S1.device)
        
        # Symmetric loss
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss1 + loss2) / 2


# =============================================================================
# OPTIMIZATION 3: Adversarial Balancing for Confounders
# =============================================================================

class AdversarialDiscriminator(nn.Module):
    """
    Adversarial discriminator to enforce treatment-invariant confounding representations
    
    Motivation: MMD loss provides statistical balance but adversarial training
    can learn more complex invariances, potentially improving counterfactual quality.
    """
    
    def __init__(self, confound_dim: int, n_treatments: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(confound_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_treatments)
        )
    
    def forward(self, C):
        return self.network(C)


def adversarial_balance_loss(encoder, discriminator, x, treatment, lambda_grad_penalty=10.0):
    """
    Compute adversarial balance loss with gradient penalty
    
    The encoder tries to fool the discriminator (making C treatment-invariant)
    while the discriminator tries to predict treatment from C.
    """
    # Forward pass through encoder
    S, C = encoder(x)
    
    # Clone C to avoid in-place operations
    C_for_discriminator = C.clone()
    
    # Discriminator prediction
    treatment_pred = discriminator(C_for_discriminator)
    
    # Adversarial loss for encoder (wants to confuse discriminator)
    encoder_loss = -F.cross_entropy(treatment_pred, treatment)
    
    # Loss for discriminator (wants to predict treatment correctly) 
    # Use detached C for discriminator training to avoid modifying encoder gradients
    C_detached = C.detach().clone().requires_grad_(True)
    treatment_pred_for_disc = discriminator(C_detached)
    discriminator_loss = F.cross_entropy(treatment_pred_for_disc, treatment)
    
    # Gradient penalty for stability
    alpha = torch.rand(C_detached.shape[0], 1, device=C_detached.device)
    
    # Create interpolated samples
    C_random = torch.randn_like(C_detached)
    C_interpolated = (alpha * C_detached + (1 - alpha) * C_random).requires_grad_(True)
    
    treatment_pred_interpolated = discriminator(C_interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=treatment_pred_interpolated.sum(),
        inputs=C_interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    discriminator_loss = discriminator_loss + lambda_grad_penalty * gradient_penalty
    
    return encoder_loss, discriminator_loss
    
    return encoder_loss, discriminator_loss


# =============================================================================
# OPTIMIZATION 4: Improved Diffusion with Learned Noise Schedule
# =============================================================================

class LearnedNoiseSchedule(nn.Module):
    """
    Learn the noise schedule instead of using fixed linear schedule
    
    Motivation: Different datasets may benefit from different noise schedules.
    Learning the schedule can improve counterfactual generation quality.
    """
    
    def __init__(self, timesteps=100):
        super().__init__()
        self.timesteps = timesteps
        
        # Parameterize betas with sigmoid to keep them in (0, 1)
        self.beta_params = nn.Parameter(torch.linspace(-2, 2, timesteps))
    
    def get_schedule(self):
        """Get current noise schedule"""
        betas = torch.sigmoid(self.beta_params) * 0.02  # Scale to (0, 0.02)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod)
        }


# =============================================================================
# OPTIMIZATION 5: Residual Connections in Denoiser
# =============================================================================

class ResidualDiffusionDenoiser(nn.Module):
    """
    Enhanced denoiser with residual connections for better gradient flow
    
    Motivation: Deep diffusion models can suffer from vanishing gradients.
    Residual connections improve training stability and final performance.
    """
    
    def __init__(self, confound_dim: int, stable_dim: int, n_treatments: int,
                 timestep_emb_dim: int = 32, treatment_emb_dim: int = 16):
        super().__init__()
        self.confound_dim = confound_dim
        
        self.timestep_embedding = nn.Embedding(1000, timestep_emb_dim)
        self.treatment_embedding = nn.Embedding(n_treatments, treatment_emb_dim)
        
        input_dim = confound_dim + timestep_emb_dim + stable_dim + treatment_emb_dim
        
        # Residual blocks
        self.input_proj = nn.Linear(input_dim, 128)
        
        self.res_block1 = self._make_res_block(128, 128)
        self.res_block2 = self._make_res_block(128, 128)
        self.res_block3 = self._make_res_block(128, 128)
        
        self.output_proj = nn.Linear(128, confound_dim)
    
    def _make_res_block(self, in_dim, out_dim):
        """Create a residual block"""
        return nn.ModuleDict({
            'net': nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim, out_dim)
            ),
            'norm': nn.LayerNorm(out_dim)
        })
    
    def _apply_res_block(self, x, block):
        """Apply residual block"""
        identity = x
        out = block['net'](x)
        out = out + identity
        out = block['norm'](out)
        out = F.relu(out)
        return out
    
    def forward(self, C_noisy, timestep, S, treatment):
        t_emb = self.timestep_embedding(timestep)
        treat_emb = self.treatment_embedding(treatment)
        
        x = torch.cat([C_noisy, t_emb, S, treat_emb], dim=1)
        x = self.input_proj(x)
        
        x = self._apply_res_block(x, self.res_block1)
        x = self._apply_res_block(x, self.res_block2)
        x = self._apply_res_block(x, self.res_block3)
        
        return self.output_proj(x)


# =============================================================================
# OPTIMIZED MODEL CLASS
# =============================================================================

class OptimizedTreatmentEffectModel(TreatmentEffectModel):
    """
    Enhanced model incorporating all optimizations
    """
    
    def __init__(self, input_dim, n_treatments, stable_dim=128, confound_dim=128,
                 device='cpu', use_attention=True, use_contrastive=True,
                 use_adversarial=True, use_learned_schedule=True, use_residual=True):
        # Initialize base components
        self.device = device
        self.n_treatments = n_treatments
        self.stable_dim = stable_dim
        self.confound_dim = confound_dim
        
        # Optimized encoder with attention
        if use_attention:
            self.encoder = AttentionDisentanglement(input_dim, stable_dim, confound_dim).to(device)
        else:
            self.encoder = RepresentationEncoder(input_dim, stable_dim, confound_dim).to(device)
        
        self.propensity_head = PropensityHead(confound_dim, n_treatments).to(device)
        self.outcome_model = OutcomeModel(stable_dim, confound_dim, n_treatments).to(device)
        self.ite_head = ITEHead(stable_dim, confound_dim, n_treatments).to(device)
        
        # Optimized denoiser with residual connections
        if use_residual:
            self.denoiser = ResidualDiffusionDenoiser(confound_dim, stable_dim, n_treatments).to(device)
        else:
            self.denoiser = DiffusionDenoiser(confound_dim, stable_dim, n_treatments).to(device)
        
        # Additional components for optimizations
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_loss = ContrastiveStabilityLoss().to(device)
        
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.discriminator = AdversarialDiscriminator(confound_dim, n_treatments).to(device)
        
        self.use_learned_schedule = use_learned_schedule
        if use_learned_schedule:
            self.noise_schedule_learner = LearnedNoiseSchedule(timesteps=100).to(device)
            self.diffusion_schedule = self.noise_schedule_learner.get_schedule()
        else:
            self.diffusion_schedule = get_diffusion_schedule(timesteps=100)
        
        for key in self.diffusion_schedule:
            self.diffusion_schedule[key] = self.diffusion_schedule[key].to(device)
        
        self.baseline_treatment = 0
    
    def train_stage1_representation(self, train_loader, val_loader, cat_embeddings,
                                   n_epochs=100, lr=1e-3, patience=15):
        """
        Enhanced Stage 1 with optimizations
        """
        print("\n" + "="*80)
        print("STAGE 1: Optimized Representation Learning")
        print("="*80)
        
        # Optimizers
        encoder_params = list(self.encoder.parameters()) + list(self.propensity_head.parameters())
        optimizer = torch.optim.Adam(encoder_params, lr=lr, weight_decay=1e-5)
        
        if self.use_adversarial:
            discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                                       lr=lr, weight_decay=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loss weights
        lambda_stability = 1.0
        lambda_propensity = 1.0
        lambda_balance = 0.5
        lambda_collapse = 0.01
        lambda_adversarial = 0.3 if self.use_adversarial else 0.0
        
        for epoch in range(n_epochs):
            self.encoder.train()
            self.propensity_head.train()
            if self.use_adversarial:
                self.discriminator.train()
            
            epoch_losses = {'total': 0, 'stability': 0, 'propensity': 0, 
                           'balance': 0, 'collapse': 0, 'adversarial': 0}
            
            for batch in train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                treatment = batch['treatment'].to(self.device)
                
                cat_emb = cat_embeddings(categorical)
                x = torch.cat([numeric, cat_emb], dim=1)
                
                # Create augmented views
                numeric_aug1, numeric_aug2 = augment_batch(numeric)
                x1 = torch.cat([numeric_aug1, cat_emb], dim=1)
                x2 = torch.cat([numeric_aug2, cat_emb], dim=1)
                
                # Forward pass
                S1, C1 = self.encoder(x1)
                S2, C2 = self.encoder(x2)
                S, C = self.encoder(x)
                
                # Stability loss (contrastive or MSE)
                if self.use_contrastive:
                    loss_stability = self.contrastive_loss(S1, S2)
                else:
                    loss_stability = compute_stability_loss(S1, S2)
                
                # Propensity loss
                propensity_logits = self.propensity_head(C)
                loss_propensity = compute_propensity_loss(propensity_logits, treatment)
                
                # Balance loss (MMD or adversarial)
                if self.use_adversarial:
                    encoder_adv_loss, disc_loss = adversarial_balance_loss(
                        self.encoder, self.discriminator, x, treatment
                    )
                    loss_balance = compute_mmd_loss(C, treatment, self.n_treatments)
                    loss_adversarial = encoder_adv_loss
                    
                    # Update discriminator
                    discriminator_optimizer.zero_grad()
                    disc_loss.backward(retain_graph=True)
                    discriminator_optimizer.step()
                else:
                    loss_balance = compute_mmd_loss(C, treatment, self.n_treatments)
                    loss_adversarial = torch.tensor(0.0)
                
                # Collapse loss
                loss_collapse = compute_collapse_loss(S, C, min_var=0.1)
                
                # Total loss
                loss = (lambda_stability * loss_stability + 
                       lambda_propensity * loss_propensity +
                       lambda_balance * loss_balance +
                       lambda_collapse * loss_collapse +
                       lambda_adversarial * loss_adversarial)
                
                # Update encoder
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['stability'] += loss_stability.item()
                epoch_losses['propensity'] += loss_propensity.item()
                epoch_losses['balance'] += loss_balance.item()
                epoch_losses['collapse'] += loss_collapse.item()
                epoch_losses['adversarial'] += loss_adversarial.item()
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Validation
            val_loss = self._validate_stage1(val_loader, cat_embeddings, lambda_stability,
                                             lambda_propensity, lambda_balance, lambda_collapse)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train: {epoch_losses['total']:.4f} | "
                  f"Val: {val_loss:.4f} | Stab: {epoch_losses['stability']:.4f} | "
                  f"Prop: {epoch_losses['propensity']:.4f} | Adv: {epoch_losses['adversarial']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
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


# =============================================================================
# ABLATION STUDY RUNNER
# =============================================================================

def run_ablation_study(filepath, device='cpu'):
    """
    Run ablation study comparing baseline vs optimized variants
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Comparing Baseline and Optimizations")
    print("="*80)
    
    # Load data once
    train_df, val_df, test_df = load_and_split_data(filepath)
    
    train_dataset = DepressionDataset(train_df, fit_scaler=True)
    val_dataset = DepressionDataset(val_df, scaler=train_dataset.scaler, 
                                    treatment_map=train_dataset.treatment_map)
    test_dataset = DepressionDataset(test_df, scaler=train_dataset.scaler,
                                     treatment_map=train_dataset.treatment_map)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    cat_embeddings = CategoricalEmbedding(train_dataset.cat_mappings, embedding_dim=8).to(device)
    
    n_numeric = len(train_dataset.numeric_features)
    n_categorical_features = len(train_dataset.categorical_features)
    input_dim = n_numeric + n_categorical_features * 8
    
    # Configurations to test
    configs = {
        'Baseline': {
            'use_attention': False,
            'use_contrastive': False,
            'use_adversarial': False,
            'use_learned_schedule': False,
            'use_residual': False
        },
        'Attention': {
            'use_attention': True,
            'use_contrastive': False,
            'use_adversarial': False,
            'use_learned_schedule': False,
            'use_residual': False
        },
        'Contrastive': {
            'use_attention': False,
            'use_contrastive': True,
            'use_adversarial': False,
            'use_learned_schedule': False,
            'use_residual': False
        },
        'Adversarial': {
            'use_attention': False,
            'use_contrastive': False,
            'use_adversarial': True,
            'use_learned_schedule': False,
            'use_residual': False
        },
        'Residual': {
            'use_attention': False,
            'use_contrastive': False,
            'use_adversarial': False,
            'use_learned_schedule': False,
            'use_residual': True
        },
        'All_Optimizations': {
            'use_attention': True,
            'use_contrastive': True,
            'use_adversarial': True,
            'use_learned_schedule': True,
            'use_residual': True
        }
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing Configuration: {config_name}")
        print(f"{'='*80}")
        print(f"Parameters: {config_params}")
        
        # Initialize model
        model = OptimizedTreatmentEffectModel(
            input_dim=input_dim,
            n_treatments=train_dataset.n_treatments,
            stable_dim=128,
            confound_dim=128,
            device=device,
            **config_params
        )
        
        # Train (with reduced epochs for ablation study)
        try:
            model.train_stage1_representation(train_loader, val_loader, cat_embeddings,
                                             n_epochs=50, lr=1e-3, patience=10)
            model.train_stage2_outcome(train_loader, val_loader, cat_embeddings,
                                       n_epochs=50, lr=1e-3, patience=10)
            model.train_stage3_diffusion(train_loader, val_loader, cat_embeddings,
                                        n_epochs=50, lr=1e-3, patience=10)
            
            # Evaluate
            config_results = evaluate_model(model, test_loader, cat_embeddings, device)
            results[config_name] = config_results
            
        except Exception as e:
            print(f"Error training {config_name}: {str(e)}")
            results[config_name] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Configuration':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*80)
    
    for config_name, config_results in results.items():
        if 'error' not in config_results:
            print(f"{config_name:<25} "
                  f"{config_results['factual_mse']:<12.4f} "
                  f"{config_results['factual_rmse']:<12.4f} "
                  f"{config_results['factual_mae']:<12.4f} "
                  f"{config_results['factual_r2']:<12.4f}")
        else:
            print(f"{config_name:<25} ERROR: {config_results['error']}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    
    import os
    output_dir = None
    possible_dirs = ['results', '../results', '.']
    for dir_path in possible_dirs:
        try:
            if dir_path != '.' and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            output_dir = dir_path
            break
        except:
            continue
    if output_dir is None:
        output_dir = '.'
    
    results_file = os.path.join(output_dir, 'ablation_results.csv')
    results_df.to_csv(results_file)
    print(f"\nAblation results saved to {os.path.abspath(results_file)}")
    
    return results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try multiple possible locations
    import os
    possible_paths = [
        'data_generated.csv',
        '../data/data_generated.csv',
        'data/data_generated.csv',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'data_generated.csv'),
        '/mnt/user-data/uploads/data_generated.csv',
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = os.path.abspath(path)
            print(f"Found data file: {filepath}")
            break
    
    if filepath is None:
        raise FileNotFoundError(
            "Could not find data_generated.csv. "
            "Please place it in the same directory or in ../data/ folder"
        )
    
    results = run_ablation_study(filepath, device=device)
