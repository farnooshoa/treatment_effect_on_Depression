import torch
import torch.nn as nn
import torch.nn.functional as F


from model import (
    CategoricalEmbeddingBlock,
    Encoder,
    OutcomeModel,
    PropensityHead
)
from data_prepare import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    get_category_sizes,
    load_and_create_outcome,
    patient_level_split,
    fit_numeric_standardizer,
    apply_numeric_standardizer,
    fit_categorical_encoders,
    apply_categorical_encoders,
    fit_treatment_mapping,
    apply_treatment_mapping,
    build_model_inputs,
    patient_level_two_folds
)

from nuisance import(
    NuisanceXEncoder,
    PropensityNuisance,
    OutcomeNuisance
)
def make_stochastic_view(x_num, noise_std=0.01, dropout_p=0.1):
    # Gaussian noise
    noise = torch.randn_like(x_num) * noise_std
    x_noisy = x_num + noise

    # Feature dropout
    dropout_mask = torch.rand_like(x_noisy) > dropout_p
    x_view = x_noisy * dropout_mask

    return x_view

def compute_stability_loss(train_df, category_sizes):
    x_num, x_cat, _, _ = build_model_inputs(train_df)

    # Models
    emb = CategoricalEmbeddingBlock(category_sizes)
    encoder = Encoder(input_dim=6 + 8 * len(category_sizes))

    # Two stochastic numeric views
    x_num_v1 = make_stochastic_view(x_num)
    x_num_v2 = make_stochastic_view(x_num)

    # Categorical embeddings (same for both views)
    x_cat_emb = emb(x_cat)

    # Full inputs
    x1 = torch.cat([x_num_v1, x_cat_emb], dim=1)
    x2 = torch.cat([x_num_v2, x_cat_emb], dim=1)

    # Encode
    S1, _ = encoder(x1)
    S2, _ = encoder(x2)

    # Stability loss
    loss = torch.mean((S1 - S2) ** 2)
    return loss


def compute_factual_loss(train_df, category_sizes, num_treatments):
    # Build tensors
    x_num, x_cat, t, y = build_model_inputs(train_df)

    # Models
    emb = CategoricalEmbeddingBlock(category_sizes)
    encoder = Encoder(input_dim=6 + 8 * len(category_sizes))
    outcome_model = OutcomeModel(ds=64, dc=64, num_treatments=num_treatments)

    # Forward
    x_cat_emb = emb(x_cat)
    x = torch.cat([x_num, x_cat_emb], dim=1)

    S, C = encoder(x)
    y_hat = outcome_model(S, C, t)

    # Loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_hat, y)

    return loss



def compute_propensity_loss(train_df, category_sizes, num_treatments):
    x_num, x_cat, t, _ = build_model_inputs(train_df)

    # Models
    emb = CategoricalEmbeddingBlock(category_sizes)
    encoder = Encoder(input_dim=6 + 8 * len(category_sizes))
    prop_head = PropensityHead(dc=64, num_treatments=num_treatments)

    # Two stochastic views
    x_num_v1 = make_stochastic_view(x_num)
    x_num_v2 = make_stochastic_view(x_num)

    x_cat_emb = emb(x_cat)

    x1 = torch.cat([x_num_v1, x_cat_emb], dim=1)
    x2 = torch.cat([x_num_v2, x_cat_emb], dim=1)

    # Encode
    _, C1 = encoder(x1)
    _, C2 = encoder(x2)

    # Predict treatment
    logits1 = prop_head(C1)
    logits2 = prop_head(C2)

    # Cross-entropy
    loss1 = F.cross_entropy(logits1, t)
    loss2 = F.cross_entropy(logits2, t)

    return 0.5 * (loss1 + loss2)

def rbf_kernel(x, y, gamma=1.0):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-gamma * dist)


def mmd(x, y, gamma=1.0):
    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

def compute_balance_loss(train_df, category_sizes):
    x_num, x_cat, t, _ = build_model_inputs(train_df)

    # Models
    emb = CategoricalEmbeddingBlock(category_sizes)
    encoder = Encoder(input_dim=6 + 8 * len(category_sizes))

    # Single stochastic view is enough here
    x_num_v = make_stochastic_view(x_num)
    x_cat_emb = emb(x_cat)
    x = torch.cat([x_num_v, x_cat_emb], dim=1)

    _, C = encoder(x)

    loss = 0.0
    treatments = t.unique()

    count = 0
    for i in range(len(treatments)):
        for j in range(i + 1, len(treatments)):
            Ci = C[t == treatments[i]]
            Cj = C[t == treatments[j]]

            if len(Ci) > 1 and len(Cj) > 1:
                loss += mmd(Ci, Cj)
                count += 1

    return loss / max(count, 1)

def variance_loss(z, eps=1e-4):
    var = z.var(dim=0)
    return torch.mean(F.relu(1.0 - var + eps))

def variance_loss(z, target=0.5, eps=1e-4):
    var = z.var(dim=0)
    return torch.mean(F.relu(target - var + eps))


def decorrelation_loss(z):
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / z.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()

def compute_anticollapse_loss(train_df, category_sizes):
    x_num, x_cat, _, _ = build_model_inputs(train_df)

    emb = CategoricalEmbeddingBlock(category_sizes)
    encoder = Encoder(input_dim=6 + 8 * len(category_sizes))

    x_num_v = make_stochastic_view(x_num)
    x_cat_emb = emb(x_cat)
    x = torch.cat([x_num_v, x_cat_emb], dim=1)

    S, _ = encoder(x)

    return variance_loss(S) + decorrelation_loss(S)



BASELINE_TID = 0

def dr_pseudo_outcome(mu_hat, p_hat, t, y, baseline_tid=BASELINE_TID, eps=0.05, winsor=10.0):
    """
    mu_hat: [N, K] predicted outcomes for each treatment
    p_hat:  [N, K] propensity probabilities
    t:      [N] observed treatment ids
    y:      [N] observed outcomes
    returns tau: [N, K] DR pseudo-outcome vector (relative to baseline)
    """
    N, K = mu_hat.shape
    p_hat = torch.clamp(p_hat, eps, 1.0 - eps)

    mu_b = mu_hat[:, baseline_tid]  # [N]
    p_b  = p_hat[:, baseline_tid]   # [N]

    tau = torch.zeros_like(mu_hat)

    # residuals under observed treatment and baseline
    # for k loop: compute DR estimate vs baseline
    for k in range(K):
        mu_k = mu_hat[:, k]
        p_k  = p_hat[:, k]

        I_k = (t == k).float()
        I_b = (t == baseline_tid).float()

        tau_k = (mu_k - mu_b) + I_k * (y - mu_k) / p_k - I_b * (y - mu_b) / p_b
        tau[:, k] = tau_k

    # by definition, baseline effect should be ~0
    tau[:, baseline_tid] = 0.0

    # winsorize (simple symmetric clip)
    tau = torch.clamp(tau, -winsor, winsor)
    return tau

def train_nuisance_on_fold(fold_train, fold_eval, category_sizes, num_treatments, epochs=5):
    x_num_tr, x_cat_tr, t_tr, y_tr = build_model_inputs(fold_train)
    x_num_ev, x_cat_ev, t_ev, y_ev = build_model_inputs(fold_eval)

    x_encoder = NuisanceXEncoder(category_sizes)
    prop_model = PropensityNuisance(x_encoder, num_treatments=num_treatments)
    out_model = OutcomeNuisance(x_encoder, num_treatments=num_treatments)

    opt = torch.optim.Adam(
        list(prop_model.parameters()) + list(out_model.parameters()),
        lr=1e-3
    )

    for ep in range(epochs):
        logits = prop_model(x_num_tr, x_cat_tr)
        y_hat = out_model(x_num_tr, x_cat_tr, t_tr)

        loss_prop = F.cross_entropy(logits, t_tr)
        loss_out = F.mse_loss(y_hat, y_tr)

        loss = loss_prop + loss_out

        opt.zero_grad()
        loss.backward()
        opt.step()

    # predictions on eval fold
    with torch.no_grad():
        p_hat = F.softmax(prop_model(x_num_ev, x_cat_ev), dim=1)

        mu_hat = []
        for k in range(num_treatments):
            t_k = torch.full_like(t_ev, k)
            mu_k = out_model(x_num_ev, x_cat_ev, t_k)
            mu_hat.append(mu_k)

        mu_hat = torch.stack(mu_hat, dim=1)

    return mu_hat, p_hat



if __name__ == "__main__":
    # Load & preprocess
    df = load_and_create_outcome("data/data_generated.csv")
    train_df, _, _ = patient_level_split(df)

    num_stats = fit_numeric_standardizer(train_df, NUMERIC_COLS)
    cat_encoders = fit_categorical_encoders(train_df, CATEGORICAL_COLS)

    train_df = apply_numeric_standardizer(train_df, NUMERIC_COLS, num_stats)
    train_df = apply_categorical_encoders(train_df, CATEGORICAL_COLS, cat_encoders)

    treatment_mapping = fit_treatment_mapping(train_df)
    train_df = apply_treatment_mapping(train_df, treatment_mapping)

    category_sizes = get_category_sizes(cat_encoders, CATEGORICAL_COLS)
    num_treatments = len(treatment_mapping)

    loss = compute_factual_loss(train_df, category_sizes, num_treatments)

    
    stab_loss = compute_stability_loss(train_df, category_sizes)
    prop_loss = compute_propensity_loss(train_df, category_sizes, num_treatments)
    bal_loss = compute_balance_loss(train_df, category_sizes)
    ac_loss = compute_anticollapse_loss(train_df, category_sizes)
    foldA, foldB = patient_level_two_folds(train_df)


    mu_hat_B, p_hat_B = train_nuisance_on_fold(foldA, foldB, category_sizes, num_treatments)

    # Build tensors for Fold B
    x_num_B, x_cat_B, t_B, y_B = build_model_inputs(foldB)

    tau_B = dr_pseudo_outcome(mu_hat_B, p_hat_B, t_B, y_B)

    print("tau_B shape:", tau_B.shape)
    print("tau_B sample (first row):", tau_B[0])

    
 
 
