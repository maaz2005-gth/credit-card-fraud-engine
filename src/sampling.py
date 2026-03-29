"""
sampling.py
===========
Handles data oversampling for the fraud dataset.
Contains both WGAN-GP (GAN) and AAE (Autoencoder) oversamplers
for the final performance showdown.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
TORCH_EPOCHS = 100

# ────────────────────────────────────────────────────────
# WGAN-GP (GAN) Models
# ────────────────────────────────────────────────────────
class _Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 256),        nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, out_dim),
        )
    def forward(self, z): return self.net(z)

class _Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128),   nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x)

def _gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score  = critic(interp)
    grads  = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

def generate_wgangp_samples(X_fraud: np.ndarray, n_samples: int) -> np.ndarray:
    dim  = X_fraud.shape[1]
    data = torch.tensor(X_fraud, dtype=torch.float32, device=DEVICE)
    G = _Generator(LATENT_DIM, dim).to(DEVICE)
    C = _Critic(dim).to(DEVICE)
    opt_g = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_c = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))
    bs = min(64, len(data))
    for _ in range(TORCH_EPOCHS):
        for _ in range(5):
            idx  = torch.randint(0, len(data), (bs,))
            real = data[idx]
            z    = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake = G(z).detach()
            gp   = _gradient_penalty(C, real, fake, DEVICE)
            c_loss = C(fake).mean() - C(real).mean() + 10 * gp
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()
        z      = torch.randn(bs, LATENT_DIM, device=DEVICE)
        g_loss = -C(G(z)).mean()
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        synthetic = G(z).cpu().numpy()
    return synthetic

# ────────────────────────────────────────────────────────
# AAE (Autoencoder) Models
# ────────────────────────────────────────────────────────
class _Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.shared  = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu_head = nn.Linear(64, latent_dim)
        self.lv_head = nn.Linear(64, latent_dim)
    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.lv_head(h)

class _Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, z): return self.net(z)

class _AAEDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32),         nn.LeakyReLU(0.2),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)

def generate_aae_samples(X_fraud: np.ndarray, n_samples: int) -> np.ndarray:
    dim  = X_fraud.shape[1]
    data = torch.tensor(X_fraud, dtype=torch.float32, device=DEVICE)
    bce  = nn.BCELoss()
    enc  = _Encoder(dim, LATENT_DIM).to(DEVICE)
    dec  = _Decoder(LATENT_DIM, dim).to(DEVICE)
    disc = _AAEDiscriminator(LATENT_DIM).to(DEVICE)
    opt_ae   = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    opt_disc = optim.Adam(disc.parameters(), lr=1e-4)
    opt_gen  = optim.Adam(enc.parameters(), lr=1e-4)

    bs = min(64, len(data))
    for _ in range(TORCH_EPOCHS):
        idx   = torch.randint(0, len(data), (bs,))
        x     = data[idx]
        mu, _ = enc(x)
        x_hat = dec(mu)
        ae_loss = nn.functional.mse_loss(x_hat, x)
        opt_ae.zero_grad(); ae_loss.backward(); opt_ae.step()
        
        z_real = torch.randn(bs, LATENT_DIM, device=DEVICE)
        mu2, _ = enc(x)
        d_loss = bce(disc(z_real), torch.ones(bs,1,device=DEVICE)) + \
                 bce(disc(mu2.detach()), torch.zeros(bs,1,device=DEVICE))
        opt_disc.zero_grad(); d_loss.backward(); opt_disc.step()
        
        mu3, _ = enc(x)
        g_loss = bce(disc(mu3), torch.ones(bs,1,device=DEVICE))
        opt_gen.zero_grad(); g_loss.backward(); opt_gen.step()
        
    with torch.no_grad():
        z         = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        synthetic = dec(z).cpu().numpy()
    return synthetic

# ────────────────────────────────────────────────────────
# Core Apply Function
# ────────────────────────────────────────────────────────
def apply_oversampling(X: pd.DataFrame, y: pd.Series, method: str, target_ratio: float = 0.15):
    majority_count = (y == 0).sum()
    minority_count = (y == 1).sum()
    target_minority = int(majority_count * target_ratio)
    n_synth = max(0, target_minority - minority_count)
    if n_synth == 0: return X, y
        
    logger.info("Oversampling (%s): %d majority, %d minority. Generating %d synthetic records...", 
                method, majority_count, minority_count, n_synth)
                
    X_fraud = X[y == 1].values.astype(np.float32)
    
    if method == "GAN":
        synthetic_fraud_np = generate_wgangp_samples(X_fraud, n_synth)
    elif method == "AE":
        synthetic_fraud_np = generate_aae_samples(X_fraud, n_synth)
    else:
        raise ValueError(f"Unknown method {method}")
        
    synthetic_fraud_np = synthetic_fraud_np[:n_synth]
    synthetic_X = pd.DataFrame(synthetic_fraud_np, columns=X.columns)
    synthetic_y = pd.Series([1] * len(synthetic_X), name=y.name)
    
    X_resampled = pd.concat([X, synthetic_X], ignore_index=True)
    y_resampled = pd.concat([y, synthetic_y], ignore_index=True)
    
    idx = np.random.permutation(len(X_resampled))
    return X_resampled.iloc[idx].reset_index(drop=True), y_resampled.iloc[idx].reset_index(drop=True)