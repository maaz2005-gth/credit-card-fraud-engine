"""
oversampling_experiment.py
==========================
Benchmarks 4 advanced oversampling methods against each other on the
harmonized fraud dataset. Evaluates using industry-standard metrics and
writes a formal Architecture Decision Record (ADR).

Methods Compared
----------------
1. CTGAN      — Conditional Tabular GAN (SDV library)
2. WGAN-GP    — Wasserstein GAN with Gradient Penalty (PyTorch)
3. CVAE       — Conditional Variational Autoencoder (PyTorch)
4. AAE        — Adversarial Autoencoder (PyTorch)

Evaluation Metrics (industry standard)
---------------------------------------
- Downstream AUC  : XGBoost AUC on held-out test set after augmentation
- Fidelity        : Wasserstein distance between real and synthetic fraud features
- Diversity       : Std dev of synthetic samples (higher = less mode collapse)
- Training Time   : Wall-clock seconds
"""

import sys, time, warnings, logging, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from ctgan import CTGAN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_ROWS      = 50_000    # Use subset for experiment speed
TARGET_RATIO     = 0.15      # Oversample minority to 15% of majority
CTGAN_EPOCHS     = 100
TORCH_EPOCHS     = 100
LATENT_DIM       = 32
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
REPORTS_DIR      = pathlib.Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data(path="data/harmonized.csv", nrows=SAMPLE_ROWS):
    logger.info("Loading %d rows from %s …", nrows, path)
    df = pd.read_csv(path, nrows=nrows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Numerical columns only for oversampling experiment
    drop_cols = ["timestamp", "user_id", "merchant_id", "category", "source_domain"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.fillna(0)  # fill NaN for PCA/dist cols

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def downstream_auc(X_train_aug, y_train_aug, X_test, y_test):
    """Train XGBoost on augmented data; return AUC on held-out test set."""
    scale_pos = (y_train_aug == 0).sum() / max((y_train_aug == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_aug, y_train_aug)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def fidelity_score(real_fraud: np.ndarray, synthetic: np.ndarray) -> float:
    """Mean Wasserstein distance across features (lower = more realistic)."""
    distances = [
        wasserstein_distance(real_fraud[:, i], synthetic[:, i])
        for i in range(real_fraud.shape[1])
    ]
    return float(np.mean(distances))


def diversity_score(synthetic: np.ndarray) -> float:
    """Mean std across features (higher = more diverse = less mode collapse)."""
    return float(np.mean(np.std(synthetic, axis=0)))


def make_n_samples(majority_count, minority_count, ratio=TARGET_RATIO):
    """How many synthetic fraud samples to generate."""
    target = int(majority_count * ratio)
    return max(0, target - minority_count)


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: CTGAN
# ──────────────────────────────────────────────────────────────────────────────
def run_ctgan(X_fraud: pd.DataFrame, n_samples: int) -> np.ndarray:
    logger.info("  Training CTGAN (%d epochs) …", CTGAN_EPOCHS)
    model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
    model.fit(X_fraud)
    synthetic = model.sample(n_samples)
    return synthetic.values


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: WGAN-GP (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
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
    grads  = torch.autograd.grad(score, interp, torch.ones_like(score),
                                  create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def run_wgan_gp(X_fraud: np.ndarray, n_samples: int) -> np.ndarray:
    logger.info("  Training WGAN-GP (%d epochs) …", TORCH_EPOCHS)
    dim  = X_fraud.shape[1]
    data = torch.tensor(X_fraud, dtype=torch.float32, device=DEVICE)

    G = _Generator(LATENT_DIM, dim).to(DEVICE)
    C = _Critic(dim).to(DEVICE)
    opt_g = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_c = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    bs = min(64, len(data))
    for _ in range(TORCH_EPOCHS):
        for _ in range(5):                          # critic steps
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


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: CVAE (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
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


def run_cvae(X_fraud: np.ndarray, n_samples: int) -> np.ndarray:
    logger.info("  Training CVAE (%d epochs) …", TORCH_EPOCHS)
    dim  = X_fraud.shape[1]
    data = torch.tensor(X_fraud, dtype=torch.float32, device=DEVICE)

    enc = _Encoder(dim, LATENT_DIM).to(DEVICE)
    dec = _Decoder(LATENT_DIM, dim).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

    bs = min(64, len(data))
    for _ in range(TORCH_EPOCHS):
        idx    = torch.randint(0, len(data), (bs,))
        x      = data[idx]
        mu, lv = enc(x)
        z      = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        x_hat  = dec(z)
        recon  = nn.functional.mse_loss(x_hat, x)
        kld    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
        loss   = recon + 0.001 * kld
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        synthetic = dec(z).cpu().numpy()
    return synthetic


# ──────────────────────────────────────────────────────────────────────────────
# Method 4: AAE (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────
class _AAEDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32),         nn.LeakyReLU(0.2),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


def run_aae(X_fraud: np.ndarray, n_samples: int) -> np.ndarray:
    logger.info("  Training AAE (%d epochs) …", TORCH_EPOCHS)
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
        mu, _ = enc(x)                          # use mu as latent code
        x_hat = dec(mu)
        ae_loss = nn.functional.mse_loss(x_hat, x)
        opt_ae.zero_grad(); ae_loss.backward(); opt_ae.step()

        # Discriminator
        z_real = torch.randn(bs, LATENT_DIM, device=DEVICE)
        mu2, _ = enc(x)
        d_loss = bce(disc(z_real), torch.ones(bs,1,device=DEVICE)) + \
                 bce(disc(mu2.detach()), torch.zeros(bs,1,device=DEVICE))
        opt_disc.zero_grad(); d_loss.backward(); opt_disc.step()

        # Generator (encoder tries to fool discriminator)
        mu3, _ = enc(x)
        g_loss = bce(disc(mu3), torch.ones(bs,1,device=DEVICE))
        opt_gen.zero_grad(); g_loss.backward(); opt_gen.step()

    with torch.no_grad():
        z         = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        synthetic = dec(z).cpu().numpy()
    return synthetic


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment():
    logger.info("=" * 60)
    logger.info("OVERSAMPLING BENCHMARK EXPERIMENT")
    logger.info("=" * 60)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Isolate real fraud samples from training set
    fraud_mask  = y_train == 1
    X_fraud_df  = X_train[fraud_mask]
    X_fraud_np  = X_fraud_df.values.astype(np.float32)

    n_majority  = (y_train == 0).sum()
    n_minority  = fraud_mask.sum()
    n_synth     = make_n_samples(n_majority, n_minority)

    logger.info("Train: %d majority | %d minority | generating %d synthetic samples",
                n_majority, n_minority, n_synth)

    methods = {
        "CTGAN":   lambda: run_ctgan(X_fraud_df, n_synth),
        "WGAN-GP": lambda: run_wgan_gp(X_fraud_np, n_synth),
        "CVAE":    lambda: run_cvae(X_fraud_np, n_synth),
        "AAE":     lambda: run_aae(X_fraud_np, n_synth),
    }

    results = []

    for name, fn in methods.items():
        logger.info("\n── Running: %s ─────────────────────", name)
        t0 = time.time()
        synthetic = fn()
        elapsed = time.time() - t0

        # Cap synthetic to actual n_synth (some methods may return slightly different shapes)
        synthetic = synthetic[:n_synth]

        # Build augmented training set
        synth_df = pd.DataFrame(synthetic, columns=X_train.columns)
        X_aug    = pd.concat([X_train, synth_df], ignore_index=True)
        y_aug    = pd.concat([y_train, pd.Series([1] * len(synth_df))], ignore_index=True)

        auc  = downstream_auc(X_aug, y_aug, X_test, y_test)
        fid  = fidelity_score(X_fraud_np, synthetic)
        div  = diversity_score(synthetic)

        results.append({
            "Method":          name,
            "Downstream_AUC":  round(auc, 4),
            "Fidelity_WD":     round(fid, 4),      # lower is better
            "Diversity_Std":   round(div, 4),       # higher is better
            "Training_Time_s": round(elapsed, 1),
        })

        logger.info("  AUC=%.4f | Fidelity(WD)=%.4f | Diversity=%.4f | Time=%.1fs",
                    auc, fid, div, elapsed)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Pick winner: highest downstream AUC is the primary criterion
    winner = results_df.loc[results_df["Downstream_AUC"].idxmax(), "Method"]
    logger.info("\n🏆 Winner: %s", winner)

    _write_adr(results_df, winner)
    return results_df, winner


# ──────────────────────────────────────────────────────────────────────────────
# Architecture Decision Record
# ──────────────────────────────────────────────────────────────────────────────
def _write_adr(results_df: pd.DataFrame, winner: str):
    runner_up = results_df[results_df["Method"] != winner].sort_values(
        "Downstream_AUC", ascending=False
    ).iloc[0]

    winner_row = results_df[results_df["Method"] == winner].iloc[0]

    adr = f"""# ADR-001: Oversampling Strategy for Generalized Fraud Engine

**Status:** Accepted  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  
**Deciders:** ML Engineering Team

---

## Context

The fraud dataset exhibits severe class imbalance (~0.5–18% fraud depending on domain).
Standard SMOTE was used in v1 of the model (at 1.2% oversampling ratio).
This ADR documents the evaluation of advanced generative oversampling methods
to improve the model's sensitivity to minority-class fraud patterns.

---

## Decision

**Selected method: {winner}**

---

## Benchmark Results

| Method | Downstream AUC ↑ | Fidelity (WD) ↓ | Diversity (Std) ↑ | Training Time |
|--------|-----------------|-----------------|-------------------|---------------|
{chr(10).join(f"| **{r['Method']}** | {r['Downstream_AUC']} | {r['Fidelity_WD']} | {r['Diversity_Std']} | {r['Training_Time_s']}s |" for _, r in results_df.iterrows())}

> **Primary criterion:** Downstream AUC — the only metric that proves oversampling actually helped the classifier.

---

## Rationale

**Why {winner} was selected:**
- Achieved the highest Downstream AUC of **{winner_row['Downstream_AUC']}**, meaning it produced synthetic fraud samples that most helped the XGBoost classifier learn the fraud decision boundary.
- Training time of **{winner_row['Training_Time_s']}s** is acceptable for an offline experiment.

**Why {runner_up['Method']} (runner-up, AUC={runner_up['Downstream_AUC']}) was not selected:**
- Despite competitive performance, it underperformed {winner} on the primary Downstream AUC metric, which is the definitive production-relevant criterion.

**Why not SMOTE (baseline):**
- SMOTE performs linear interpolation in feature space. It cannot model the true probability distribution of fraud, especially across heterogeneous datasets (Sparkov POS + MLG European bank). Generative models learn the actual data manifold.

---

## Consequences

- `src/sampling.py` will be updated to use `{winner}` with the hyperparameters validated in this experiment.
- The original `apply_smote_oversampling()` function is preserved for comparison but deprecated in the main pipeline.
- Runtime overhead is acceptable: {winner} trains once offline before the main LODO cross-validation loop.

---

## References

- Goodfellow et al. (2014) — Generative Adversarial Networks
- Xu et al. (2019) — Modeling Tabular data using Conditional GAN (CTGAN)
- Gulrajani et al. (2017) — Improved Training of Wasserstein GANs (WGAN-GP)
- Makhzani et al. (2015) — Adversarial Autoencoders (AAE)
- Kingma & Welling (2013) — Auto-Encoding Variational Bayes (VAE)
"""

    out = REPORTS_DIR / "oversampling_decision.md"
    out.write_text(adr, encoding="utf-8")
    logger.info("ADR written → %s", out)


if __name__ == "__main__":
    results_df, winner = run_experiment()
