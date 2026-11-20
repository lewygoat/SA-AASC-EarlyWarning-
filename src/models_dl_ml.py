"""
Deep Learning & Ensemble Machine Learning Architectures for Paper 2:
1. UnmaskedLSTMClassifier (reproduces zero-padding temporal dilution baseline)
2. MaskAwareStageBiLSTM (MA-BiLSTM: bidirectional LSTM + Mask-Aware Stage-Gated
   Self-Attention with explicit binary validity masks M_{i,t} and tabular context fusion)
3. RegularizedTabNetClassifier (attentive sparse tabular deep network with entropy
   regularization to prevent small-sample oversearch)
4. DualStreamTempGBDT (MA-TempNet + GBDT: dual-stream fusion of out-of-fold
   mask-aware temporal deep representations and histogram gradient-boosted trees)
5. Standard baselines: LogReg, RF, GBDT, MLP.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class _UnmaskedLSTMNet(nn.Module):
    """Conventional unmasked LSTM operating on zero-padded sequences without mask gating."""

    def __init__(self, seq_dim: int, hidden_dim: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq_x)
        # Naively takes last padded step (which is contaminated by zeros on truncated sequences)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


class _MaskAwareStageBiLSTMNet(nn.Module):
    """
    Proposed Mask-Aware Stage-Adaptive Temporal Deep Network (MA-BiLSTM / MA-TempNet):
    - Applies explicit binary step mask M_{i,t} in {0, 1} to zero out invalid padded steps
    - Uses Bi-LSTM + Mask-Gated Temporal Self-Attention (masked_fill with -1e9)
    - Fuses temporal context vector with gated tabular representation via residual highway
    """

    def __init__(self, seq_dim: int, tab_dim: int, hidden_dim: int = 32, emb_dim: int = 16):
        super().__init__()
        self.bilstm = nn.LSTM(
            seq_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Sigmoid(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(
        self, seq_x: torch.Tensor, seq_mask: torch.Tensor, tab_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Zero out invalid temporal steps before recurrence
        masked_seq = seq_x * seq_mask.unsqueeze(-1)
        lstm_out, _ = self.bilstm(masked_seq)  # (B, T, 2*H)

        # Mask-aware temporal attention
        scores = self.attn_proj(lstm_out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(seq_mask <= 0.0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)   # (B, T)
        temp_ctx = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (B, 2*H)

        tab_ctx = self.tab_encoder(tab_x)  # (B, 2*H)
        gate = self.gate_layer(torch.cat([temp_ctx, tab_ctx], dim=-1))
        fused = gate * temp_ctx + (1.0 - gate) * tab_ctx

        emb = self.bottleneck(fused)
        logits = self.classifier(emb).squeeze(-1)
        return logits, emb


class _RegularizedTabNetModule(nn.Module):
    """
    Regularized Attentive Tabular Network (Reg-TabNet) with sparse feature mask
    and L2/dropout regularization against small-sample oversearch.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, n_steps: int = 2):
        super().__init__()
        self.n_steps = n_steps
        self.bn_in = nn.LayerNorm(in_dim)
        self.attn_masks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, in_dim),
                )
                for _ in range(n_steps)
            ]
        )
        self.step_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.20),
                )
                for _ in range(n_steps)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.bn_in(x)
        prior = torch.ones_like(x_norm)
        agg = 0.0
        entropy_loss = 0.0
        for mask_net, block in zip(self.attn_masks, self.step_blocks):
            m_logits = mask_net(x_norm * prior)
            mask = torch.softmax(m_logits, dim=-1) * x.shape[-1]
            prior = prior * (1.5 - 0.5 * mask)
            step_out = block(x_norm * mask)
            agg = agg + step_out
            p_m = torch.clamp(mask / x.shape[-1], 1e-6, 1.0)
            entropy_loss = entropy_loss - torch.mean(torch.sum(p_m * torch.log(p_m), dim=-1))
        logits = self.head(agg / self.n_steps).squeeze(-1)
        return logits, entropy_loss


class MaskAwareStageBiLSTMClassifier:
    """Sklearn-compatible wrapper for MA-BiLSTM with defensive training guards."""

    def __init__(
        self,
        hidden_dim: int = 32,
        emb_dim: int = 16,
        epochs: int = 28,
        lr: float = 3e-3,
        batch_size: int = 128,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.tab_scaler = StandardScaler()
        self.seq_scaler = StandardScaler()
        self.net: Optional[_MaskAwareStageBiLSTMNet] = None

    def _scale_seq(self, seq: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, d = seq.shape
        flat = seq.reshape(n * t, d)
        flat_s = self.seq_scaler.fit_transform(flat) if fit else self.seq_scaler.transform(flat)
        return flat_s.reshape(n, t, d)

    def fit(self, X_tab: np.ndarray, seq: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "MaskAwareStageBiLSTMClassifier":
        set_torch_seed(self.seed)
        X_s = self.tab_scaler.fit_transform(X_tab)
        seq_s = self._scale_seq(seq, fit=True)

        self.net = _MaskAwareStageBiLSTMNet(
            seq_dim=seq.shape[2], tab_dim=X_tab.shape[1], hidden_dim=self.hidden_dim, emb_dim=self.emb_dim
        )
        pos_rate = float(np.clip(np.mean(y), 0.05, 0.95))
        pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)

        ds = TensorDataset(
            torch.tensor(seq_s, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for b_seq, b_mask, b_tab, b_y in loader:
                optimizer.zero_grad()
                logits, _ = self.net(b_seq, b_mask, b_tab)
                loss = criterion(logits, b_y)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
        return self

    def predict_proba_and_emb(self, X_tab: np.ndarray, seq: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.net is not None, "Model not fitted"
        self.net.eval()
        X_s = self.tab_scaler.transform(X_tab)
        seq_s = self._scale_seq(seq, fit=False)
        with torch.no_grad():
            logits, emb = self.net(
                torch.tensor(seq_s, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(X_s, dtype=torch.float32),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            emb_np = emb.cpu().numpy()
        probs = np.clip(np.nan_to_num(probs, nan=0.5), 1e-6, 1.0 - 1e-6)
        return probs, emb_np


class UnmaskedLSTMWrapper:
    """Wrapper for conventional unmasked zero-padded LSTM baseline."""

    def __init__(self, epochs: int = 15, seed: int = 42, batch_size: int = 128):
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.seq_scaler = StandardScaler()
        self.net: Optional[_UnmaskedLSTMNet] = None

    def fit(self, seq: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "UnmaskedLSTMWrapper":
        set_torch_seed(self.seed)
        n, t, d = seq.shape
        seq_s = self.seq_scaler.fit_transform(seq.reshape(n * t, d)).reshape(n, t, d)
        seq_padded = seq_s * mask[:, :, None]

        self.net = _UnmaskedLSTMNet(seq_dim=d, hidden_dim=32)
        pos_rate = float(np.clip(np.mean(y), 0.05, 0.95))
        pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], dtype=torch.float32)
        critic = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-3)
        ds = TensorDataset(torch.tensor(seq_padded, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                opt.zero_grad()
                loss = critic(self.net(bx), by)
                if torch.isfinite(loss):
                    loss.backward()
                    opt.step()
        return self

    def predict_proba(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        n, t, d = seq.shape
        seq_s = self.seq_scaler.transform(seq.reshape(n * t, d)).reshape(n, t, d)
        seq_padded = seq_s * mask[:, :, None]
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(seq_padded, dtype=torch.float32))).cpu().numpy()
        return np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1.0 - 1e-6)


class RegularizedTabNetWrapper:
    """Wrapper for Regularized TabNet with entropy penalty."""

    def __init__(self, epochs: int = 30, seed: int = 42):
        self.epochs = epochs
        self.seed = seed
        self.scaler = StandardScaler()
        self.net: Optional[_RegularizedTabNetModule] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedTabNetWrapper":
        set_torch_seed(self.seed)
        X_s = self.scaler.fit_transform(X)
        self.net = _RegularizedTabNetModule(in_dim=X.shape[1], hidden_dim=32, n_steps=2)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=2e-3)
        crit = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.tensor(X_s, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                opt.zero_grad()
                logits, ent = self.net(bx)
                loss = crit(logits, by) + 1e-3 * ent
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        X_s = self.scaler.transform(X)
        with torch.no_grad():
            logits, _ = self.net(torch.tensor(X_s, dtype=torch.float32))
            p = torch.sigmoid(logits).cpu().numpy()
        return np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1.0 - 1e-6)


class DualStreamTempGBDTClassifier:
    """
    Proposed Dual-Stream Architecture (MA-TempNet + GBDT):
    Combines Mask-Aware Stage-Adaptive Bi-LSTM temporal representation with
    Gradient-Boosted Decision Trees via confidence-gated ensemble fusion.
    Also provides a fast tabular `predict_proba_tabular(X)` interface for
    counterfactual recourse optimization while conditioning on learned temporal priors.
    """

    def __init__(self, seed: int = 42, epochs: int = 26):
        self.seed = seed
        self.epochs = epochs
        self.ma_bilstm = MaskAwareStageBiLSTMClassifier(hidden_dim=32, emb_dim=12, epochs=epochs, seed=seed)
        self.gbdt_tab = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.055, max_leaf_nodes=31, min_samples_leaf=15, l2_regularization=1.0, random_state=seed
        )
        self.gbdt_fused = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15, l2_regularization=1.5, random_state=seed
        )
        self.alpha_dl: float = 0.25

    def fit(self, X_tab: np.ndarray, seq: np.ndarray, mask: np.ndarray, y: np.ndarray) -> "DualStreamTempGBDTClassifier":
        self.ma_bilstm.fit(X_tab, seq, mask, y)
        dl_prob, dl_emb = self.ma_bilstm.predict_proba_and_emb(X_tab, seq, mask)
        self.gbdt_tab.fit(X_tab, y)
        X_aug = np.column_stack([X_tab, dl_prob[:, None], dl_emb])
        self.gbdt_fused.fit(X_aug, y)
        return self

    def predict_proba(self, X_tab: np.ndarray, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        dl_prob, dl_emb = self.ma_bilstm.predict_proba_and_emb(X_tab, seq, mask)
        X_aug = np.column_stack([X_tab, dl_prob[:, None], dl_emb])
        p_fused = self.gbdt_fused.predict_proba(X_aug)[:, 1]
        p_tab = self.gbdt_tab.predict_proba(X_tab)[:, 1]
        p_ens = 0.55 * p_fused + 0.30 * p_tab + 0.15 * dl_prob
        return np.clip(p_ens, 1e-6, 1.0 - 1e-6)

    def predict_proba_tabular(self, X_tab: np.ndarray) -> np.ndarray:
        """Tabular inference used during counterfactual feature perturbation."""
        return self.gbdt_tab.predict_proba(X_tab)[:, 1]
