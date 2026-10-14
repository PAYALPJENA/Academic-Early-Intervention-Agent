"""PyTorch tabular models wrapped as scikit-learn classifiers (CPU or CUDA).

MLPClassifierTorch      : feed-forward network on the 55 one-hot + scaled features.
FTTransformerClassifier : FT-Transformer (Gorishniy et al., 2021). Each numeric feature gets a learned
                          linear token, each categorical feature an embedding token; a [CLS] token is
                          prepended and passed through pre-norm Transformer blocks. Simplification:
                          standard GELU feed-forward instead of ReGLU.
"""

import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

_n_jobs = int(__import__("os").environ.get("PIPELINE_N_JOBS", "-1"))
torch.set_num_threads(_n_jobs if _n_jobs > 0 else max(1, torch.get_num_threads()))


def _set_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class _TorchClassifierBase(ClassifierMixin, BaseEstimator):
    """Shared training loop: AdamW, cross-entropy, fixed number of epochs, mini-batches."""

    def _build(self, X):
        raise NotImplementedError

    def _to_tensor(self, X):
        return torch.as_tensor(np.asarray(X, dtype=np.float32))

    def fit(self, X, y, eval_set=None, eval_callback=None):
        self.device_ = torch.device(self.device)
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        _set_seed(self.random_state, self.device_)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        X_t = self._to_tensor(X).to(self.device_)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=self.device_)
        self.model_ = self._build(X).to(self.device_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator(device=self.device_).manual_seed(self.random_state)
        n = len(y_t)
        self.epoch_seconds_ = []
        for epoch in range(self.epochs):
            t0 = time.time()
            self.model_.train()
            perm = torch.randperm(n, generator=gen, device=self.device_)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(X_t[idx]), y_t[idx])
                loss.backward()
                opt.step()
            self.epoch_seconds_.append(time.time() - t0)
            if eval_callback is not None:
                eval_callback(epoch + 1, self)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.model_.eval()
        X_t = self._to_tensor(X).to(self.device_)
        out = [torch.softmax(self.model_(X_t[i:i + 8192]), dim=1) for i in range(0, len(X_t), 8192)]
        return torch.cat(out).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------- MLP

class _MLP(nn.Module):
    def __init__(self, d_in, hidden, n_layers, dropout, n_out):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifierTorch(_TorchClassifierBase):
    def __init__(self, hidden=256, n_layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-5,
                 batch_size=1024, epochs=20, random_state=0, device="cpu"):
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def _build(self, X):
        return _MLP(X.shape[1], self.hidden, self.n_layers, self.dropout, len(self.classes_))


# ---------------------------------------------------------------- FT-Transformer

class _FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token, n_blocks, n_heads, dropout, n_out):
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.num_weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.num_bias, a=5 ** 0.5)
        offsets = np.concatenate([[0], np.cumsum(cat_cardinalities)[:-1]])
        self.register_buffer("cat_offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.cat_embed = nn.Embedding(int(sum(cat_cardinalities)), d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token ** -0.5)
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=int(d_token * 4 / 3) * 2,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_out))

    def forward(self, x):
        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:].long() + self.cat_offsets
        tokens = torch.cat([
            self.cls.expand(len(x), -1, -1),
            x_num.unsqueeze(-1) * self.num_weight + self.num_bias,
            self.cat_embed(x_cat),
        ], dim=1)
        return self.head(self.blocks(tokens)[:, 0])


class FTTransformerClassifier(_TorchClassifierBase):
    """Expects input columns [n_num numeric (scaled) | categorical codes (0 = unknown)]."""

    def __init__(self, n_num=21, d_token=64, n_blocks=2, n_heads=8, dropout=0.1, lr=1e-3,
                 weight_decay=1e-5, batch_size=512, epochs=10, random_state=0, device="cpu"):
        self.n_num = n_num
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def _build(self, X):
        X = np.asarray(X)
        self.cat_cardinalities_ = [int(X[:, j].max()) + 2 for j in range(self.n_num, X.shape[1])]
        return _FTTransformer(self.n_num, self.cat_cardinalities_, self.d_token, self.n_blocks, self.n_heads,
                              self.dropout, len(self.classes_))

    def _to_tensor(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        if hasattr(self, "cat_cardinalities_"):  # unseen categories -> index 0
            for k, j in enumerate(range(self.n_num, X.shape[1])):
                col = X[:, j]
                col[(col < 0) | (col >= self.cat_cardinalities_[k] - 1)] = -1
        X[:, self.n_num:] += 1  # OrdinalEncoder unknown (-1) -> 0
        return torch.as_tensor(X)
