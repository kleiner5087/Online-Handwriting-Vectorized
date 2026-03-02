import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STROKE_DIM   = 3
HIDDEN_SIZE  = 512
EMBED_DIM    = 64
K_ATTN       = 10
M_MDN        = 10        # Reducido de 20: menos componentes → H(π) más fácil de mantener sano
SIGMA_REG_W  = 0.30      # Calibrado para batch≥32


class SoftAttentionWindow(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int, K: int = K_ATTN):
        super().__init__()
        self.K    = K
        self.proj = nn.Linear(hidden_size, 3 * K)
        self.kappa: torch.Tensor | None = None

        nn.init.normal_(self.proj.weight, std=0.01)
        with torch.no_grad():
            self.proj.bias.zero_()
            self.proj.bias[2 * K:].fill_(-4.0)

    def reset(self, batch_size: int, device: torch.device):
        self.kappa = torch.zeros(batch_size, self.K, device=device)

    def forward(self, h: torch.Tensor, char_embeds: torch.Tensor):
        B, U, _ = char_embeds.shape

        raw = self.proj(h)
        log_alpha, log_beta, log_delta = raw.chunk(3, dim=-1)

        alpha      = torch.exp(log_alpha)
        beta       = torch.exp(log_beta)
        self.kappa = self.kappa + torch.exp(log_delta)

        u   = torch.arange(U, device=h.device, dtype=h.dtype)
        phi = (
            alpha.unsqueeze(2)
            * torch.exp(-beta.unsqueeze(2) * (self.kappa.unsqueeze(2) - u) ** 2)
        ).sum(dim=1)

        window = torch.bmm(phi.unsqueeze(1), char_embeds).squeeze(1)
        return window, phi


class HandwritingGenerator(nn.Module):
    def __init__(
        self,
        char_vocab:  dict,
        M:           int = M_MDN,
        K:           int = K_ATTN,
        hidden_size: int = HIDDEN_SIZE,
        embed_dim:   int = EMBED_DIM,
    ):
        super().__init__()
        self.M           = M
        self.hidden_size = hidden_size
        self.embed_dim   = embed_dim
        self.char_vocab  = char_vocab
        pad_idx          = len(char_vocab)

        self.char_embed = nn.Embedding(len(char_vocab) + 1, embed_dim, padding_idx=pad_idx)
        self.attention  = SoftAttentionWindow(hidden_size, embed_dim, K)

        self.lstm1 = nn.LSTM(STROKE_DIM + embed_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(STROKE_DIM + embed_dim + hidden_size, hidden_size, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.mdn_head = nn.Linear(hidden_size * 2, M * 6 + 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mdn_head.weight)
        nn.init.zeros_(self.mdn_head.bias)
        with torch.no_grad():
            self.mdn_head.bias[-1].fill_(-3.0)

    def _zero_hidden(self, batch: int, device: torch.device):
        z = torch.zeros(1, batch, self.hidden_size, device=device)
        return (z, z.clone())

    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        pad_idx = len(self.char_vocab)
        max_len = max(len(t) for t in texts)
        idx     = torch.full((len(texts), max_len), pad_idx, dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            for j, ch in enumerate(text):
                idx[i, j] = self.char_vocab.get(ch, pad_idx)
        return idx

    def forward(self, strokes: torch.Tensor, texts: list):
        B, T, _ = strokes.shape
        device  = strokes.device

        char_idx    = self.encode_text(texts, device)
        char_embeds = self.char_embed(char_idx)

        self.attention.reset(B, device)
        h1     = self._zero_hidden(B, device)
        h2     = self._zero_hidden(B, device)
        window = torch.zeros(B, self.embed_dim, device=device)

        outs = []
        for t in range(T):
            x_t = strokes[:, t, :]

            inp1      = torch.cat([x_t, window], dim=1).unsqueeze(1)
            o1, h1    = self.lstm1(inp1, h1)
            o1        = self.norm1(o1.squeeze(1))
            window, _ = self.attention(o1, char_embeds)

            inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
            o2, h2 = self.lstm2(inp2, h2)
            o2     = self.norm2(o2.squeeze(1))

            outs.append(torch.cat([o1, o2], dim=1))

        output     = torch.stack(outs, dim=1)
        mdn_params = self.mdn_head(output)
        return mdn_params

    @torch.no_grad()
    def generate(
        self,
        text:       str,
        max_steps:  int   = 1000,
        bias:       float = 1.0,
        device:     torch.device | None = None,
    ) -> list[np.ndarray]:
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        char_embeds = self.char_embed(self.encode_text([text], device))
        U           = char_embeds.shape[1]

        self.attention.reset(1, device)
        h1     = self._zero_hidden(1, device)
        h2     = self._zero_hidden(1, device)
        window = torch.zeros(1, self.embed_dim, device=device)
        x_t    = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        strokes = [x_t.squeeze(0).cpu().numpy()]

        for step in range(max_steps):
            inp1        = torch.cat([x_t, window], dim=1).unsqueeze(1)
            o1, h1      = self.lstm1(inp1, h1)
            o1          = self.norm1(o1.squeeze(1))
            window, phi = self.attention(o1, char_embeds)

            # Condición agnóstica a la magnitud absoluta de phi (iter 1 master_train)
            phi_vals   = phi.squeeze(0)
            phi_norm   = phi_vals / (phi_vals.sum() + 1e-8)
            last_ratio = phi_norm[-1].item()
            cond_phi   = last_ratio > phi_norm[:-1].max().item() and last_ratio > 0.6
            cond_len   = step > U * 80
            if cond_phi or cond_len:
                break

            inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
            o2, h2 = self.lstm2(inp2, h2)
            o2     = self.norm2(o2.squeeze(1))

            params = self.mdn_head(torch.cat([o1, o2], dim=1)).squeeze(0)
            x_t    = sample_from_mdn(params, M=self.M, bias=bias).unsqueeze(0)
            strokes.append(x_t.squeeze(0).cpu().numpy())

        return strokes


def parse_mdn_params(params: torch.Tensor, M: int = M_MDN, bias: float = 0.0):
    pi_raw  = params[..., :M]
    mu_x    = params[..., M     : 2 * M]
    mu_y    = params[..., 2 * M : 3 * M]
    s_x_raw = params[..., 3 * M : 4 * M]
    s_y_raw = params[..., 4 * M : 5 * M]
    rho_raw = params[..., 5 * M : 6 * M]
    e_raw   = params[..., -1]

    pi      = F.softmax(pi_raw * (1.0 + bias), dim=-1)
    # softplus garantiza gradiente siempre activo — sin zona muerta como exp+clamp
    sigma_x = F.softplus(s_x_raw - bias) + 0.10
    sigma_y = F.softplus(s_y_raw - bias) + 0.10
    rho     = torch.tanh(rho_raw)
    e       = torch.sigmoid(e_raw)
    return pi, mu_x, mu_y, sigma_x, sigma_y, rho, e


def _bivariate_log_prob(
    dx:      torch.Tensor,
    dy:      torch.Tensor,
    mu_x:    torch.Tensor,
    mu_y:    torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    rho:     torch.Tensor,
) -> torch.Tensor:
    rho     = torch.clamp(rho, -0.99, 0.99)
    eps     = 1e-6
    norm_x  = (dx - mu_x) / (sigma_x + eps)
    norm_y  = (dy - mu_y) / (sigma_y + eps)
    rho2    = 1.0 - rho ** 2
    Z       = norm_x ** 2 + norm_y ** 2 - 2.0 * rho * norm_x * norm_y
    log_det = torch.log(2.0 * math.pi * sigma_x * sigma_y * rho2.sqrt() + eps)
    return -Z / (2.0 * rho2) - log_det


def mdn_loss(
    mdn_params:  torch.Tensor,
    targets:     torch.Tensor,
    mask:        torch.Tensor,
    M:           int   = M_MDN,
    sigma_reg:   float = SIGMA_REG_W,
    mu_weight:   float = 0.0,   # eliminado por defecto — amplificaba grad explosion
    pen_weight:  float = 8.0,   # corregido de 12 → sobrecorrección documentada en iter 2
) -> tuple[torch.Tensor, torch.Tensor]:
    pi, mu_x, mu_y, sigma_x, sigma_y, rho, _ = parse_mdn_params(mdn_params, M)

    dx  = targets[..., 0:1]
    dy  = targets[..., 1:2]
    pen = torch.clamp(targets[..., 2], 0.0, 1.0)

    log_p       = _bivariate_log_prob(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho)
    log_mixture = torch.logsumexp(torch.log(pi + 1e-8) + log_p, dim=-1)

    e_raw      = mdn_params[..., -1]
    pen_w      = torch.tensor(pen_weight, device=e_raw.device)
    pen_loss   = F.binary_cross_entropy_with_logits(
        e_raw, pen, pos_weight=pen_w, reduction='none'
    )

    per_step      = -log_mixture + pen_loss
    safe_per_step = torch.where(mask > 0.5, per_step, torch.zeros_like(per_step))
    nll_loss      = safe_per_step.sum() / (mask.sum() + 1e-8)

    wmu_x    = (pi * mu_x).sum(dim=-1)
    wmu_y    = (pi * mu_y).sum(dim=-1)
    mse_mask = mask > 0.5
    mse_loss = (
        (wmu_x - targets[..., 0]).pow(2) +
        (wmu_y - targets[..., 1]).pow(2)
    )
    mse_loss = mse_loss[mse_mask].mean() * mu_weight

    s_x_raw          = mdn_params[..., 3 * M : 4 * M]
    s_y_raw          = mdn_params[..., 4 * M : 5 * M]
    sigma_target_log = -0.5   # exp(-0.5)≈0.61 — alejado del floor 0.10 para gradiente activo
    reg = sigma_reg * (
        (s_x_raw - sigma_target_log).pow(2) +
        (s_y_raw - sigma_target_log).pow(2)
    ).mean()

    entropy_reg = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()

    total = nll_loss + reg + mse_loss + (-0.30 * entropy_reg)  # 0.05→0.30: check 2 confirma pi collapse
    return total, nll_loss


@torch.no_grad()
def sample_from_mdn(
    params: torch.Tensor,
    M:      int   = M_MDN,
    bias:   float = 0.0,
) -> torch.Tensor:
    if params.dim() == 1:
        params = params.unsqueeze(0)
    return sample_from_mdn_batch(params, M, bias).squeeze(0)


@torch.no_grad()
def sample_from_mdn_batch(
    params: torch.Tensor,
    M:      int   = M_MDN,
    bias:   float = 0.0,
) -> torch.Tensor:
    pi, mu_x, mu_y, sigma_x, sigma_y, rho, e = parse_mdn_params(params, M, bias)
    B = params.shape[0]

    k   = torch.multinomial(pi, num_samples=1).squeeze(1)
    idx = k.unsqueeze(1)

    mx  = mu_x.gather(1, idx).squeeze(1)
    my  = mu_y.gather(1, idx).squeeze(1)
    sx  = sigma_x.gather(1, idx).squeeze(1)
    sy  = sigma_y.gather(1, idx).squeeze(1)
    r   = rho.gather(1, idx).squeeze(1).clamp(-0.99, 0.99)
    e_k = e

    z1  = torch.randn(B, device=params.device)
    z2  = torch.randn(B, device=params.device)

    dx  = mx + sx * z1
    dy  = my + sy * (r * z1 + torch.sqrt(1.0 - r ** 2) * z2)
    pen = torch.bernoulli(e_k)

    return torch.stack([dx, dy, pen], dim=1)