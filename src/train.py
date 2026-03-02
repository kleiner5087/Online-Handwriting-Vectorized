import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import HandwritingGenerator, mdn_loss, parse_mdn_params, sample_from_mdn_batch

# ─── Semilla ──────────────────────────────────────────────────────────────────
SEED = 42

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 500
LR          = 1e-4
EPOCH_SIZE  = 3000
SAVE_PATH   = './modelos/handwriting_model.pt'
RESUME      = True

# Scheduled Sampling
SS_WARMUP   = 200    # épocas con teacher forcing puro
SS_MIN      = 0.20   # ratio mínimo de TF al final del decay

# Loss
MU_WEIGHT   = 0.0    # eliminado — debug check 4 confirma que amplifica grad explosion
SIGMA_REG   = 0.30
PEN_WEIGHT  = 8.0    # 12.0 producía sobrecorrección (pen→0.277 documentado)
CLIP        = 1.5    # check 3: norma media ~80-120, clip=5 era inefectivo

# Truncated BPTT — limita la acumulación de gradiente a K pasos hacia atrás
# Los estados ocultos fluyen forward sin restricción; solo el gradiente se trunca
TBPTT_K     = 75     # ~2 caracteres por ventana a 43 pasos/char

# Log
LOG_EVERY   = 10

# Mixed Precision (AMP)
USE_AMP     = False   # desactivado: fp16 tiene precisión insuficiente para log/exp de la loss gaussiana


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Entrenamiento HandwritingGenerator')
    p.add_argument(
        '--optimizer', choices=['y', 'n'], default=None,
        help=(
            'Cargar estado del optimizer desde checkpoint. '
            '"y" = cargar (continuación directa). '
            '"n" = optimizer fresco (útil si cambiaste hiperparámetros de loss). '
            'Si se omite, el script pregunta interactivamente.'
        )
    )
    return p.parse_args()


def build_vocab(dataset) -> dict:
    chars = sorted(dataset.data_by_char.keys())
    return {ch: i for i, ch in enumerate(chars)}


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths      = [len(s) for s in seqs]
    max_len      = max(lengths)
    B            = len(seqs)

    padded = np.zeros((B, max_len, 3), dtype=np.float32)
    mask   = np.zeros((B, max_len),    dtype=np.float32)

    for i, (s, l) in enumerate(zip(seqs, lengths)):
        padded[i, :l] = s
        mask[i, :l]   = 1.0

    return torch.from_numpy(padded), torch.from_numpy(mask), list(labels)


def get_teacher_ratio(epoch: int) -> float:
    if epoch <= SS_WARMUP:
        return 1.0
    progress = (epoch - SS_WARMUP) / max(EPOCHS - SS_WARMUP, 1)
    return max(SS_MIN, 1.0 - progress * (1.0 - SS_MIN))


def forward_tbptt(
    model:         HandwritingGenerator,
    strokes:       torch.Tensor,
    texts:         list,
    device:        torch.device,
    teacher_ratio: float,
) -> torch.Tensor:
    """
    Forward pass con Truncated BPTT.
    El estado oculto fluye a través de toda la secuencia (memoria larga preservada).
    El gradiente se trunca cada TBPTT_K pasos para evitar la explosión documentada
    en Check 3: lstm1 norma ~80-120 con T≈430 pasos completos.
    """
    B, T, _ = strokes.shape

    char_idx    = model.encode_text(texts, device)
    char_embeds = model.char_embed(char_idx)

    model.attention.reset(B, device)
    h1     = model._zero_hidden(B, device)
    h2     = model._zero_hidden(B, device)
    window = torch.zeros(B, model.embed_dim, device=device)

    x_t        = strokes[:, 0, :]
    all_params = []

    for t in range(T - 1):
        # Truncar el grafo de cómputo cada TBPTT_K pasos
        # Los valores de h1, h2, window se preservan para el forward
        # pero el backward no puede atravesar este punto
        if t > 0 and t % TBPTT_K == 0:
            h1     = (h1[0].detach(), h1[1].detach())
            h2     = (h2[0].detach(), h2[1].detach())
            window = window.detach()
            model.attention.kappa = model.attention.kappa.detach()

        inp1      = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1    = model.lstm1(inp1, h1)
        o1        = model.norm1(o1.squeeze(1))
        window, _ = model.attention(o1, char_embeds)

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        params_t = model.mdn_head(torch.cat([o1, o2], dim=1))
        all_params.append(params_t)

        # Siguiente input: teacher forcing o muestra propia
        if teacher_ratio >= 1.0:
            x_t = strokes[:, t + 1, :]
        else:
            use_teacher = torch.rand(B, device=device) < teacher_ratio
            teacher_x   = strokes[:, t + 1, :]
            sampled     = sample_from_mdn_batch(params_t.detach(), M=model.M, bias=0.0)
            x_t         = torch.where(use_teacher.unsqueeze(1), teacher_x, sampled)

    return torch.stack(all_params, dim=1)


def train_epoch(
    model:         HandwritingGenerator,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    scaler:        torch.cuda.amp.GradScaler,
    device:        torch.device,
    teacher_ratio: float,
) -> tuple[float, dict]:
    model.train()
    total            = 0.0
    metric_sigma_x   = []
    metric_sigma_y   = []
    metric_sigma_min = []
    metric_wmu_x     = []
    metric_wmu_y     = []
    metric_pen       = []
    metric_grad      = []
    metric_nll       = []
    metric_entropy   = []

    for strokes, mask, texts in loader:
        strokes = strokes.to(device, non_blocking=True)
        mask    = mask.to(device,    non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            params = forward_tbptt(model, strokes, texts, device, teacher_ratio)

            target = strokes[:, 1:, :]
            t_mask = mask[:, 1:]

            loss, nll = mdn_loss(
                params, target, t_mask,
                mu_weight=MU_WEIGHT,
                sigma_reg=SIGMA_REG,
                pen_weight=PEN_WEIGHT,
            )

        if not torch.isfinite(loss):
            print('  [WARN] Loss no finita, saltando batch.')
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.mdn_head.parameters(), 0.5)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        metric_grad.append(grad_norm.item())

        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        metric_nll.append(nll.item())

        with torch.no_grad():
            pi, mu_x, mu_y, sx, sy, _, e = parse_mdn_params(params.detach())
            metric_sigma_x.append(sx.mean().item())
            metric_sigma_y.append(sy.mean().item())
            metric_sigma_min.append(sx.min().item())
            metric_wmu_x.append((pi * mu_x).sum(-1).mean().item())
            metric_wmu_y.append((pi * mu_y).sum(-1).mean().item())
            metric_pen.append(e.mean().item())
            entropy = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
            metric_entropy.append(entropy.item())

    metrics = {
        'sigma_x':   np.mean(metric_sigma_x),
        'sigma_y':   np.mean(metric_sigma_y),
        'sigma_min': np.mean(metric_sigma_min),
        'wmu_x':     np.mean(metric_wmu_x),
        'wmu_y':     np.mean(metric_wmu_y),
        'pen':       np.mean(metric_pen),
        'grad':      np.mean(metric_grad),
        'nll':       np.mean(metric_nll),
        'entropy':   np.mean(metric_entropy),
    }
    return total / max(len(loader), 1), metrics


def main():
    args = parse_args()
    set_seed(SEED)
    sys.path.insert(0, '.')
    from src.UJIPen import UJIDataset

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {DEVICE}  |  Semilla: {SEED}  |  AMP: {USE_AMP}')

    dataset = UJIDataset('./data/ujipenchars2.txt', epoch_size=EPOCH_SIZE)
    dataset.load_dictionary_from_txt('./data/words.txt')

    char_vocab = build_vocab(dataset)
    model      = HandwritingGenerator(char_vocab).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parámetros entrenables: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # CosineAnnealingLR sobre la fase SS — immune al ruido de SS
    # Solo se llama scheduler.step() a partir de la época SS_WARMUP+1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(EPOCHS - SS_WARMUP, 1),
        eta_min = 5e-6,
    )

    scaler      = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    start_epoch = 1
    best_loss   = float('inf')

    if RESUME and os.path.exists(SAVE_PATH):
        print(f"\nCheckpoint encontrado en '{SAVE_PATH}'.")

        # Resolver la decisión del optimizer
        load_opt = args.optimizer
        if load_opt is None:
            ans = input('¿Cargar estado del optimizer? [y/n]: ').strip().lower()
            load_opt = ans if ans in ('y', 'n') else 'y'

        checkpoint  = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss   = checkpoint['loss']

        if load_opt == 'y':
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f'  Optimizer cargado. Reanudando desde época {start_epoch}.')
            else:
                print('  Checkpoint sin estado de optimizer — arrancando optimizer fresco.')
        else:
            print(f'  Optimizer fresco. Reanudando desde época {start_epoch}.')

        # Avanzar el scheduler hasta la época actual sin modificar LR del warmup
        epochs_past_warmup = max(0, start_epoch - 1 - SS_WARMUP)
        for _ in range(epochs_past_warmup):
            scheduler.step()

        print(f'  best_loss={best_loss:.4f}  |  '
              f'LR actual: {optimizer.param_groups[0]["lr"]:.2e}\n')
    else:
        if RESUME:
            print(f"  No se encontró checkpoint en '{SAVE_PATH}'. Iniciando desde cero.\n")

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,           # 0 es necesario en WSL con CUDA
        pin_memory  = (DEVICE.type == 'cuda'),
        persistent_workers = False,
    )

    print('─' * 90)
    print(f'{"Época":>6}  {"loss":>8}  {"nll":>8}  {"lr":>9}  {"tf":>4}  │  '
          f'{"σx":>5}  {"σy":>5}  {"σmin":>5}  │  '
          f'{"wμx":>6}  {"wμy":>6}  {"pen":>5}  │  '
          f'{"grad":>7}  {"H(π)":>5}')
    print('─' * 90)

    for epoch in range(start_epoch, EPOCHS + 1):
        teacher_ratio = get_teacher_ratio(epoch)
        loss, metrics = train_epoch(model, loader, optimizer, scaler, DEVICE, teacher_ratio)

        # Scheduler solo activo en la fase SS para no interpretar como plateau el warmup
        if epoch > SS_WARMUP:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == start_epoch:
            print(
                f'{epoch:6d}  '
                f'{loss:8.4f}  {metrics["nll"]:8.4f}  '
                f'{lr_now:9.2e}  {teacher_ratio:4.2f}  │  '
                f'{metrics["sigma_x"]:5.3f}  {metrics["sigma_y"]:5.3f}  '
                f'{metrics["sigma_min"]:5.3f}  │  '
                f'{metrics["wmu_x"]:6.3f}  {metrics["wmu_y"]:6.3f}  '
                f'{metrics["pen"]:5.3f}  │  '
                f'{metrics["grad"]:7.2f}  {metrics["entropy"]:5.2f}'
            )

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    'epoch':      epoch,
                    'loss':       best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'char_vocab': char_vocab,
                    'std_dx':     dataset.std_dx,
                    'std_dy':     dataset.std_dy,
                    'mean_dy':    dataset.mean_dy,
                },
                SAVE_PATH,
            )
            print(f'  ✓ Checkpoint guardado  (época={epoch}  loss={best_loss:.4f})')

    print('\nEntrenamiento finalizado.')

# Ejecucion
if __name__ == '__main__':
    main()