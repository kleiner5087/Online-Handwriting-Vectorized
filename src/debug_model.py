import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from src.model import HandwritingGenerator, mdn_loss, parse_mdn_params, sample_from_mdn_batch

# ─── Config ───────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 600
LR           = 1e-4
EPOCH_SIZE   = 1000
SAVE_PATH    = './modelos/model10.pt'   # checkpoint de origen (Iter 8, ep277)
RESUME       = True

# SS parcial sobre XY — pen siempre ground truth para no contaminar pen_head.
# SS_WARMUP bajo (75) porque el covariate shift se instala desde las primeras épocas.
# Con model3 el gap train→inference fue de 4.6 unidades logit; SS desde temprano
# es la única intervención que lo reduce estructuralmente.
SS_WARMUP    = 75
SS_MIN       = 0.80

MU_WEIGHT    = 0.0
SIGMA_REG    = 0.08
PEN_WEIGHT   = 5.0
CLIP         = 5.0
TBPTT_K      = 15
T_MAX_TRAIN  = 300

# anchor_loss: hinge unilateral que ancla e_raw_norm >= ANCHOR_TARGET.
# Impide que sep creciente colapse e_raw_norm → -inf sin frenar sep.
# ANCHOR_TARGET=-3.5 derivado del equilibrio BCE con gap inference~1.5u:
#   σ(-3.5+1.5)=σ(-2.0)≈0.12 → pen_rate×~1.0 a bias=3.5 (observado en model6 ep277).
# Arrancar en 0.15: efectivo desde ep~50, conservador para no dominar gradiente MDN.
# Si sep se estanca también → anchor_weight demasiado alto → bajar a 0.08.
ANCHOR_WEIGHT  = 0.15
ANCHOR_TARGET  = -3.5

# freq_floor: activa cuando nll_s < FREQ_ACTIVATE_NLL (modelo estabilizado).
# Monitorear que sep no caiga de 2.0 al activarlo.
FREQ_WEIGHT        = 0.0
# Activar freq_floor solo cuando el modelo ya tiene morfología estable.
# nll_s < 3.0 se alcanza en ep=6 con LR=1e-4 — demasiado temprano.
# nll_s < 0.5 corresponde a un modelo que ya convergió en la parte gaussiana.
FREQ_ACTIVATE_NLL  = 0.5
TRUE_PEN_RATE      = 0.0296

LOG_EVERY    = 10
USE_AMP      = False
DEBUG_DIR    = './debug_logs'
DEBUG_BATCHES = 5


# ─── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_vocab(dataset):
    chars = sorted(dataset.data_by_char.keys())
    return {ch: i for i, ch in enumerate(chars)}


def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs    = [s[:T_MAX_TRAIN] if len(s) > T_MAX_TRAIN else s for s in seqs]
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    B       = len(seqs)
    padded  = np.zeros((B, max_len, 3), dtype=np.float32)
    mask    = np.zeros((B, max_len),    dtype=np.float32)
    for i, (s, l) in enumerate(zip(seqs, lengths)):
        padded[i, :l] = s
        mask[i, :l]   = 1.0
    return torch.from_numpy(padded), torch.from_numpy(mask), list(labels)


def get_teacher_ratio(actual_epoch):
    if actual_epoch <= SS_WARMUP:
        return 1.0
    progress = (actual_epoch - SS_WARMUP) / max(EPOCHS - SS_WARMUP, 1)
    return max(SS_MIN, 1.0 - progress * (1.0 - SS_MIN))


# ─── Forward TBPTT ────────────────────────────────────────────────────────────
def forward_tbptt(model, strokes, texts, device, teacher_ratio):
    B, T, _ = strokes.shape

    char_idx    = model.encode_text(texts, device)
    char_embeds = model.char_embed(char_idx)

    model.attention.reset(B, device)
    h1     = model._zero_hidden(B, device)
    h2     = model._zero_hidden(B, device)
    window = torch.zeros(B, model.embed_dim, device=device)
    x_t    = strokes[:, 0, :]
    all_params = []

    for t in range(T - 1):
        if t > 0 and t % TBPTT_K == 0:
            h1     = (h1[0].detach(), h1[1].detach())
            h2     = (h2[0].detach(), h2[1].detach())
            window = window.detach()
            model.attention.kappa = model.attention.kappa.detach()

        inp1      = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1    = model.lstm1(inp1, h1)
        o1        = model.norm1(o1.squeeze(1))
        window, _ = model.attention(o1, char_embeds)
        window_n = F.layer_norm(window, [window.shape[-1]])

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        features = torch.cat([o1, o2, window_n], dim=1)
        params_t = torch.cat([model.mdn_head(features), model.pen_head(features)], dim=-1)
        all_params.append(params_t)

        if teacher_ratio >= 1.0:
            x_t = strokes[:, t + 1, :]
        else:
            # SS solo en XY: expone al LSTM a trayectorias propias para cerrar el
            # gap train→inference, sin tocar pen que va siempre como ground truth.
            # Esto evita que pen_head reciba estados ocultos contaminados por su
            # propia señal, que fue la causa del círculo vicioso en model2/model3.
            use_teacher = torch.rand(B, device=device) < teacher_ratio
            sampled     = sample_from_mdn_batch(params_t.detach(), M=model.M, bias=0.5)
            x_t_xy      = torch.where(
                use_teacher.unsqueeze(1),
                strokes[:, t + 1, :2],
                sampled[:, :2],
            )
            x_t = torch.cat([x_t_xy, strokes[:, t + 1, 2:3]], dim=1)

    return torch.stack(all_params, dim=1)


# ─── Métricas de debug (vectorizadas post-forward) ────────────────────────────
@torch.no_grad()
def collect_debug(params, target, mask, model, texts):
    B  = mask.shape[0]
    M  = model.M

    e_raw    = params[..., -1]
    pen_tgt  = target[..., 2] > 0.5
    valid    = mask > 0.5
    pen_m    = valid & pen_tgt
    norm_m   = valid & ~pen_tgt

    e_pen_t  = e_raw[pen_m]
    e_norm_t = e_raw[norm_m]

    sep  = (e_pen_t.mean() - e_norm_t.mean()).item() if (pen_m.any() and norm_m.any()) else float('nan')

    pred = (e_raw > 0.0) & valid
    tp   = int((pred &  pen_m).sum())
    fp   = int((pred & norm_m).sum())
    fn   = int((~pred & pen_m).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    pred_x = (tp + fp) / max(valid.sum().item() * TRUE_PEN_RATE, 1)

    e_norm_np = e_norm_t.cpu().float().numpy()
    naz       = float(np.mean(e_norm_np > 0.0)) if len(e_norm_np) > 0 else float('nan')
    p90n      = float(np.percentile(e_norm_np, 90)) if len(e_norm_np) > 0 else float('nan')

    pred_np  = pred.cpu().numpy()
    pen_np   = pen_tgt.cpu().numpy()
    valid_np = valid.cpu().numpy()
    isolated_flags = []
    for b in range(B):
        pen_pos = np.where(pen_np[b] & valid_np[b])[0]
        fp_pos  = np.where(pred_np[b] & ~pen_np[b] & valid_np[b])[0]
        if len(fp_pos) == 0 or len(pen_pos) == 0:
            continue
        min_dists = np.abs(fp_pos[:, None] - pen_pos[None, :]).min(axis=1)
        isolated_flags.extend((min_dists > 5).tolist())
    fp_iso = float(np.mean(isolated_flags)) if isolated_flags else float('nan')

    mean_U  = float(np.mean([len(t) for t in texts]))
    kap_end = model.attention.kappa.mean().item()
    kap_ov  = kap_end / max(mean_U, 1.0)

    bce_raw = F.binary_cross_entropy_with_logits(
        e_raw[pen_m], torch.ones_like(e_raw[pen_m]), reduction='mean'
    ).item() if pen_m.any() else float('nan')

    # enorm_mean: media de e_raw en pasos normales (pen_hard=0).
    # Indicador directo del estado del equilibrio BCE en inferencia.
    # Objetivo operativo: mantenerse en [-4.0, -3.0].
    # Si cae bajo -4.0 → anchor_loss activo y peleando contra el BCE.
    # Si sube sobre -2.5 → anchor demasiado agresivo, posible interferencia con MDN.
    enorm_mean = e_norm_t.mean().item() if norm_m.any() else float('nan')

    return dict(sep=sep, prec=prec, rec=rec, pred_x=pred_x,
                naz=naz, p90n=p90n, fp_iso=fp_iso, kap_ov=kap_ov,
                bce_raw=bce_raw, enorm_mean=enorm_mean)


# ─── Epoch de entrenamiento ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device, teacher_ratio, collect, use_freq_floor):
    model.train()

    tot_loss = 0.0
    m_smin, m_grad, m_Hpi, m_nll_s, m_nll_g, m_anchor = [], [], [], [], [], []
    debug_acc = []

    freq_w = FREQ_WEIGHT if use_freq_floor else 0.0

    for batch_idx, (strokes, mask, texts) in enumerate(loader):
        strokes = strokes.to(device, non_blocking=True)
        mask    = mask.to(device,    non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            params = forward_tbptt(model, strokes, texts, device, teacher_ratio)
            target = strokes[:, 1:, :]
            t_mask = mask[:, 1:]

            loss, nll, nll_stroke, anchor_l = mdn_loss(
                params, target, t_mask,
                mu_weight=MU_WEIGHT,
                sigma_reg=SIGMA_REG,
                pen_weight=PEN_WEIGHT,
                freq_weight=freq_w,
                true_pen_rate=TRUE_PEN_RATE,
                anchor_weight=ANCHOR_WEIGHT,
                anchor_target=ANCHOR_TARGET,
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.mdn_head.parameters(), 1.5)
        nn.utils.clip_grad_norm_(model.pen_head.parameters(), 1.5)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item()
        m_nll_s.append(nll_stroke.item())
        m_nll_g.append(nll.item())
        m_grad.append(grad_norm.item())
        m_anchor.append(anchor_l.item())

        with torch.no_grad():
            pi, _, _, sx, _, _, _ = parse_mdn_params(params.detach())
            m_smin.append(sx.min().item())
            m_Hpi.append(-(pi * torch.log(pi + 1e-8)).sum(-1).mean().item())

        if collect and batch_idx < DEBUG_BATCHES:
            d = collect_debug(params.detach(), target, t_mask, model, texts)
            debug_acc.append(d)

    agg_debug = None
    if debug_acc:
        keys      = debug_acc[0].keys()
        agg_debug = {k: float(np.nanmean([d[k] for d in debug_acc])) for k in keys}

    metrics = dict(
        nll_s  = float(np.mean(m_nll_s)),
        nll_g  = float(np.mean(m_nll_g)),
        smin   = float(np.mean(m_smin)),
        grad   = float(np.mean(m_grad)),
        Hpi    = float(np.mean(m_Hpi)),
        anchor = float(np.mean(m_anchor)),
        loss   = tot_loss / max(len(loader), 1),
        debug  = agg_debug,
    )
    return metrics


# ─── CSV logger ───────────────────────────────────────────────────────────────
CSV_HEADER = 'ep,tf,ff,nll_s,nll_g,loss,smin,grad,Hpi,anchor,bce,sep,prec,rec,pred_x,naz,p90n,fp_iso,kap_ov,enorm_mean'

def _f(v, d=4):
    return f'{v:.{d}f}' if (isinstance(v, float) and not np.isnan(v)) else 'nan'

def log_csv(epoch, tf, use_freq_floor, metrics, log_file):
    d = metrics.get('debug') or {}
    row = ','.join([
        str(epoch),
        _f(tf, 2),
        '1' if use_freq_floor else '0',
        _f(metrics['nll_s']),
        _f(metrics['nll_g']),
        _f(metrics['loss']),
        _f(metrics['smin']),
        _f(metrics['grad'], 2),
        _f(metrics['Hpi'], 2),
        _f(metrics.get('anchor', float('nan'))),
        _f(d.get('bce_raw',    float('nan'))),
        _f(d.get('sep',        float('nan'))),
        _f(d.get('prec',       float('nan'))),
        _f(d.get('rec',        float('nan'))),
        _f(d.get('pred_x',     float('nan')), 1),
        _f(d.get('naz',        float('nan'))),
        _f(d.get('p90n',       float('nan'))),
        _f(d.get('fp_iso',     float('nan'))),
        _f(d.get('kap_ov',     float('nan'))),
        _f(d.get('enorm_mean', float('nan'))),
    ])
    with open(log_file, 'a') as f:
        f.write(row + '\n')


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    from src.UJIPen import UJIDataset

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f'Device: {DEVICE}')

    dataset = UJIDataset('./data/ujipenchars2.txt', epoch_size=EPOCH_SIZE)
    dataset.load_dictionary_from_txt('./data/words.txt')

    char_vocab = build_vocab(dataset)
    model      = HandwritingGenerator(char_vocab).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - SS_WARMUP, 1), eta_min=5e-6
    )
    scaler      = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    start_epoch = 1
    best_nll           = float('inf')
    last_sep           = 0.0
    last_prec          = 0.0
    last_enorm_mean    = float('nan')   # media e_raw en pasos normales; objetivo: [-4.0, -3.0]
    quality_gate_armed = False           # se arma cuando sep supera 1.0 por primera vez

    if RESUME and os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing:
            print(f'  Pesos no cargados (nuevos): {missing}')
        start_epoch = ckpt['epoch'] + 1
        best_nll    = ckpt.get('best_nll', float('inf'))

        ans = input('¿Cargar optimizer? [y/n]: ').strip().lower()
        if ans == 'y' and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f'  Optimizer cargado. Reanudando ep {start_epoch}')
        else:
            print(f'  Optimizer fresco. Reanudando ep {start_epoch}')
            with torch.no_grad():
                K = model.attention.K
                model.attention.proj.bias[2 * K:].fill_(-3.82)

        for _ in range(max(0, start_epoch - 1 - SS_WARMUP)):
            scheduler.step()
    else:
        print('  Sin checkpoint. Iniciando desde cero.')

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'), persistent_workers=False,
    )

    log_file = os.path.join(DEBUG_DIR, f'run_ep{start_epoch}.csv')
    with open(log_file, 'w') as f:
        f.write(CSV_HEADER + '\n')
    print(f'Log: {log_file}')
    print(CSV_HEADER)

    for local_ep in range(1, EPOCHS + 1):
        actual_ep      = start_epoch + local_ep - 1
        teacher_ratio  = get_teacher_ratio(actual_ep)
        use_freq_floor = (best_nll < FREQ_ACTIVATE_NLL)
        do_collect     = (local_ep % LOG_EVERY == 0 or local_ep == 1)

        metrics = train_epoch(
            model, loader, optimizer, scaler, DEVICE,
            teacher_ratio, do_collect, use_freq_floor,
        )

        if actual_ep > SS_WARMUP:
            scheduler.step()

        d = metrics.get('debug') or {}
        if d:
            last_sep        = d.get('sep',        last_sep)
            last_prec       = d.get('prec',       last_prec)
            last_enorm_mean = d.get('enorm_mean', last_enorm_mean)
            if last_sep > 2.0:
                quality_gate_armed = True
            global FREQ_WEIGHT
            if d.get('naz', 0.0) > 0.05 and FREQ_WEIGHT > 0.0:
                FREQ_WEIGHT = max(FREQ_WEIGHT * 0.5, 0.05)
                print(f'  [NAZ ALARM] naz={d["naz"]:.3f} > 0.05 → FREQ_WEIGHT reducido a {FREQ_WEIGHT:.3f}')

        if do_collect:
            log_csv(actual_ep, teacher_ratio, use_freq_floor, metrics, log_file)
            print(','.join([
                str(actual_ep),
                f'{teacher_ratio:.2f}',
                '1' if use_freq_floor else '0',
                f'{metrics["nll_s"]:.4f}',
                f'{metrics["nll_g"]:.4f}',
                f'{metrics["loss"]:.4f}',
                f'{metrics["smin"]:.3f}',
                f'{metrics["grad"]:.1f}',
                f'{metrics["Hpi"]:.2f}',
                f'{metrics.get("anchor", float("nan")):.4f}',
                f'{d.get("bce_raw",    float("nan")):.4f}',
                f'{d.get("sep",        float("nan")):.3f}',
                f'{d.get("prec",       float("nan")):.3f}',
                f'{d.get("rec",        float("nan")):.3f}',
                f'{d.get("pred_x",     float("nan")):.1f}',
                f'{d.get("naz",        float("nan")):.3f}',
                f'{d.get("p90n",       float("nan")):.3f}',
                f'{d.get("fp_iso",     float("nan")):.3f}',
                f'{d.get("kap_ov",     float("nan")):.2f}',
                f'{d.get("enorm_mean", float("nan")):.3f}',
            ]))

        # Criterio de checkpoint compuesto:
        # — Gate desarmado (sep nunca superó 1.0): guardar por nll_s puro.
        #   Cubre todo el arranque hasta que pen_head desarrolle separación real.
        # — Gate armado (sep cruzó 1.0 al menos una vez): exigir:
        #     · sep > 2.0 y prec > 0.22  → pen_head con separación real.
        #     · enorm_mean > -4.0        → anchor efectivo, equilibrio BCE saludable.
        #       (si nan = debug no corrió esta época → no bloquear checkpoint).
        #     · tolerancia 2% en nll_s   → evita guardar regresiones morfológicas.
        enorm_ok = np.isnan(last_enorm_mean) or (last_enorm_mean > -4.0)

        if not quality_gate_armed:
            quality_ok = True
            nll_ok     = metrics['nll_s'] < best_nll
        else:
            quality_ok = (last_sep > 2.0) and (last_prec > 0.22) and enorm_ok
            nll_ok = metrics['nll_s'] < best_nll + abs(best_nll) * 0.02

        if quality_ok and nll_ok:
            best_nll = metrics['nll_s']
            torch.save({
                'epoch':      actual_ep,
                'loss':       metrics['loss'],
                'best_nll':   best_nll,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'char_vocab': char_vocab,
                'std_dx':     dataset.std_dx,
                'std_dy':     dataset.std_dy,
                'mean_dx':    dataset.mean_dx,
                'mean_dy':    dataset.mean_dy,
            }, SAVE_PATH)
            enorm_str = f'{last_enorm_mean:.3f}' if not np.isnan(last_enorm_mean) else 'nan'
            print(f'  ✓ ep={actual_ep}  nll_s={best_nll:.4f}  sep={last_sep:.3f}  prec={last_prec:.3f}  enorm={enorm_str}')

    print(f'\nFin. Log: {log_file}')


if __name__ == '__main__':
    main()