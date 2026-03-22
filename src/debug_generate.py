import argparse
import csv
import os
import random
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, '.')
from src.model import HandwritingGenerator, sample_from_mdn

TRUE_PEN_RATE = 0.0296


def parse_range(raw: str) -> list[float]:
    try:
        parts = [float(x) for x in raw.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Formato requerido: start,stop,step  —  recibido: '{raw}'")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Formato requerido: start,stop,step  —  recibido: '{raw}'")
    start, stop, step = parts
    vals, v = [], start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HandwritingGenerator(ckpt['char_vocab']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    stats = {
        'char_vocab': ckpt['char_vocab'],
        'std_dx':     float(ckpt['std_dx']),
        'std_dy':     float(ckpt['std_dy']),
        'mean_dx':    float(ckpt.get('mean_dx', 0.0)),
        'mean_dy':    float(ckpt.get('mean_dy', 0.0)),
        'epoch':      int(ckpt.get('epoch', 0)),
        'best_nll':   float(ckpt.get('best_nll', float('nan'))),
    }
    return model, stats


@torch.no_grad()
def generate_one(
    model: HandwritingGenerator,
    texto: str,
    device: torch.device,
    bias: float,
    pen_bias: float,
    max_steps: int,
    steps_per_char: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    char_embeds = model.char_embed(model.encode_text([texto], device))
    U = char_embeds.shape[1]
    expected = len(texto.replace(' ', '')) * steps_per_char

    model.attention.reset(1, device)
    h1     = model._zero_hidden(1, device)
    h2     = model._zero_hidden(1, device)
    window = torch.zeros(1, model.embed_dim, device=device)
    x_t    = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    strokes_out    = [np.array([0., 0., 1.], dtype=np.float32)]
    attentions_out = []
    params_out     = []
    stop_reason    = 'max_steps'

    for step in range(max_steps):
        inp1        = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1      = model.lstm1(inp1, h1)
        o1          = model.norm1(o1.squeeze(1))
        window, phi = model.attention(o1, char_embeds)

        phi_n      = phi.squeeze(0) / (phi.squeeze(0).sum() + 1e-8)
        last_ratio = phi_n[-1].item()
        if len(phi_n) > 1 and last_ratio > phi_n[:-1].max().item() and last_ratio > 0.6:
            stop_reason = 'phi'
            break
        if step > expected:
            stop_reason = 'length'
            break

        attentions_out.append(phi.squeeze(0).cpu().numpy())

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        features   = torch.cat([o1, o2], dim=1)
        raw_params = torch.cat([model.mdn_head(features), model.pen_head(features)], dim=-1).squeeze(0)
        params_out.append(raw_params.cpu().numpy())

        sample_params        = raw_params.clone()
        sample_params[-1]   -= pen_bias
        x_t = sample_from_mdn(sample_params, M=model.M, bias=bias).unsqueeze(0)

        if x_t[0, 2].item() > 0.5:
            x_t = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        strokes_out.append(x_t.squeeze(0).cpu().numpy())

    strokes    = np.array(strokes_out,    dtype=np.float32)
    attentions = np.array(attentions_out, dtype=np.float32) if attentions_out else np.zeros((0, U), dtype=np.float32)
    params_arr = np.array(params_out,     dtype=np.float32) if params_out     else np.zeros((0, 1), dtype=np.float32)
    return strokes, attentions, params_arr, stop_reason


def compute_sample_metrics(
    strokes:    np.ndarray,
    attentions: np.ndarray,
    params_arr: np.ndarray,
    std_dx:     float,
    std_dy:     float,
    stop_reason: str,
    M:          int,
) -> dict:
    T   = len(strokes)
    dx  = strokes[1:, 0]
    dy  = strokes[1:, 1]
    pen = strokes[1:, 2]

    abs_x = np.cumsum(dx * std_dx)
    abs_y = np.cumsum(dy * std_dy)

    pen_idx  = np.where(pen > 0.5)[0]
    norm_idx = np.where(pen <= 0.5)[0]
    n_pen    = len(pen_idx)

    pen_gap_median = float('nan')
    micro_pct      = float('nan')
    if n_pen > 1:
        gaps           = np.diff(pen_idx)
        pen_gap_median = float(np.median(gaps))
        micro_pct      = float(np.mean(gaps < 5))

    attn_cobertura   = float('nan')
    attn_monotonia   = float('nan')
    attn_llego_final = 0
    kap_ov           = float('nan')
    if attentions.shape[0] > 0:
        phi_peaks        = np.argmax(attentions, axis=1)
        n_chars          = attentions.shape[1]
        attn_cobertura   = float(len(set(phi_peaks)) / max(n_chars, 1))
        attn_monotonia   = float(np.mean(np.diff(phi_peaks) >= 0)) if len(phi_peaks) > 1 else float('nan')
        attn_llego_final = int(phi_peaks[-1] == n_chars - 1)
        kap_ov           = float(attentions[-1].sum()) / max(n_chars, 1)

    eraw_mean = eraw_pct_pos = eraw_p10 = eraw_p90 = float('nan')
    sep = naz = p90n = Hpi = smin = float('nan')

    if params_arr.shape[0] > 0 and params_arr.shape[1] > 1:
        e_raw = params_arr[:, -1]

        e_at_pen  = e_raw[pen_idx]  if len(pen_idx)  > 0 else np.array([])
        e_at_norm = e_raw[norm_idx] if len(norm_idx) > 0 else np.array([])

        eraw_mean    = float(np.mean(e_raw))
        eraw_pct_pos = float(np.mean(e_raw > 0))
        eraw_p10     = float(np.percentile(e_raw, 10))
        eraw_p90     = float(np.percentile(e_raw, 90))

        sep  = float(e_at_pen.mean()  - e_at_norm.mean())  if (len(e_at_pen)  > 0 and len(e_at_norm) > 0) else float('nan')
        naz  = float(np.mean(e_at_norm > 0))                if len(e_at_norm) > 0                          else float('nan')
        p90n = float(np.percentile(e_at_norm, 90))          if len(e_at_norm) > 0                          else float('nan')

        if params_arr.shape[1] > M:
            pi_raw  = params_arr[:, :M]
            pi_raw -= pi_raw.max(axis=1, keepdims=True)
            pi      = np.exp(pi_raw)
            pi     /= pi.sum(axis=1, keepdims=True)
            Hpi     = float(np.mean(-np.sum(pi * np.log(pi + 1e-8), axis=1)))

            sigma_raw = params_arr[:, 3 * M : 4 * M]
            smin      = float(np.min(sigma_raw))

    return {
        'stop_reason':     stop_reason,
        'n_pasos':         T,
        'drift_y_total':   float(abs_y[-1]) if len(abs_y) > 0 else 0.0,
        'drift_y_max':     float(np.max(np.abs(abs_y))) if len(abs_y) > 0 else 0.0,
        'drift_x_total':   float(abs_x[-1]) if len(abs_x) > 0 else 0.0,
        'dy_norm_mean':    float(np.mean(dy)),
        'dx_norm_mean':    float(np.mean(dx)),
        'pen_rate':        float(n_pen / max(T - 1, 1)),
        'pen_rate_x_true': float(n_pen / max(T - 1, 1)) / TRUE_PEN_RATE,
        'pen_gap_median':  pen_gap_median,
        'micro_pct':       micro_pct,
        'attn_cobertura':  attn_cobertura,
        'attn_monotonia':  attn_monotonia,
        'attn_llego_final': attn_llego_final,
        'kap_ov':          kap_ov,
        'eraw_mean':       eraw_mean,
        'eraw_pct_pos':    eraw_pct_pos,
        'eraw_p10':        eraw_p10,
        'eraw_p90':        eraw_p90,
        'sep':             sep,
        'naz':             naz,
        'p90n':            p90n,
        'Hpi':             Hpi,
        'smin':            smin,
    }


SCALAR_KEYS = [
    'n_pasos', 'drift_y_total', 'drift_y_max', 'drift_x_total',
    'dy_norm_mean', 'dx_norm_mean',
    'pen_rate', 'pen_rate_x_true', 'pen_gap_median', 'micro_pct',
    'attn_cobertura', 'attn_monotonia', 'attn_llego_final', 'kap_ov',
    'eraw_mean', 'eraw_pct_pos', 'eraw_p10', 'eraw_p90',
    'sep', 'naz', 'p90n', 'Hpi', 'smin',
]

MEAN_STD_KEYS = {
    'pen_rate_x_true', 'drift_y_total', 'dy_norm_mean', 'sep', 'naz',
}


def aggregate(sample_metrics: list[dict]) -> dict:
    n   = len(sample_metrics)
    agg = {}

    for k in SCALAR_KEYS:
        vals = [m[k] for m in sample_metrics if isinstance(m[k], (int, float)) and not np.isnan(m[k])]
        mean_v = float(np.mean(vals)) if vals else float('nan')
        agg[f'{k}_mean'] = mean_v
        if k in MEAN_STD_KEYS:
            agg[f'{k}_std'] = float(np.std(vals)) if len(vals) > 1 else float('nan')

    stop_counts = {}
    for m in sample_metrics:
        stop_counts[m['stop_reason']] = stop_counts.get(m['stop_reason'], 0) + 1

    agg['stop_phi_pct']    = stop_counts.get('phi',       0) / n
    agg['stop_length_pct'] = stop_counts.get('length',    0) / n
    agg['stop_max_pct']    = stop_counts.get('max_steps', 0) / n
    return agg


CSV_COLUMNS = [
    'bias', 'pen_bias',
    'pen_rate_x_true_mean', 'pen_rate_x_true_std',
    'drift_y_total_mean',   'drift_y_total_std',
    'dy_norm_mean_mean',    'dy_norm_mean_std',
    'sep_mean',             'sep_std',
    'naz_mean',             'naz_std',
    'eraw_pct_pos_mean', 'eraw_mean_mean', 'eraw_p10_mean', 'eraw_p90_mean',
    'p90n_mean',
    'Hpi_mean', 'smin_mean',
    'attn_cobertura_mean', 'attn_monotonia_mean', 'attn_llego_final_mean', 'kap_ov_mean',
    'micro_pct_mean', 'pen_gap_median_mean',
    'n_pasos_mean',
    'stop_phi_pct', 'stop_length_pct', 'stop_max_pct',
]


def fmt(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, float) and np.isnan(v):
        return 'nan'
    if isinstance(v, int):
        return str(v)
    return f'{v:.5f}'


def run_sweep(args, model: HandwritingGenerator, stats: dict, device: torch.device):
    bias_vals     = parse_range(args.bias_range)
    pen_bias_vals = parse_range(args.pen_bias_range)
    combinations  = list(product(bias_vals, pen_bias_vals))
    total         = len(combinations)

    print(f"Checkpoint  ep={stats['epoch']}  nll_s={stats['best_nll']:.4f}  vocab={len(stats['char_vocab'])}")
    print(f"Texto: '{args.texto}'  |  nsamples: {args.nsamples}")
    print(f"bias: {bias_vals}")
    print(f"pen_bias: {pen_bias_vals}")
    print(f"Combinaciones: {total}  |  Ciclos totales: {total * args.nsamples}\n")

    out_path = os.path.join(
        args.out_dir,
        f"sweep_{args.texto}_ep{stats['epoch']}.csv",
    )
    os.makedirs(args.out_dir, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()

        for done, (bias, pen_bias) in enumerate(combinations, 1):
            sample_metrics = []

            for i in range(args.nsamples):
                if args.seed is not None:
                    torch.manual_seed(args.seed + i)
                    np.random.seed(args.seed + i)
                    random.seed(args.seed + i)

                strokes, attentions, params_arr, stop_reason = generate_one(
                    model, args.texto, device,
                    bias, pen_bias, args.max_steps, args.steps_per_char,
                )
                m = compute_sample_metrics(
                    strokes, attentions, params_arr,
                    stats['std_dx'], stats['std_dy'],
                    stop_reason, model.M,
                )
                sample_metrics.append(m)

            agg = aggregate(sample_metrics)
            row = {'bias': fmt(bias), 'pen_bias': fmt(pen_bias)}
            row.update({k: fmt(agg.get(k, float('nan'))) for k in CSV_COLUMNS if k not in ('bias', 'pen_bias')})
            writer.writerow(row)
            fh.flush()

            pen_x  = agg.get('pen_rate_x_true_mean', float('nan'))
            drift  = agg.get('drift_y_total_mean',   float('nan'))
            cob    = agg.get('attn_cobertura_mean',  float('nan'))
            sep_v  = agg.get('sep_mean',             float('nan'))
            naz_v  = agg.get('naz_mean',             float('nan'))
            phi_p  = agg.get('stop_phi_pct',         float('nan'))

            alarm = ''
            if not np.isnan(pen_x)  and pen_x  > 3.0:   alarm += ' [ALARM:pen×>3]'
            if not np.isnan(sep_v)  and sep_v  < 2.0:   alarm += ' [ALARM:sep<2]'
            if not np.isnan(naz_v)  and naz_v  > 0.05:  alarm += ' [ALARM:naz>5%]'

            print(
                f"[{done:3d}/{total}] bias={bias:.1f} pen_bias={pen_bias:+.1f} | "
                f"pen×={pen_x:.2f}  drift_y={drift:+.0f}px  "
                f"sep={sep_v:.2f}  naz={naz_v:.3f}  "
                f"cob={cob:.2f}  phi%={phi_p:.0%}"
                f"{alarm}"
            )

    print(f"\n→ {out_path}")
    print(f"  {total} filas × {len(CSV_COLUMNS)} columnas")


def main():
    parser = argparse.ArgumentParser(
        description='Sweep bias × pen_bias para diagnóstico de inferencia'
    )
    parser.add_argument('--checkpoint',     type=str,   default='./modelos/model4.pt')
    parser.add_argument('--texto',          type=str,   default='hola')
    parser.add_argument('--nsamples',       type=int,   default=10)
    parser.add_argument('--bias_range',     type=str,   default='2.0,4.0,0.5',
                        help='start,stop,step  (ej: 2.0,4.0,0.5)')
    parser.add_argument('--pen_bias_range', type=str,   default='-3.0,3.0,0.5',
                        help='start,stop,step  (ej: -3.0,3.0,0.5)')
    parser.add_argument('--max_steps',      type=int,   default=2000)
    parser.add_argument('--steps_per_char', type=int,   default=80)
    parser.add_argument('--device',         type=str,   default='auto')
    parser.add_argument('--out_dir',        type=str,   default='./audit_results')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    device = (
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto'
        else torch.device(args.device)
    )

    model, stats = load_checkpoint(args.checkpoint, device)
    run_sweep(args, model, stats, device)


if __name__ == '__main__':
    main()