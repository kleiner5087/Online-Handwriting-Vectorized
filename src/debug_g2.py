"""
audit_generate.py — Auditoría de inferencia para HandwritingGenerator

Ejecutar:
    python audit_generate.py --checkpoint ./modelos/model1.pt --texto hola --nsamples 20

Salidas en ./audit_results/:
    audit_<texto>_ep<N>.csv    — métricas por muestra (una fila por generación)
    audit_<texto>_ep<N>.png    — panel diagnóstico: distribuciones + ejemplos visuales
    audit_<texto>_ep<N>_raw.npz — arrays raw (strokes, attentions) de todas las muestras

Ejes de auditoría:
    1. DRIFT      — ¿se desplazan los trazos verticalmente?
    2. PEN-LIFTS  — ¿cuántos hay, cómo se distribuyen, cómo de aislados?
    3. ATENCIÓN   — ¿avanza kappa?, ¿cubre todos los caracteres?, ¿causa de paro?
    4. MDN RAW    — distribución de e_raw, sigma, pi antes del muestreo
    5. TRAZOS     — longitud media, proporción de microtrazos, aspecto visual
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
sys.path.insert(0, '.')
from src.model import HandwritingGenerator, parse_mdn_params, sample_from_mdn

TRUE_PEN_RATE  = 0.0296


# ─── Carga ────────────────────────────────────────────────────────────────────
def cargar_modelo(ckpt_path, device):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    char_vocab = ckpt['char_vocab']
    std_dx     = float(ckpt['std_dx'])
    std_dy     = float(ckpt['std_dy'])
    mean_dx    = float(ckpt.get('mean_dx', 0.0))
    mean_dy    = float(ckpt.get('mean_dy', 0.0))
    epoch      = ckpt.get('epoch', 0)
    best_nll   = ckpt.get('best_nll', float('nan'))

    model = HandwritingGenerator(char_vocab).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"Checkpoint: ep={epoch}  nll_s={best_nll:.4f}  vocab={len(char_vocab)}")
    print(f"Stats:  std_dx={std_dx:.3f}  std_dy={std_dy:.3f}  "
          f"mean_dx={mean_dx:.3f}  mean_dy={mean_dy:.3f}")
    return model, char_vocab, std_dx, std_dy, mean_dx, mean_dy, epoch


# ─── Generación instrumentada ─────────────────────────────────────────────────
@torch.no_grad()
def generar_auditado(model, texto, device, std_dy, bias, pen_bias,
                     max_steps, steps_per_char):
    """
    Igual que generate.py::generar() pero guarda todos los params raw
    antes del muestreo para auditar la distribución MDN.
    """
    char_embeds = model.char_embed(model.encode_text([texto], device))
    U           = char_embeds.shape[1]

    model.attention.reset(1, device)
    h1     = model._zero_hidden(1, device)
    h2     = model._zero_hidden(1, device)
    window = torch.zeros(1, model.embed_dim, device=device)
    x_t    = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    strokes     = [np.array([0., 0., 1.], dtype=np.float32)]
    attentions  = []
    params_raw  = []          # [T, param_dim] — para auditoría MDN
    stop_reason = 'max_steps'
    expected    = len(texto.replace(' ', '')) * steps_per_char

    for step in range(max_steps):
        inp1        = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1      = model.lstm1(inp1, h1)
        o1          = model.norm1(o1.squeeze(1))
        window, phi = model.attention(o1, char_embeds)
        attentions.append(phi.squeeze(0).cpu().numpy())

        phi_n      = phi.squeeze(0) / (phi.squeeze(0).sum() + 1e-8)
        last_ratio = phi_n[-1].item()
        if (len(phi_n) > 1 and last_ratio > phi_n[:-1].max().item() and last_ratio > 0.6):
            stop_reason = 'phi'
            break
        if step > expected:
            stop_reason = 'length'
            break

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        features = torch.cat([o1, o2], dim=1)
        params   = torch.cat([model.mdn_head(features), model.pen_head(features)], dim=-1).squeeze(0)
        params_raw.append(params.cpu().numpy())

        if pen_bias != 0.0:
            params = params.clone()
            params[-1] = params[-1] - pen_bias

        x_t = sample_from_mdn(params, M=model.M, bias=bias).unsqueeze(0)
        if x_t[0, 2].item() > 0.5:
            x_t = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        strokes.append(x_t.squeeze(0).cpu().numpy())

    return (np.array(strokes, dtype=np.float32),
            np.array(attentions, dtype=np.float32),
            np.array(params_raw, dtype=np.float32) if params_raw else np.zeros((0,1)),
            stop_reason,
            U)


# ─── Métricas por muestra ─────────────────────────────────────────────────────
def calcular_metricas(strokes, attentions, params_raw, texto, std_dx, std_dy,
                      stop_reason, U):
    """
    Retorna dict con todas las métricas auditables. Todas son escalares o
    arrays pequeños para poder volcarlos en CSV / comparar entre muestras.
    """
    m = {}
    T = len(strokes)
    m['n_pasos']     = T
    m['stop_reason'] = stop_reason

    dx = strokes[1:, 0]   # normalizado
    dy = strokes[1:, 1]
    pen= strokes[1:, 2]

    # ── 1. DRIFT ──────────────────────────────────────────────────────────────
    mean_dx = stats.get('mean_dx', 0.0)
    mean_dy = stats.get('mean_dy', 0.0)
    abs_x = np.cumsum(dx * std_dx + mean_dx)
    abs_y = np.cumsum(dy * std_dy + mean_dy)

    m['drift_y_total']   = float(abs_y[-1]) if len(abs_y) > 0 else 0.
    m['drift_y_max']     = float(np.max(np.abs(abs_y))) if len(abs_y) > 0 else 0.
    m['drift_x_total']   = float(abs_x[-1]) if len(abs_x) > 0 else 0.
    m['dy_norm_mean']    = float(np.mean(dy))     # debe ser ~0 si datos centrados
    m['dy_norm_std']     = float(np.std(dy))
    m['dx_norm_mean']    = float(np.mean(dx))

    # ── 2. PEN-LIFTS ──────────────────────────────────────────────────────────
    pen_pos = np.where(pen > 0.5)[0]
    n_pen   = len(pen_pos)
    m['n_pen_lifts']     = n_pen
    m['pen_rate']        = float(n_pen / max(T-1, 1))
    m['pen_rate_x_true'] = m['pen_rate'] / TRUE_PEN_RATE

    if n_pen > 1:
        gaps = np.diff(pen_pos)
        m['pen_gap_mean']  = float(np.mean(gaps))
        m['pen_gap_min']   = float(np.min(gaps))
        m['pen_gap_median']= float(np.median(gaps))
        # Microtrazos: pen-lifts con gap < 5 pasos = trazos casi vacíos
        m['micro_pct']     = float(np.mean(gaps < 5))
    else:
        m['pen_gap_mean']  = float('nan')
        m['pen_gap_min']   = float('nan')
        m['pen_gap_median']= float('nan')
        m['micro_pct']     = float('nan')

    # ── 3. ATENCIÓN ───────────────────────────────────────────────────────────
    # kappa_final = posición final de la ventana de atención
    # phi_peak_per_step: índice del carácter con mayor atención en cada paso
    if len(attentions) > 0:
        phi_peaks  = np.argmax(attentions, axis=1)          # [T]
        kappa_last = float(attentions[-1].argmax())
        n_chars    = attentions.shape[1]

        # Cobertura: ¿qué fracción de los U caracteres fue el char dominante al menos 1 paso?
        chars_visitados = len(set(phi_peaks))
        m['attn_cobertura'] = float(chars_visitados / max(n_chars, 1))

        # Monotonía: fracción de pasos en que kappa_peak avanzó o se mantuvo
        m['attn_monotonia'] = float(np.mean(np.diff(phi_peaks) >= 0)) if len(phi_peaks) > 1 else float('nan')

        # Pasos por carácter: promedio de pasos que cada carácter fue dominante
        char_counts = np.bincount(phi_peaks, minlength=n_chars)
        m['attn_pasos_x_char_mean'] = float(np.mean(char_counts[char_counts > 0]))
        m['attn_pasos_x_char_min']  = float(np.min(char_counts))   # 0 = carácter nunca dominante

        # ¿La atención llegó al último carácter?
        m['attn_llego_final'] = int(phi_peaks[-1] == n_chars - 1)
        m['kappa_last_char']  = float(kappa_last)
    else:
        for k in ['attn_cobertura','attn_monotonia','attn_pasos_x_char_mean',
                  'attn_pasos_x_char_min','attn_llego_final','kappa_last_char']:
            m[k] = float('nan')

    # ── 4. MDN RAW ────────────────────────────────────────────────────────────
    if params_raw.shape[0] > 0:
        e_raw    = params_raw[:, -1]
        M        = model_M_global
        sigma_x  = params_raw[:, 3*M:4*M]    # antes de softplus — log-escala aprox

        m['eraw_mean']       = float(np.mean(e_raw))
        m['eraw_std']        = float(np.std(e_raw))
        m['eraw_pct_pos']    = float(np.mean(e_raw > 0))     # naz equivalente en inferencia
        m['eraw_p10']        = float(np.percentile(e_raw, 10))
        m['eraw_p90']        = float(np.percentile(e_raw, 90))

        # sigma bruto (pre-softplus): si muy negativo → sigma muy pequeño → MDN colapsa
        m['sigma_raw_mean']  = float(np.mean(sigma_x))
        m['sigma_raw_min']   = float(np.min(sigma_x))
    else:
        for k in ['eraw_mean','eraw_std','eraw_pct_pos','eraw_p10','eraw_p90',
                  'sigma_raw_mean','sigma_raw_min']:
            m[k] = float('nan')

    return m


# ─── Panel visual ─────────────────────────────────────────────────────────────
def guardar_panel(texto, todas_muestras, std_dx, std_dy, out_path, params_list):
    """
    Panel de 3 filas:
      Fila 1: 4 ejemplos de trazos reconstruidos
      Fila 2: distribución de e_raw + distribución dy_norm
      Fila 3: drift_y_total por muestra + pen_rate por muestra
    """
    N  = len(todas_muestras)
    metricas = [m for _, _, _, m in todas_muestras]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)

    # ── Fila 1: 4 trazos ──────────────────────────────────────────────────────
    indices_mostrar = np.linspace(0, N-1, 4, dtype=int)
    for col, idx in enumerate(indices_mostrar):
        ax  = fig.add_subplot(gs[0, col])
        strokes, attentions, _, m = todas_muestras[idx]
        abs_x = np.concatenate([[0], np.cumsum(strokes[1:, 0] * std_dx)])
        abs_y = np.concatenate([[0], np.cumsum(strokes[1:, 1] * std_dy)])
        pen   = strokes[1:, 2]

        start = 0
        for t in range(len(pen)):
            if pen[t] > 0.5 or t == len(pen) - 1:
                end = t + 1
                ax.plot(abs_x[start:end+1], abs_y[start:end+1], 'k-', lw=1.2)
                start = end + 1

        drift = m['drift_y_total']
        ax.set_title(f"#{idx+1}  drift_y={drift:.0f}px\n"
                     f"pen_x={m['pen_rate_x_true']:.1f}×  "
                     f"cob={m['attn_cobertura']:.2f}", fontsize=7)
        ax.invert_yaxis()
        ax.axis('equal')
        ax.axis('off')

    # ── Fila 2: distribuciones ────────────────────────────────────────────────
    # e_raw de todos los pasos de todas las muestras
    ax_eraw = fig.add_subplot(gs[1, :2])
    all_eraw = np.concatenate([p[:, -1] for p in params_list if p.shape[0] > 0])
    ax_eraw.hist(all_eraw, bins=60, color='steelblue', alpha=0.8, edgecolor='none')
    ax_eraw.axvline(0,    color='red',    lw=1.5, label='umbral pen_bias=0')
    ax_eraw.axvline(-2.0, color='orange', lw=1.2, ls='--', label='pen_bias=-2.0 (~1×)')
    ax_eraw.axvline(-3.0, color='green',  lw=1.2, ls='--', label='pen_bias=-3.0 (sweet spot)')
    pct_pos = float(np.mean(all_eraw > 0)) * 100
    ax_eraw.set_title(f"Distribución e_raw  (N={len(all_eraw)})  "
                      f"pct>0: {pct_pos:.1f}%", fontsize=8)
    ax_eraw.set_xlabel("e_raw (logit pen-lift)")
    ax_eraw.legend(fontsize=7)

    # dy_norm de todas las muestras
    ax_dy = fig.add_subplot(gs[1, 2:])
    all_dy = np.concatenate([s[1:, 1] for s, _, _, _ in todas_muestras])
    ax_dy.hist(all_dy, bins=60, color='darkorange', alpha=0.8, edgecolor='none')
    ax_dy.axvline(0, color='red', lw=1.5)
    dy_mean = float(np.mean(all_dy))
    dy_std  = float(np.std(all_dy))
    ax_dy.set_title(f"Distribución dy_norm  mean={dy_mean:.4f}  std={dy_std:.3f}\n"
                    f"(drift si mean≠0: {dy_mean * std_dy:.2f} px/paso)", fontsize=8)
    ax_dy.set_xlabel("dy normalizado (esperado: media≈0 con Z-norm)")

    # ── Fila 3: métricas por muestra ─────────────────────────────────────────
    ax_drift  = fig.add_subplot(gs[2, :2])
    ax_penx   = fig.add_subplot(gs[2, 2:])

    drift_vals = [m['drift_y_total']   for m in metricas]
    penx_vals  = [m['pen_rate_x_true'] for m in metricas]
    cob_vals   = [m['attn_cobertura']  for m in metricas]
    ep_idxs    = list(range(1, N+1))

    ax_drift.bar(ep_idxs, drift_vals, color=['tomato' if abs(d) > 200 else 'steelblue'
                                              for d in drift_vals])
    ax_drift.axhline(0,   color='k',   lw=0.8)
    ax_drift.axhline(200, color='red', lw=1, ls='--', label='±200px')
    ax_drift.axhline(-200,color='red', lw=1, ls='--')
    ax_drift.set_title(f"Drift Y total por muestra  "
                       f"(media={np.mean(drift_vals):.0f}px, "
                       f"std={np.std(drift_vals):.0f}px)", fontsize=8)
    ax_drift.set_xlabel("Muestra #")
    ax_drift.set_ylabel("px")
    ax_drift.legend(fontsize=7)

    bars = ax_penx.bar(ep_idxs, penx_vals,
                       color=['tomato' if v > 3 else 'steelblue' for v in penx_vals])
    ax_penx.axhline(1.0, color='green', lw=1.2, ls='--', label='tasa real (1×)')
    ax_penx.axhline(3.0, color='red',   lw=1,   ls='--', label='alarma (3×)')

    ax_penx2 = ax_penx.twinx()
    ax_penx2.plot(ep_idxs, cob_vals, 'o-', color='purple', ms=4, lw=1.2, label='cobertura atención')
    ax_penx2.set_ylim(0, 1.1)
    ax_penx2.set_ylabel("Cobertura atención", fontsize=7, color='purple')
    ax_penx2.tick_params(axis='y', labelcolor='purple', labelsize=7)

    ax_penx.set_title(f"pen_rate × real  +  cobertura atención", fontsize=8)
    ax_penx.set_xlabel("Muestra #")
    ax_penx.set_ylabel("pen_rate / tasa_real")
    lines1, labels1 = ax_penx.get_legend_handles_labels()
    lines2, labels2 = ax_penx2.get_legend_handles_labels()
    ax_penx.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle(f"Auditoría generación  |  texto='{texto}'  |  N={N} muestras", fontsize=10)
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Panel: {os.path.basename(out_path)}")


# ─── CSV ──────────────────────────────────────────────────────────────────────
CSV_KEYS = [
    'n_pasos', 'stop_reason',
    'drift_y_total', 'drift_y_max', 'drift_x_total', 'dy_norm_mean', 'dy_norm_std',
    'dx_norm_mean',
    'n_pen_lifts', 'pen_rate', 'pen_rate_x_true',
    'pen_gap_mean', 'pen_gap_min', 'pen_gap_median', 'micro_pct',
    'attn_cobertura', 'attn_monotonia', 'attn_pasos_x_char_mean',
    'attn_pasos_x_char_min', 'attn_llego_final', 'kappa_last_char',
    'eraw_mean', 'eraw_std', 'eraw_pct_pos', 'eraw_p10', 'eraw_p90',
    'sigma_raw_mean', 'sigma_raw_min',
]

def _f(v):
    if isinstance(v, str): return v
    if isinstance(v, float) and (v != v): return 'nan'  # nan check
    if isinstance(v, int): return str(v)
    return f'{v:.4f}'

def guardar_csv(filas_metricas, out_path):
    with open(out_path, 'w') as f:
        f.write('muestra,' + ','.join(CSV_KEYS) + '\n')
        for i, m in enumerate(filas_metricas):
            row = [str(i+1)] + [_f(m.get(k, float('nan'))) for k in CSV_KEYS]
            f.write(','.join(row) + '\n')
    print(f"  CSV: {os.path.basename(out_path)}")


def imprimir_resumen(filas, texto, bias, pen_bias, std_dy):
    import statistics

    def mn(k):
        vals = [m[k] for m in filas if isinstance(m.get(k), (int, float))
                and m[k] == m[k]]
        return statistics.mean(vals) if vals else float('nan')

    print(f"\n{'─'*60}")
    print(f"RESUMEN  texto='{texto}'  bias={bias}  pen_bias={pen_bias}  N={len(filas)}")
    print(f"{'─'*60}")
    print(f"  DRIFT")
    print(f"    drift_y_total  media={mn('drift_y_total'):+.0f}px  "
          f"[min={min(m['drift_y_total'] for m in filas):+.0f}  "
          f"max={max(m['drift_y_total'] for m in filas):+.0f}]")
    print(f"    dy_norm_mean   {mn('dy_norm_mean'):.5f}  "
          f"(drift por paso: {mn('dy_norm_mean') * std_dy:.2f}px aprox)")
    print(f"  PEN-LIFTS")
    print(f"    pen_rate_x_true {mn('pen_rate_x_true'):.2f}×  "
          f"(alarma >3×, ideal ~1×)")
    print(f"    pen_gap_median  {mn('pen_gap_median'):.1f} pasos entre lifts")
    print(f"    micro_pct       {mn('micro_pct')*100:.1f}%  (<5 pasos = microtrazo)")
    print(f"  ATENCIÓN")
    print(f"    cobertura       {mn('attn_cobertura'):.3f}  (ideal=1.0)")
    print(f"    monotonía       {mn('attn_monotonia'):.3f}  (ideal>0.95)")
    print(f"    llego_final     {sum(int(m['attn_llego_final']) for m in filas)}/{len(filas)}")
    print(f"  MDN RAW")
    print(f"    e_raw_mean      {mn('eraw_mean'):.3f}  e_raw_pct>0: {mn('eraw_pct_pos')*100:.1f}%")
    print(f"    e_raw p10/p90   {mn('eraw_p10'):.2f} / {mn('eraw_p90'):.2f}")
    s_reasons = {}
    for m in filas:
        s_reasons[m['stop_reason']] = s_reasons.get(m['stop_reason'], 0) + 1
    print(f"  PARO:  {s_reasons}")
    print(f"{'─'*60}")


# ─── Main ──────────────────────────────────────────────────────────────────────
model_M_global = 10   # se sobreescribe en main

def main():
    global model_M_global

    parser = argparse.ArgumentParser(description="Auditoría de generación")
    parser.add_argument('--checkpoint',     type=str,   default='./modelos/model4.pt')
    parser.add_argument('--texto',          type=str,   default='hola')
    parser.add_argument('--nsamples',       type=int,   default=20)
    parser.add_argument('--bias',           type=float, default=1.0)
    parser.add_argument('--pen_bias',       type=float, default=0.0)
    parser.add_argument('--max_steps',      type=int,   default=2000)
    parser.add_argument('--steps_per_char', type=int,   default=80)
    parser.add_argument('--device',         type=str,   default='auto')
    parser.add_argument('--out_dir',        type=str,   default='./audit_results')
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Semilla para reproducibilidad. Cada muestra usa seed+i.')
    args = parser.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    os.makedirs(args.out_dir, exist_ok=True)

    model, char_vocab, std_dx, std_dy, mean_dx, mean_dy, epoch = \
        cargar_modelo(args.checkpoint, device)

    model_M_global = model.M

    # Validar que el texto es generablem
    chars_ok = [ch for ch in args.texto if ch in char_vocab or ch == ' ']
    if len(chars_ok) < len(args.texto):
        missing = [ch for ch in args.texto if ch not in char_vocab and ch != ' ']
        print(f"[WARN] Caracteres fuera de vocab: {missing}")

    seed_info = f"  seed={args.seed}" if args.seed is not None else "  seed=None (no reproducible)"
    print(f"\nGenerando {args.nsamples} muestras de '{args.texto}'...{seed_info}")

    todas_muestras = []
    params_list    = []

    for i in range(args.nsamples):
        if args.seed is not None:
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)
            random.seed(args.seed + i)

        strokes, attentions, params_raw, stop_reason, U = generar_auditado(
            model, args.texto, device,
            std_dy=std_dy, bias=args.bias, pen_bias=args.pen_bias,
            max_steps=args.max_steps, steps_per_char=args.steps_per_char,
        )
        m = calcular_metricas(strokes, attentions, params_raw, args.texto,
                              std_dx, std_dy, stop_reason, U)
        todas_muestras.append((strokes, attentions, params_raw, m))
        params_list.append(params_raw)
        print(f"  [{i+1:02d}/{args.nsamples}]  pasos={m['n_pasos']}  "
              f"drift_y={m['drift_y_total']:+.0f}px  "
              f"pen×={m['pen_rate_x_true']:.1f}  "
              f"cob={m['attn_cobertura']:.2f}  "
              f"stop={stop_reason}")

    stem = f"audit_{args.texto}_ep{epoch}"
    metricas = [m for _, _, _, m in todas_muestras]

    guardar_csv(metricas, os.path.join(args.out_dir, stem + '.csv'))
    guardar_panel(args.texto, todas_muestras, std_dx, std_dy,
                  os.path.join(args.out_dir, stem + '.png'), params_list)

    # Guardar arrays raw para inspección posterior
    np.savez_compressed(
        os.path.join(args.out_dir, stem + '_raw.npz'),
        strokes   = np.array([s for s, _, _, _ in todas_muestras], dtype=object),
        attentions= np.array([a for _, a, _, _ in todas_muestras], dtype=object),
    )
    print(f"  Raw: {stem}_raw.npz")

    imprimir_resumen(metricas, args.texto, args.bias, args.pen_bias, std_dy)


if __name__ == '__main__':
    main()