import argparse
import os
import sys
import math
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, '.')
from src.model import HandwritingGenerator, sample_from_mdn

TRUE_PEN_RATE = 0.0535
STROKE_COLOR  = '#1a1a1a'
BG_COLOR      = '#fafaf8'


def load_checkpoint(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        sys.exit(f"[ERROR] Checkpoint no encontrado: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
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
def generate_deltas(
    model:          HandwritingGenerator,
    texto:          str,
    device:         torch.device,
    bias:           float,
    pen_bias:       float,
    max_steps:      int,
    steps_per_char: int,
    seed:           int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if not texto.endswith('#'):
        texto = texto + '#'

    char_embeds = model.char_embed(model.encode_text([texto], device))
    U           = char_embeds.shape[1]
    expected    = len(texto.replace(' ', '')) * steps_per_char

    model.attention.reset(1, device)
    h1     = model._zero_hidden(1, device)
    h2     = model._zero_hidden(1, device)
    window = torch.zeros(1, model.embed_dim, device=device)
    x_t    = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    strokes_out    = [np.array([0., 0., 1.], dtype=np.float32)]
    attentions_out = []
    stop_reason    = 'max_steps'

    for step in range(max_steps):
        inp1        = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1      = model.lstm1(inp1, h1)
        o1          = model.norm1(o1.squeeze(1))
        window, phi = model.attention(o1, char_embeds)

        if torch.argmax(phi.squeeze(0)).item() == U - 1:
            stop_reason = 'phi'
            break
        if step > expected:
            stop_reason = 'length'
            break

        attentions_out.append(phi.squeeze(0).cpu().numpy())

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        features   = torch.cat([o1, o2, window], dim=1)
        raw_params = torch.cat(
            [model.mdn_head(features), model.pen_head(features)], dim=-1
        ).squeeze(0)

        sample_params      = raw_params.clone()
        sample_params[-1] -= pen_bias
        x_t = sample_from_mdn(sample_params, M=model.M, bias=bias).unsqueeze(0)

        strokes_out.append(x_t.squeeze(0).cpu().numpy())

    strokes    = np.array(strokes_out, dtype=np.float32)
    attentions = (
        np.array(attentions_out, dtype=np.float32)
        if attentions_out
        else np.zeros((0, U), dtype=np.float32)
    )
    return strokes, attentions, stop_reason

def deltas_to_strokes(
    deltas:  np.ndarray,
    std_dx:  float,
    std_dy:  float,
    mean_dx: float,
    mean_dy: float,
) -> list[np.ndarray]:

    dx  = deltas[1:, 0]
    dy  = deltas[1:, 1]
    pen = deltas[1:, 2]

    abs_dx = dx * std_dx + mean_dx
    abs_dy = dy * std_dy + mean_dy

    abs_x = np.concatenate(([0.0], np.cumsum(abs_dx)))
    abs_y = np.concatenate(([0.0], np.cumsum(abs_dy)))
    
    pen = np.concatenate(([0.0], pen))

    strokes = []
    cur_x   = []
    cur_y   = []

    for x, y, p in zip(abs_x, abs_y, pen):
        cur_x.append(float(x))
        cur_y.append(float(y))

        if p > 0.5:
            if len(cur_x) >= 2:
                strokes.append(np.column_stack([cur_x, cur_y]))
            cur_x = []
            cur_y = []

    if len(cur_x) >= 2:
        strokes.append(np.column_stack([cur_x, cur_y]))

    return strokes


def quick_metrics(deltas: np.ndarray, attentions: np.ndarray, stop_reason: str) -> dict:
    pen   = deltas[1:, 2]
    T     = len(pen)
    n_pen = int(np.sum(pen > 0.5))
    pen_rate_x = (n_pen / max(T, 1)) / TRUE_PEN_RATE

    attn_cob = attn_mono = float('nan')
    if attentions.shape[0] > 0:
        peaks    = np.argmax(attentions, axis=1)
        n_chars  = attentions.shape[1]
        attn_cob = len(set(peaks)) / max(n_chars, 1)
        attn_mono = float(np.mean(np.diff(peaks) >= 0)) if len(peaks) > 1 else float('nan')

    return dict(
        pasos      = T,
        n_trazos   = n_pen,
        pen_rate_x = pen_rate_x,
        stop       = stop_reason,
        attn_cob   = attn_cob,
        attn_mono  = attn_mono,
    )


def render_strokes_on_ax(
    ax:        plt.Axes,
    strokes:   list[np.ndarray],
    title:     str   = '',
    linewidth: float = 1.6,
    color:     str   = STROKE_COLOR,
    pad_frac:  float = 0.08,
):
    if not strokes:
        ax.set_visible(False)
        return

    all_x = np.concatenate([s[:, 0] for s in strokes])
    all_y = np.concatenate([s[:, 1] for s in strokes])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    px, py = w * pad_frac, h * pad_frac

    for stroke in strokes:
        ax.plot(stroke[:, 0], stroke[:, 1], '-', color=color,
                linewidth=linewidth, solid_capstyle='round',
                solid_joinstyle='round', antialiased=True)

    ax.set_xlim(x_min - px, x_max + px)
    ax.set_ylim(y_max + py, y_min - py)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=9, pad=4, color='#444444')


def export_svg(
    strokes:  list[np.ndarray],
    out_path: str,
    padding:  float = 20.0,
    stroke_w: float = 2.0,
    color:    str   = '#1a1a1a',
):
    if not strokes:
        return

    all_x = np.concatenate([s[:, 0] for s in strokes])
    all_y = np.concatenate([s[:, 1] for s in strokes])
    x_min, y_min = all_x.min() - padding, all_y.min() - padding
    w = all_x.max() - all_x.min() + 2 * padding
    h = all_y.max() - all_y.min() + 2 * padding

    svg = ET.Element('svg', {
        'xmlns':   'http://www.w3.org/2000/svg',
        'width':   f'{w:.1f}',
        'height':  f'{h:.1f}',
        'viewBox': f'0 0 {w:.1f} {h:.1f}',
    })
    ET.SubElement(svg, 'rect', {
        'width': '100%', 'height': '100%', 'fill': BG_COLOR,
    })

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts = ' '.join(
            f'{x - x_min:.2f},{y - y_min:.2f}'
            for x, y in stroke
        )
        ET.SubElement(svg, 'polyline', {
            'points':          pts,
            'fill':            'none',
            'stroke':          color,
            'stroke-width':    str(stroke_w),
            'stroke-linecap':  'round',
            'stroke-linejoin': 'round',
        })

    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(out_path, encoding='unicode', xml_declaration=True)


def mode_single(args, model, stats, device):
    deltas, attentions, stop = generate_deltas(
        model, args.texto, device,
        args.bias, args.pen_bias,
        args.max_steps, args.steps_per_char,
        args.seed,
    )
    strokes = deltas_to_strokes(
        deltas, stats['std_dx'], stats['std_dy'],
        stats['mean_dx'], stats['mean_dy'],
    )
    m = quick_metrics(deltas, attentions, stop)

    show_attn = (not args.no_attn) and attentions.shape[0] > 0

    if show_attn:
        fig = plt.figure(figsize=(12, 5), facecolor=BG_COLOR)
        gs  = gridspec.GridSpec(
            2, 1, height_ratios=[3, 1], hspace=0.12,
            left=0.03, right=0.97, top=0.90, bottom=0.08,
        )
        ax_ink  = fig.add_subplot(gs[0])
        ax_attn = fig.add_subplot(gs[1])
    else:
        fig, ax_ink = plt.subplots(1, 1, figsize=(12, 4), facecolor=BG_COLOR)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
        ax_attn = None

    render_strokes_on_ax(ax_ink, strokes)

    diag = (
        f"pasos={m['pasos']}  trazos={m['n_trazos']}  "
        f"penx={m['pen_rate_x']:.2f}  "
        f"cob={m['attn_cob']:.2f}  mono={m['attn_mono']:.2f}  "
        f"stop={m['stop']}"
    )
    fig.suptitle(
        f'"{args.texto}"   ep={stats["epoch"]}   '
        f'bias={args.bias}  pen_bias={args.pen_bias:+.1f}\n{diag}',
        fontsize=9, y=0.97, color='#333333',
    )

    if show_attn and ax_attn is not None:
        ax_attn.imshow(
            attentions.T,
            aspect='auto', cmap='Blues', interpolation='nearest',
            origin='upper',
        )
        ax_attn.set_ylabel('char', fontsize=7)
        ax_attn.set_xlabel('paso', fontsize=7)
        ax_attn.set_yticks(range(len(args.texto) + 1))
        ax_attn.set_yticklabels(list(args.texto) + ['#'], fontsize=7)
        ax_attn.tick_params(labelsize=6)

    out_base = _out_base(args, args.texto)
    fig.savefig("texto" + '.png', dpi=args.dpi, bbox_inches='tight',
                facecolor=BG_COLOR)
    plt.close(fig)

    if args.svg:
        export_svg(strokes, out_base + '.svg')


def mode_grid(args, model, stats, device):
    n    = args.n
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.2),
                             facecolor=BG_COLOR)
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor(BG_COLOR)

    pen_rates = []
    for i in range(n):
        seed_i = (args.seed + i) if args.seed is not None else None
        deltas, attentions, stop = generate_deltas(
            model, args.texto, device,
            args.bias, args.pen_bias,
            args.max_steps, args.steps_per_char,
            seed_i,
        )
        strokes = deltas_to_strokes(
            deltas, stats['std_dx'], stats['std_dy'],
            stats['mean_dx'], stats['mean_dy'],
        )
        m = quick_metrics(deltas, attentions, stop)
        pen_rates.append(m['pen_rate_x'])

        title = f"#{i+1}  penx={m['pen_rate_x']:.2f}  {m['stop']}"
        render_strokes_on_ax(axes[i], strokes, title=title, linewidth=1.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    pen_mean = float(np.mean(pen_rates))
    pen_std  = float(np.std(pen_rates))
    fig.suptitle(
        f'"{args.texto}"  x{n}  ep={stats["epoch"]}  '
        f'bias={args.bias}  pen_bias={args.pen_bias:+.1f}  '
        f'penx={pen_mean:.2f}+-{pen_std:.2f}',
        fontsize=9, y=1.01, color='#333333',
    )
    fig.tight_layout(pad=0.8)

    out_base = _out_base(args, f'{args.texto}_grid{n}')
    fig.savefig(out_base + '.png', dpi=args.dpi, bbox_inches='tight',
                facecolor=BG_COLOR)
    plt.close(fig)


def mode_compare(args, model, stats, device):
    textos = args.textos
    n      = len(textos)

    fig, axes = plt.subplots(1, n, figsize=(max(n * 3.5, 7), 3.0),
                             facecolor=BG_COLOR)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG_COLOR)

    for ax, texto in zip(axes, textos):
        deltas, attentions, stop = generate_deltas(
            model, texto, device,
            args.bias, args.pen_bias,
            args.max_steps, args.steps_per_char,
            args.seed,
        )
        strokes = deltas_to_strokes(
            deltas, stats['std_dx'], stats['std_dy'],
            stats['mean_dx'], stats['mean_dy'],
        )
        m     = quick_metrics(deltas, attentions, stop)
        title = f'"{texto}"\npenx={m["pen_rate_x"]:.2f}  {m["stop"]}'
        render_strokes_on_ax(ax, strokes, title=title)

    fig.suptitle(
        f'ep={stats["epoch"]}  bias={args.bias}  pen_bias={args.pen_bias:+.1f}',
        fontsize=9, y=1.03, color='#333333',
    )
    fig.tight_layout(pad=1.0)

    slug     = '_'.join(t[:6] for t in textos)
    out_base = _out_base(args, f'compare_{slug}')
    fig.savefig(out_base + '.png', dpi=args.dpi, bbox_inches='tight',
                facecolor=BG_COLOR)
    plt.close(fig)


def _out_base(args, slug: str) -> str:
    os.makedirs(args.out, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in slug)
    ep   = args._stats_epoch if hasattr(args, '_stats_epoch') else 0
    return os.path.join(args.out, f'{safe}_ep{ep}_b{args.bias}_pb{args.pen_bias:+.1f}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      default='./modelos/model12.pt')
    p.add_argument('--texto',           default='hola')
    p.add_argument('--textos',          nargs='+', default=None)
    p.add_argument('--bias',            type=float, default=3.5)
    p.add_argument('--pen_bias',        type=float, default=0.5)
    p.add_argument('--mode',            choices=['single', 'grid', 'compare'],
                   default='single')
    p.add_argument('--n',               type=int, default=9)
    p.add_argument('--steps_per_char',  type=int, default=100)
    p.add_argument('--max_steps',       type=int, default=2000)
    p.add_argument('--seed',            type=int, default=4)
    p.add_argument('--no_attn',         action='store_true')
    p.add_argument('--svg',             action='store_true')
    p.add_argument('--out',             default='./generaciones')
    p.add_argument('--dpi',             type=int, default=200)
    p.add_argument('--device',          default='auto')
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    device = (
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto'
        else torch.device(args.device)
    )

    model, stats = load_checkpoint(args.checkpoint, device)
    args._stats_epoch = stats['epoch']

    if args.textos is not None and args.mode == 'single':
        args.mode = 'compare'

    if args.mode == 'single':
        mode_single(args, model, stats, device)
    elif args.mode == 'grid':
        mode_grid(args, model, stats, device)
    elif args.mode == 'compare':
        if args.textos is None:
            args.textos = [args.texto]
        mode_compare(args, model, stats, device)


if __name__ == '__main__':
    main()