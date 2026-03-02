import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.model import HandwritingGenerator, sample_from_mdn, parse_mdn_params


def cargar_modelo(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt       = torch.load(checkpoint_path, map_location=device)
    char_vocab = ckpt['char_vocab']
    std_dx     = ckpt['std_dx']
    std_dy     = ckpt['std_dy']

    mean_dy = ckpt.get('mean_dy', 0.0)

    model = HandwritingGenerator(char_vocab).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss',  '?')
    print(f"Modelo cargado  |  época={epoch}  |  loss={loss:.4f}  |  vocab={len(char_vocab)} chars")
    return model, char_vocab, std_dx, std_dy, mean_dy


@torch.no_grad()
def generar(
    model:     HandwritingGenerator,
    texto:     str,
    device:    torch.device,
    mean_dy:   float = 0.0,
    std_dy:    float = 1.0,
    bias:      float = 1.0,
    max_steps: int   = 2000,
) -> tuple:
    char_vocab = model.char_vocab
    chars_desconocidos = [ch for ch in texto if ch not in char_vocab and ch != ' ']
    if chars_desconocidos:
        print(f"[WARN] Caracteres no vistos durante entrenamiento: {chars_desconocidos}")

    char_embeds = model.char_embed(model.encode_text([texto], device))
    model.attention.reset(1, device)
    h1     = model._zero_hidden(1, device)
    h2     = model._zero_hidden(1, device)
    window = torch.zeros(1, model.embed_dim, device=device)
    x_t    = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    dy_bias = mean_dy / (std_dy + 1e-6)

    strokes    = [x_t.squeeze(0).cpu().numpy()]
    attentions = []
    expected_steps = len(texto.replace(' ', '')) * 80

    for step in range(max_steps):
        inp1        = torch.cat([x_t, window], dim=1).unsqueeze(1)
        o1, h1      = model.lstm1(inp1, h1)
        o1          = model.norm1(o1.squeeze(1))
        window, phi = model.attention(o1, char_embeds)
        attentions.append(phi.squeeze(0).cpu().numpy())

        phi_vals = phi.squeeze(0)
        phi_norm   = phi_vals / (phi_vals.sum() + 1e-8)
        last_ratio = phi_norm[-1].item()
        cond_phi = last_ratio > phi_norm[:-1].max().item() and last_ratio > 0.6
        cond_len = step > expected_steps
        if cond_phi or cond_len:
            print(f"Paro en paso {step + 1}  ({'phi' if cond_phi else 'longitud'})")
            break

        inp2   = torch.cat([x_t, window, o1], dim=1).unsqueeze(1)
        o2, h2 = model.lstm2(inp2, h2)
        o2     = model.norm2(o2.squeeze(1))

        params = model.mdn_head(torch.cat([o1, o2], dim=1)).squeeze(0)
        x_t    = sample_from_mdn(params, M=model.M, bias=bias).unsqueeze(0)
        x_t[0, 1] -= dy_bias
        strokes.append(x_t.squeeze(0).cpu().numpy())

    return strokes, np.array(attentions)


def reconstruir_absolutos(strokes: list, std_dx: float, std_dy: float) -> list:
    trazos_abs = []
    trazo_actual = []
    abs_x, abs_y = 0.0, 0.0

    for i in range(1, len(strokes)):
        dx, dy, pen_lift = strokes[i]
        abs_x += dx * std_dx
        abs_y += dy * std_dy
        trazo_actual.append((abs_x, abs_y))

        if pen_lift > 0.5:
            if trazo_actual:
                trazos_abs.append(trazo_actual)
            trazo_actual = []

    if trazo_actual:
        trazos_abs.append(trazo_actual)

    return trazos_abs


def guardar_imagen(
    texto:      str,
    trazos_abs: list,
    attentions: np.ndarray,
    bias:       float,
    filename:   str,
):
    fig = plt.figure(figsize=(max(8, len(texto) * 1.5), 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1])

    ax1 = fig.add_subplot(gs[0])
    for trazo in trazos_abs:
        xs, ys = zip(*trazo)
        ax1.plot(xs, ys, 'k-', linewidth=2)

    n_pasos = sum(len(t) for t in trazos_abs)
    ax1.set_title(f"'{texto}'  |  bias={bias}  |  pasos={n_pasos}")
    ax1.invert_yaxis()
    ax1.axis('equal')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    chars_display = [ch for ch in texto if ch != ' ']
    im  = ax2.imshow(attentions.T, aspect='auto', cmap='viridis', origin='upper')
    ax2.set_title("Atención por carácter")
    ax2.set_xlabel("Paso de tiempo generado")
    ax2.set_ylabel("Caracteres")
    ax2.set_yticks(np.arange(len(texto)))
    ax2.set_yticklabels(list(texto))
    plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.2)

    plt.tight_layout()
    plt.savefig(f'./results/{filename}', dpi=150, bbox_inches='tight')
    print(f"Imagen guardada: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generador de escritura a mano")
    parser.add_argument('--checkpoint', type=str,   default='./modelos/handwriting_model.pt') # No ejecutar hasta tener un modelo entrenado
    parser.add_argument('--texto',      type=str,   default='hola')
    parser.add_argument('--bias',       type=float, default=1.0,
                        help='Temperatura de muestreo. Mayor = más limpio. Rango sugerido: 0.5–3.0')
    parser.add_argument('--max_steps',  type=int,   default=2000)
    parser.add_argument('--salida',     type=str,   default='generado.png')
    parser.add_argument('--device',     type=str,   default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Dispositivo: {device}")
    print(f"Texto: '{args.texto}'  |  bias={args.bias}")

    model, _, std_dx, std_dy, mean_dy = cargar_modelo(args.checkpoint, device)

    strokes, attentions = generar(
        model, args.texto, device,
        mean_dy=mean_dy, std_dy=std_dy,
        bias=args.bias, max_steps=args.max_steps,
    )

    trazos_abs = reconstruir_absolutos(strokes, std_dx, std_dy)

    if not trazos_abs:
        print("[ERROR] No se generaron trazos. Prueba reducir el bias o revisar el checkpoint.")
        sys.exit(1)

    guardar_imagen(args.texto, trazos_abs, attentions, args.bias, args.salida)
    print("Generación completada.")


if __name__ == '__main__':
    main()