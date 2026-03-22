"""
AUDITORÍA DE UJIPen.py
======================
Script de diagnóstico independiente. No modifica nada.
Mide 5 propiedades estructurales del pipeline de datos que pueden
explicar el fallo crónico en Check C y Check G.

Ejecutar desde la raíz del proyecto:
    python3 audit_ujipen.py

Salida: consola + audit_report.txt en ./debug_logs/
"""

import sys
import os
import random
import numpy as np

sys.path.insert(0, '.')
from src.UJIPen import UJIDataset

# ─── Configuración ────────────────────────────────────────────────────────────
DATASET_PATH = './data/ujipenchars2.txt'
DICT_PATH    = './data/words.txt'
N_WORDS      = 2000      # palabras a muestrear para estadísticas
SEED         = 42
OUTPUT_DIR   = './debug_logs'
REPORT_FILE  = os.path.join(OUTPUT_DIR, 'audit_ujipen.txt')

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Carga ────────────────────────────────────────────────────────────────────
print("Cargando dataset...")
dataset = UJIDataset(DATASET_PATH, epoch_size=100)
dataset.load_dictionary_from_txt(DICT_PATH)
print(f"Dataset listo. std_dx={dataset.std_dx:.2f}  std_dy={dataset.std_dy:.2f}\n")


# ─── Utilidades ───────────────────────────────────────────────────────────────
def flat_deltas_raw(sample):
    """
    Replica to_deltas sin normalizar.
    Retorna lista de (dx, dy, pen_lift, was_new_stroke, was_skipped_dup)
    incluyendo los eventos omitidos por duplicados.
    """
    if not sample['strokes'] or not sample['strokes'][0]:
        return []

    flat_points = []
    for s_idx, stroke in enumerate(sample['strokes']):
        for i, pt in enumerate(stroke):
            is_new_stroke = (i == 0 and s_idx > 0)
            flat_points.append((pt, is_new_stroke))

    last_x, last_y = flat_points[0][0]
    records = []

    for (x, y), is_new_stroke in flat_points[1:]:
        dx, dy = x - last_x, y - last_y
        is_dup = abs(dx) < 1e-4 and abs(dy) < 1e-4
        # registrar SIEMPRE, marcando si fue descartado
        records.append({
            'dx':            dx,
            'dy':            dy,
            'pen_lift':      1.0 if is_new_stroke else 0.0,
            'was_new_stroke': is_new_stroke,
            'was_skipped':   is_dup,
        })
        if not is_dup:
            last_x, last_y = x, y
        else:
            last_x, last_y = x, y   # actualizar de todos modos para no acumular drift

    return records


def stroke_count_dist(data_by_char):
    """Distribución del número de trazos por carácter en el dataset real."""
    counts = {}
    for char, samples in data_by_char.items():
        n = np.mean([len(s['strokes']) for s in samples])
        counts[char] = round(n, 2)
    return counts


# ─── AUDITORÍA 1: Pérdida de pen_lift por filtro de duplicados ────────────────
print("─" * 60)
print("AUDITORÍA 1 — Pen_lift perdidas por filtro de duplicados")
print("─" * 60)

total_pen_events    = 0
lost_pen_events     = 0
total_normal_events = 0
skipped_normal      = 0

for _ in range(N_WORDS):
    sample = dataset.get_training_sample()
    records = flat_deltas_raw(sample)
    for r in records:
        if r['was_new_stroke']:
            total_pen_events += 1
            if r['was_skipped']:
                lost_pen_events += 1
        else:
            total_normal_events += 1
            if r['was_skipped']:
                skipped_normal += 1

pct_lost = lost_pen_events / max(total_pen_events, 1) * 100
a1_lines = [
    f"Palabras analizadas:           {N_WORDS}",
    f"Eventos pen_lift totales:      {total_pen_events}",
    f"Pen_lifts PERDIDOS (dup=0,0):  {lost_pen_events}  ({pct_lost:.2f}%)",
    f"  → % pen_lift perdida:        {pct_lost:.2f}%",
    f"Movimientos normales saltados: {skipped_normal}  ({skipped_normal/max(total_normal_events,1)*100:.2f}%)",
    f"",
    f"Diagnóstico: {'CRÍTICO — >5% pen_lifts se pierden silenciosamente' if pct_lost > 5 else 'BAJO — <5% pen_lifts perdidas' if pct_lost > 0 else 'LIMPIO — no hay pen_lifts perdidas'}",
]
for l in a1_lines: print(l)


# ─── AUDITORÍA 2: Intra-char vs Inter-char pen_lifts ─────────────────────────
print()
print("─" * 60)
print("AUDITORÍA 2 — Composición de pen_lifts: intra-char vs inter-char")
print("─" * 60)

intra_char_lifts = 0
inter_char_lifts = 0
total_tokens     = 0
chars_per_word   = []

for _ in range(N_WORDS):
    texto = dataset.get_training_sample()['label']
    chars_validos = [c for c in texto if c in dataset.data_by_char]
    if not chars_validos:
        continue
    chars_per_word.append(len(chars_validos))

    # Para cada carácter, contar sus trazos propios
    for char in chars_validos:
        variantes = dataset.data_by_char.get(char, [])
        if not variantes:
            continue
        muestra = random.choice(variantes)
        n_strokes = len(muestra['strokes'])
        intra_char_lifts += max(n_strokes - 1, 0)   # trazos intra-char
        inter_char_lifts += 1                         # siempre hay 1 inter-char por letra
        total_tokens += 1

total_lifts = intra_char_lifts + inter_char_lifts
a2_lines = [
    f"Tokens procesados:             {total_tokens}",
    f"Pen_lifts INTRA-character:     {intra_char_lifts}  ({intra_char_lifts/max(total_lifts,1)*100:.1f}%)",
    f"  (entre trazos de la misma letra, señal geométrica consistente)",
    f"Pen_lifts INTER-character:     {inter_char_lifts}  ({inter_char_lifts/max(total_lifts,1)*100:.1f}%)",
    f"  (entre letras distintas, salto de cursor arbitrario)",
    f"Ratio inter/intra:             {inter_char_lifts/max(intra_char_lifts,1):.2f}x",
    f"Media chars por palabra:       {np.mean(chars_per_word):.1f}",
    f"",
    f"Diagnóstico: el modelo recibe señales de pen_lift de dos naturalezas radicalmente distintas.",
    f"El {'%.0f'%(inter_char_lifts/max(total_lifts,1)*100)}% inter-char tiene saltos (dx,dy) determinados por el cursor, no por la letra.",
]
for l in a2_lines: print(l)


# ─── AUDITORÍA 3: Inflación del std por saltos inter-trazo ────────────────────
print()
print("─" * 60)
print("AUDITORÍA 3 — Inflación del std_dx/std_dy por pen_lift jumps")
print("─" * 60)

all_dx_normal, all_dy_normal = [], []
all_dx_penlift, all_dy_penlift = [], []

for _ in range(N_WORDS):
    sample = dataset.get_training_sample()
    records = flat_deltas_raw(sample)
    for r in records:
        if r['was_skipped']:
            continue
        if r['was_new_stroke']:
            all_dx_penlift.append(r['dx'])
            all_dy_penlift.append(r['dy'])
        else:
            all_dx_normal.append(r['dx'])
            all_dy_normal.append(r['dy'])

std_dx_all    = dataset.std_dx
std_dy_all    = dataset.std_dy
std_dx_normal = float(np.std(all_dx_normal)) if all_dx_normal else 1.0
std_dy_normal = float(np.std(all_dy_normal)) if all_dy_normal else 1.0
std_dx_pen    = float(np.std(all_dx_penlift)) if all_dx_penlift else 1.0
std_dy_pen    = float(np.std(all_dy_penlift)) if all_dy_penlift else 1.0

mean_abs_dx_normal_norm = float(np.mean(np.abs(all_dx_normal))) / std_dx_all
mean_abs_dy_normal_norm = float(np.mean(np.abs(all_dy_normal))) / std_dy_all
mean_abs_dx_pen_norm    = float(np.mean(np.abs(all_dx_penlift))) / std_dx_all
mean_abs_dy_pen_norm    = float(np.mean(np.abs(all_dy_penlift))) / std_dy_all

a3_lines = [
    f"std_dx usado para normalizar (TODOS los deltas):  {std_dx_all:.2f}",
    f"std_dy usado para normalizar (TODOS los deltas):  {std_dy_all:.2f}",
    f"",
    f"std_dx de movimientos NORMALES solamente:         {std_dx_normal:.2f}",
    f"std_dy de movimientos NORMALES solamente:         {std_dy_normal:.2f}",
    f"Inflación dx por incluir pen_lifts:               {std_dx_all/std_dx_normal:.2f}x",
    f"Inflación dy por incluir pen_lifts:               {std_dy_all/std_dy_normal:.2f}x",
    f"",
    f"Media |delta_x| normalizado — movimientos NORMALES: {mean_abs_dx_normal_norm:.4f}",
    f"Media |delta_y| normalizado — movimientos NORMALES: {mean_abs_dy_normal_norm:.4f}",
    f"  (ideal: ~1.0 si std fuera calculado solo sobre normales)",
    f"",
    f"Media |delta_x| normalizado — PEN_LIFT jumps:       {mean_abs_dx_pen_norm:.4f}",
    f"Media |delta_y| normalizado — PEN_LIFT jumps:       {mean_abs_dy_pen_norm:.4f}",
    f"  (los pen_lift jumps son {mean_abs_dx_pen_norm/max(mean_abs_dx_normal_norm,1e-6):.1f}x más grandes que los normales en x)",
    f"",
    f"Diagnóstico: los movimientos normales quedan comprimidos a {mean_abs_dx_normal_norm:.2f} unidades",
    f"normalizadas (el modelo los ve todos 'pequeños'). El std inflado reduce",
    f"la señal de movimiento normal que llega al LSTM.",
]
for l in a3_lines: print(l)


# ─── AUDITORÍA 4: Señal predictiva antes de cada pen_lift ─────────────────────
print()
print("─" * 60)
print("AUDITORÍA 4 — ¿Qué ve el modelo en el paso justo ANTES de un pen_lift?")
print("─" * 60)

pre_penlift_dx  = []    # |delta_x| en t (el paso antes del pen_lift en t+1)
pre_penlift_dy  = []    # |delta_y| en t
normal_dx       = []    # |delta_x| en pasos normales
normal_dy       = []    # |delta_y| en pasos normales
pre_type_intra  = []    # ¿el pen_lift en t+1 es intra o inter char?
pre_type_inter  = []

for _ in range(N_WORDS):
    sample   = dataset.get_training_sample()
    records  = flat_deltas_raw(sample)
    # Filtrar los skipped para simular la secuencia real de to_deltas
    clean    = [r for r in records if not r['was_skipped']]

    if len(clean) < 2:
        continue

    # Contar pen_lifts intra-char calculando stroques del sample
    # (el primer pen_lift de cada carácter es inter-char si es la primera stroke,
    # o intra-char si hay múltiples strokes del mismo carácter)
    # Simplificación: el contexto de si es intra o inter no está fácilmente disponible
    # en los records, así que medimos solo la señal del paso previo

    for i in range(1, len(clean)):
        r_next = clean[i]
        r_prev = clean[i - 1]
        if r_next['pen_lift'] == 1.0:
            pre_penlift_dx.append(abs(r_prev['dx']))
            pre_penlift_dy.append(abs(r_prev['dy']))
        else:
            normal_dx.append(abs(r_prev['dx']))
            normal_dy.append(abs(r_prev['dy']))

mean_pre_dx  = float(np.mean(pre_penlift_dx)) if pre_penlift_dx else 0
mean_pre_dy  = float(np.mean(pre_penlift_dy)) if pre_penlift_dy else 0
mean_norm_dx = float(np.mean(normal_dx))       if normal_dx      else 0
mean_norm_dy = float(np.mean(normal_dy))       if normal_dy      else 0

# Kolmogorov-Smirnov: ¿son distinguibles las distribuciones?
from scipy import stats as scipy_stats
ks_dx = scipy_stats.ks_2samp(
    np.random.choice(pre_penlift_dx, min(5000, len(pre_penlift_dx)), replace=False),
    np.random.choice(normal_dx,      min(5000, len(normal_dx)),      replace=False)
)
ks_dy = scipy_stats.ks_2samp(
    np.random.choice(pre_penlift_dy, min(5000, len(pre_penlift_dy)), replace=False),
    np.random.choice(normal_dy,      min(5000, len(normal_dy)),      replace=False)
)

a4_lines = [
    f"Pasos PREVIOS a un pen_lift analizados: {len(pre_penlift_dx)}",
    f"Pasos normales analizados:              {len(normal_dx)}",
    f"",
    f"Media |dx| raw en paso PREVIO a pen_lift:  {mean_pre_dx:.2f}  (normalizado: {mean_pre_dx/std_dx_all:.4f})",
    f"Media |dx| raw en paso NORMAL:             {mean_norm_dx:.2f}  (normalizado: {mean_norm_dx/std_dx_all:.4f})",
    f"Media |dy| raw en paso PREVIO a pen_lift:  {mean_pre_dy:.2f}  (normalizado: {mean_pre_dy/std_dy_all:.4f})",
    f"Media |dy| raw en paso NORMAL:             {mean_norm_dy:.2f}  (normalizado: {mean_norm_dy/std_dy_all:.4f})",
    f"",
    f"Test KS (distinguibilidad dx): D={ks_dx.statistic:.4f}  p={ks_dx.pvalue:.4f}",
    f"Test KS (distinguibilidad dy): D={ks_dy.statistic:.4f}  p={ks_dy.pvalue:.4f}",
    f"  (p < 0.05 → distribuciones estadísticamente distinguibles)",
    f"  (D cercano a 0 → distribuciones muy similares → señal débil para LSTM)",
    f"",
    f"Diagnóstico: el movimiento en t-1 antes de un pen_lift",
    f"{'ES distinguible' if ks_dx.pvalue < 0.05 else 'NO ES estadísticamente distinguible'} del movimiento normal en x",
    f"{'ES distinguible' if ks_dy.pvalue < 0.05 else 'NO ES estadísticamente distinguible'} del movimiento normal en y",
]
for l in a4_lines: print(l)


# ─── AUDITORÍA 5: Variabilidad del pen_lift jump (¿qué tan consistente es?) ───
print()
print("─" * 60)
print("AUDITORÍA 5 — Variabilidad del jump en pen_lift: ¿predecible o caótico?")
print("─" * 60)

pen_jumps_dx = []
pen_jumps_dy = []
pen_jumps_dist = []

for _ in range(N_WORDS):
    sample  = dataset.get_training_sample()
    records = flat_deltas_raw(sample)
    for r in records:
        if r['was_skipped'] or r['pen_lift'] != 1.0:
            continue
        pen_jumps_dx.append(r['dx'])
        pen_jumps_dy.append(r['dy'])
        pen_jumps_dist.append(np.sqrt(r['dx']**2 + r['dy']**2))

if pen_jumps_dist:
    dist_arr = np.array(pen_jumps_dist)
    dx_arr   = np.array(pen_jumps_dx)
    dy_arr   = np.array(pen_jumps_dy)
    pct_neg_dy = float(np.mean(dy_arr < 0)) * 100   # pen_lift que sube (y negativa)

    a5_lines = [
        f"Pen_lift jumps analizados: {len(pen_jumps_dist)}",
        f"",
        f"Distancia Euclidea del jump:",
        f"  media: {dist_arr.mean():.2f}   std: {dist_arr.std():.2f}   CV: {dist_arr.std()/dist_arr.mean():.2f}",
        f"  p5:  {np.percentile(dist_arr, 5):.2f}   p25: {np.percentile(dist_arr, 25):.2f}",
        f"  p50: {np.percentile(dist_arr, 50):.2f}   p75: {np.percentile(dist_arr, 75):.2f}   p95: {np.percentile(dist_arr, 95):.2f}",
        f"",
        f"Delta_x del jump:",
        f"  media: {dx_arr.mean():.2f}  std: {dx_arr.std():.2f}  min: {dx_arr.min():.2f}  max: {dx_arr.max():.2f}",
        f"  % jumps con dx > 0 (hacia la derecha, esperado): {float(np.mean(dx_arr > 0))*100:.1f}%",
        f"  % jumps con dx < 0 (hacia la izquierda, anómalo): {float(np.mean(dx_arr < 0))*100:.1f}%",
        f"",
        f"Delta_y del jump:",
        f"  media: {dy_arr.mean():.2f}  std: {dy_arr.std():.2f}",
        f"  % jumps hacia arriba (dy < 0): {pct_neg_dy:.1f}%  — esperado para inicio de nueva letra",
        f"",
        f"CV (coef. de variación) de la distancia: {dist_arr.std()/dist_arr.mean():.2f}",
        f"  (CV > 1.0 → distribución extremadamente dispersa → MDN necesita gaussianas muy anchas)",
        f"  (CV > 1.5 → distribución caótica → prácticamente imposible de predecir con precisión)",
        f"",
        f"Distancia normalizada (dividida por std_all): {dist_arr.mean()/((std_dx_all+std_dy_all)/2):.3f} unidades",
        f"  (comparar con movimiento normal normalizado: ~{mean_abs_dx_normal_norm:.3f} en x, ~{mean_abs_dy_normal_norm:.3f} en y)",
    ]
    for l in a5_lines: print(l)


# ─── AUDITORÍA 6: Número de trazos por carácter (señal de pen_lift disponible) 
print()
print("─" * 60)
print("AUDITORÍA 6 — Distribución de trazos por carácter")
print("─" * 60)

stroke_counts = {}
for char, samples in dataset.data_by_char.items():
    sc = [len(s['strokes']) for s in samples]
    stroke_counts[char] = (np.mean(sc), np.std(sc), sc)

n1 = sum(1 for c, (m, s, _) in stroke_counts.items() if m < 1.5)
n2 = sum(1 for c, (m, s, _) in stroke_counts.items() if 1.5 <= m < 2.5)
n3 = sum(1 for c, (m, s, _) in stroke_counts.items() if m >= 2.5)

chars_inconsistent = [(c, m, s) for c, (m, s, _) in stroke_counts.items() if s > 0.3]

a6_lines = [
    f"Total clases de caracteres: {len(stroke_counts)}",
    f"",
    f"Caracteres con ~1 trazo:    {n1}  ({n1/len(stroke_counts)*100:.1f}%)",
    f"Caracteres con ~2 trazos:   {n2}  ({n2/len(stroke_counts)*100:.1f}%)",
    f"Caracteres con ~3+ trazos:  {n3}  ({n3/len(stroke_counts)*100:.1f}%)",
    f"",
    f"Caracteres con número de trazos INCONSISTENTE entre escritores (std > 0.3):",
    f"  {len(chars_inconsistent)} caracteres con variabilidad alta",
]
if chars_inconsistent:
    a6_lines.append("  Los 10 más variables:")
    for c, m, s in sorted(chars_inconsistent, key=lambda x: -x[2])[:10]:
        a6_lines.append(f"    '{c}': media={m:.2f} trazos, std={s:.2f}")
a6_lines += [
    f"",
    f"Diagnóstico: en los {len(chars_inconsistent)} caracteres inconsistentes, el mismo carácter",
    f"puede tener 1 o 2 trazos dependiendo del escritor. El modelo ve",
    f"pen_lifts en posiciones impredecibles para estos caracteres.",
]
for l in a6_lines: print(l)


# ─── REPORTE FINAL ────────────────────────────────────────────────────────────
separator = "═" * 62
sep2      = "─" * 62
report_lines = [
    separator,
    "REPORTE DE AUDITORÍA — UJIPen.py",
    f"N={N_WORDS} palabras muestreadas  |  seed={SEED}",
    sep2,
    "",
    "HALLAZGO 1 — Pen_lifts perdidas por filtro de duplicados",
    *["  " + l for l in a1_lines],
    "",
    "HALLAZGO 2 — Composición intra/inter-character",
    *["  " + l for l in a2_lines],
    "",
    "HALLAZGO 3 — Inflación de std",
    *["  " + l for l in a3_lines],
    "",
    "HALLAZGO 4 — Señal predictiva antes del pen_lift",
    *["  " + l for l in a4_lines],
    "",
    "HALLAZGO 5 — Variabilidad del jump",
    *(["  " + l for l in a5_lines] if pen_jumps_dist else ["  (sin datos)"]),
    "",
    "HALLAZGO 6 — Inconsistencia de trazos por escritor",
    *["  " + l for l in a6_lines],
    "",
    separator,
    "RESUMEN EJECUTIVO",
    sep2,
]

# Determinar diagnóstico automático
problems = []
if pct_lost > 2:
    problems.append(f"P1 [CRITICO] {pct_lost:.1f}% de pen_lifts se pierden silenciosamente por filtro de duplicados")
if inter_char_lifts / max(total_lifts, 1) > 0.6:
    problems.append(f"P2 [ALTO]    {inter_char_lifts/max(total_lifts,1)*100:.0f}% de pen_lifts son inter-char con saltos arbitrarios")
if std_dx_all / std_dx_normal > 1.3:
    problems.append(f"P3 [MEDIO]   std inflado {std_dx_all/std_dx_normal:.2f}x por pen_lift jumps → movimientos normales comprimidos a {mean_abs_dx_normal_norm:.2f} unidades")
if ks_dx.pvalue > 0.05 or ks_dy.pvalue > 0.05:
    problems.append(f"P4 [ALTO]    El paso PREVIO a pen_lift es indistinguible del paso normal (KS p={min(ks_dx.pvalue, ks_dy.pvalue):.4f})")
if pen_jumps_dist and dist_arr.std() / dist_arr.mean() > 1.0:
    problems.append(f"P5 [ALTO]    CV del jump = {dist_arr.std()/dist_arr.mean():.2f} (>1.0) — los saltos son altamente variables")
if len(chars_inconsistent) > 10:
    problems.append(f"P6 [MEDIO]   {len(chars_inconsistent)} caracteres con número de trazos inconsistente entre escritores")

if problems:
    for p in problems:
        report_lines.append(f"  {p}")
else:
    report_lines.append("  No se detectaron problemas estructurales significativos.")

report_lines += [
    "",
    "IMPLICACIÓN PARA Check C y G:",
    "  Si P4 está confirmado (señal previa indistinguible), el LSTM no puede",
    "  aprender a predecir pen_lifts desde el movimiento inmediatamente anterior.",
    "  La única señal disponible es la atención (kappa entre caracteres).",
    "  Esto es un límite de información del dataset, no un problema de la loss.",
    "",
    "  Si P1 está confirmado (pen_lifts perdidas), la tasa real de pen_lift",
    "  en el tensor de entrenamiento es MENOR que TRUE_PEN_RATE=0.029.",
    "  Esto sesga el equilibrio de gradientes BCE hacia abajo.",
    "",
    "  Si P5 está confirmado (CV > 1.0), el MDN no puede aprender una",
    "  gaussiana estrecha para los jumps — necesita sigmas grandes,",
    "  lo que reduce la presión hacia e_raw_pen alto.",
    separator,
]

for l in report_lines:
    print(l)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')

print(f"\nReporte guardado en: {REPORT_FILE}")