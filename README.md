# Generación de Escritura a Mano en Español — LSTM + MDN + Soft Attention

Modelo generativo **online** (vectorial) de escritura manuscrita en español, entrenado sobre el dataset **UJI Pen Characters v2**. El sistema sintetiza palabras completas a partir de caracteres aislados y las genera como secuencias de deltas (dx, dy, pen\_lift) usando una LSTM con ventana de atención suave y una Mixture Density Network como cabeza de salida.

---

## Estado Actual del Proyecto

| Fase | Estado |
|---|---|
| Pipeline de datos (`UJIPen.py`) | ✅ Funcional y validado |
| Arquitectura del modelo (`model.py`) | ✅ Implementada |
| Sanity check (overfitting 2 muestras) | ✅ Superado en iteración 17 |
| Entrenamiento principal (`train.py`) | 🔄 En progreso — estancado en época ~280 |
| Generación / Inferencia (`generate.py`) | ⚠️ Operativa con limitaciones documentadas |

**Mejor checkpoint guardado:** época 191, `loss = -2.5040`

---

## Dataset

**UJI Pen Characters v2** — 11.640 muestras de caracteres manuscritos aislados, 97 clases únicas (letras ASCII + español, dígitos, símbolos). Recopilado en dos sitios (UJI y UPV), con diferente resolución de captura.

- **Corrección de escala UPV:** coordenadas divididas por 1.52 (152 ink units/mm vs 100 en UJI).
- **Solo coordenadas X/Y** — sin presión ni tiempo.
- **40 escritores** para entrenamiento, **20** para test.

---

## Ingeniería de Datos (`UJIPen.py`)

### Síntesis de palabras
El dataset contiene únicamente letras aisladas. La clase `UJIDataset` las concatena horizontalmente para formar palabras sintéticas:

- Calcula la bounding box de cada letra.
- Aplica una **heurística de línea base**: las letras descendentes (`g, j, p, q, y`) empujan su límite inferior un 30% hacia abajo, reproduciendo el comportamiento real de la escritura.
- Espaciado horizontal fijo de 20 unidades entre caracteres.

### Normalización
- **Solo se divide por la desviación estándar** — la media no se resta para preservar la direccionalidad de los trazos.
- Estadísticas globales calculadas sobre 1.000 muestras sintéticas: `std_dx ≈ 49`, `std_dy ≈ 70`.
- Clipping de seguridad en `[-20, 20]` (inactivo en condiciones normales; los valores oscilan entre -0.8 y 0.8).

### Vectorización — formato Strokes-3
Cada secuencia de trazos se convierte a deltas relativos:

```
[SOS]         → [0.0,  0.0,  1.0]   # Token de inicio
[trazo]       → [dx/σ, dy/σ, 0.0]   # Pluma en papel
[levantamiento] → [dx/σ, dy/σ, 1.0]  # Pen lift
```

### Generador híbrido
El `DataLoader` alterna entre dos fuentes en cada epoch:

- **70%** — palabras reales del diccionario (`words.txt`, 819.392 palabras con tildes y ñ).
- **30%** — cadenas aleatorias de caracteres disponibles (robustez ante transiciones inusuales).

---

## Arquitectura del Modelo (`model.py`)

```
Entrada: secuencia de deltas (T × 3)
  └─► LSTM-1 (512 unidades)
       └─► Soft Attention Window  ──► contexto de letra actual (vector de embed_dim=64)
  └─► LSTM-2 (512 unidades)  ←── skip connection: x_t + window + h1
       └─► LayerNorm
  └─► MDN Head  →  M=20 Gaussianas bivariadas + pen_lift
```

### A. Codificador de texto — Soft Attention Window
Implementa la ventana de atención suave de Graves (2013). En cada paso `t`, calcula un vector de "gravedad" `φ(t)` sobre los caracteres del texto objetivo usando `K=10` componentes gaussianas. La atención se desplaza progresivamente de izquierda a derecha conforme la red dibuja.

Parámetros de inicialización críticos: `bias[-K:] = -4.0` para forzar avance lento de `kappa` al inicio del entrenamiento.

### B. LSTM de 2 capas apiladas
- **LSTM-1:** recibe `(x_t, window)`. Alimenta la atención.
- **LSTM-2:** recibe `(x_t, window, h1)` — skip connection directa desde la entrada.
- **LayerNorm** después de cada capa (estabiliza el entrenamiento largo).

### C. Mixture Density Network (MDN)
La cabeza MDN predice los parámetros de una mezcla de **20 Gaussianas bivariadas** por cada paso de tiempo:

- `π` — pesos de mezcla (softmax).
- `(μx, μy)` — centros de cada Gaussiana.
- `(σx, σy)` — desviaciones estándar; `clamp(min=0.10)` para evitar colapso.
- `ρ` — correlación entre ejes; `tanh` para mantener en `(-1, 1)`.
- `e` — probabilidad de pen lift (BCE con `pos_weight=12.0` para compensar desbalance ~8%).

---

## Función de Pérdida

```
L = NLL_mezcla + pen_BCE + MSE_medias_ponderadas + reg_sigma + (-0.05 · H(π))
```

- **NLL:** log-verosimilitud negativa de la mezcla gaussiana bivariada.
- **pen\_BCE:** `pos_weight=12.0` para compensar la baja frecuencia de pen lifts.
- **MSE medias:** penaliza que la media ponderada `Σ(π·μ)` se aleje del target real. `mu_weight=0.5` en producción.
- **reg\_sigma:** empuja `log(σ)` hacia un valor objetivo (`-1.4 ≈ σ=0.24`). `sigma_reg=0.35`.
- **Entropía de π:** penalización negativa `(-0.05 · H(π))` para evitar el colapso del MDN a un único componente.

---

## Entrenamiento (`train.py`)

### Hiperparámetros activos

| Parámetro | Valor |
|---|---|
| `BATCH_SIZE` | 64 |
| `EPOCHS` | 500 |
| `LR` inicial | 1e-4 |
| `EPOCH_SIZE` | 3.000 batches |
| `CLIP` (grad norm) | 5.0 |
| `SS_WARMUP` | 200 épocas |
| `SS_MIN` | 0.20 |
| `MU_WEIGHT` | 0.5 |
| `SIGMA_REG` | 0.35 |

### Scheduled Sampling
- Épocas 1–200: teacher forcing puro (`tf=1.0`).
- Épocas 200+: decay lineal hasta `tf=0.20`.
- El muestreo durante SS se ejecuta **completamente en GPU** (`sample_from_mdn_batch`) — eliminó 1.49 millones de llamadas NumPy por época y redujo el tiempo de `~36 min → ~3-4 min/época`.

### Hardware
- NVIDIA RTX 3060 6GB VRAM, vía WSL + CUDA.
- Uso de memoria GPU: ~2.9/6.0 GB con `BATCH_SIZE=64`.

---

## Sistema de Monitoreo

El log de cada época reporta:

```
Época XXXX  loss=X.XXXX  nll=X.XXXX  lr=X.XXe-XX  tf=X.XX  |
σx=X.XXX  σy=X.XXX  σmin=X.XXX  |  wμx=X.XXX  wμy=X.XXX  pen=X.XXX  |
grad=XXX.XX  H(π)=X.XX
```

### Tabla de referencia de salud del modelo

| Métrica | Sano | Alerta | Crítico |
|---|---|---|---|
| `σmin` | > 0.12 | 0.08–0.12 | < 0.08 |
| `grad` | 0.5–3.0 | 3.0–4.5 | > 4.8 (clip siempre activo) |
| `H(π)` | 1.5–2.5 | < 1.0 o > 2.8 | < 0.5 (colapso MDN) |
| `pen` | ~0.08 | < 0.05 antes de época 200 | — |

---

## Problemas Documentados y Fixes Aplicados

### Colapso del MDN (época 190)
**Causa:** `σmin` tocó el floor (`0.05`) de forma silenciosa durante ~150 épocas. Una Gaussiana muy estrecha domina la NLL, el optimizador concentra todo `π` en ese componente y `H(π)` colapsó a 0.21.

**Fix:** `sigma_floor` elevado a `0.10`, `sigma_target_log` a `-1.4`, y regularización de entropía añadida a la loss. Adam reiniciado sin cargar `optimizer.state_dict()`.

### Scheduler demasiado agresivo durante SS
**Causa:** `ReduceLROnPlateau` no distingue entre "modelo divergiendo" y "ruido inherente del scheduled sampling". Redujo el LR cuatro veces en 30 épocas (1e-4 → 6.25e-6), congelando prácticamente el modelo.

**Estado:** Pendiente de evaluar `CosineAnnealingLR` con margen suficiente de épocas.

### Condición de parada en `generate()` inalcanzable
**Causa:** El umbral `phi[-1] > 2.0` era correcto en el sanity check (sobreajuste extremo), pero en producción `phi_max ≈ 0.20`.

**Fix actual:** Condición agnóstica a magnitud absoluta:
```python
phi_norm   = phi_vals / (phi_vals.sum() + 1e-8)
last_ratio = phi_norm[-1].item()
cond_phi   = last_ratio > phi_norm[:-1].max().item() and last_ratio > 0.6
cond_len   = step > len(texto.replace(' ', '')) * 80
```

---

## Próximos Pasos

1. **Reinicio desde época 0** con todas las correcciones integradas (~30 h estimadas a 3-4 min/época).
2. Reducir MDN de `M=20` a `M=10` componentes — `H(π)` tardó 150 épocas en recuperarse del colapso; menos componentes reducen la presión sobre el optimizador.
3. `pen_weight=12.0` desde el inicio para evitar la caída monotónica documentada en épocas 1–180.
4. Añadir flag `--optimizer [y/n]` en `train.py` para decidir en terminal si cargar el estado del optimizador al reanudar.
5. Evaluar `CosineAnnealingLR` como reemplazo de `ReduceLROnPlateau` para tolerar el ruido de SS.

---

## Estructura de Archivos

```
.
├── ujipenchars2.txt        # Dataset UJI Pen Characters v2
├── words.txt               # Diccionario (819.392 palabras en español)
├── UJIPen.py               # Dataset class: carga, síntesis, normalización, vectorización
├── model.py                # Arquitectura: LSTM, SoftAttention, MDN, loss function
├── train.py                # Loop de entrenamiento principal con scheduled sampling
├── sanity_train.py         # Overfitting controlado sobre 2 muestras para validar arquitectura
├── handwriting_model.pt    # Checkpoint del mejor modelo (época 191, loss=-2.5040)
└── README.md               # Este archivo
```

---

## Referencia del Dataset

> F. Prat, M. J. Castro, D. Llorens, A. Marzal, J. M. Vilar.
> *UJIpenchars2: A Pen-Based Database with More Than 11K Isolated Handwritten Characters.*
> Universitat Jaume I / Universidad Politécnica de Valencia, 2008.
