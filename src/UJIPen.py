import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

class UJIDataset(Dataset):
    """
    Dataset UJI Pen Characters optimizado para generación de texto manuscrito.
    """
    def __init__(self, file_path, epoch_size=10000):
        self.file_path = file_path
        self.epoch_size = epoch_size
        self.data_by_char = {}
        self.vocabulario = []

        self.html_map = {
            'ntilde': 'ñ', 'Ntilde': 'Ñ',
            'aacute': 'á', 'Aacute': 'Á', 'eacute': 'é', 'Eacute': 'É',
            'iacute': 'í', 'Iacute': 'Í', 'oacute': 'ó', 'Oacute': 'Ó',
            'uacute': 'ú', 'Uacute': 'Ú', 'uuml': 'ü', 'Uuml': 'Ü',
            'iquest': '¿', 'iexcl': '¡'
        }

        self._load_data()
        self.mean_dx, self.std_dx, self.mean_dy, self.std_dy = self._compute_stats()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        sample = self.get_training_sample()
        deltas = self.to_deltas(sample)
        return deltas, sample['label']

    def _load_data(self):
        print(f"Cargando datos desde {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('//') or not line:
                i += 1
                continue

            if line.startswith('WORD'):
                parts = line.split()
                char_label = self.html_map.get(parts[1], parts[1])
                session_id = parts[2]
                scale_factor = 1.52 if "UPV" in session_id else 1.0

                current_sample = {'label': char_label, 'strokes': [], 'origin': session_id}

                i += 1
                if i < len(lines) and lines[i].strip().startswith('NUMSTROKES'):
                    num_strokes = int(lines[i].strip().split()[1])
                    for _ in range(num_strokes):
                        i += 1
                        coords = list(map(int, lines[i].strip().split('#')[1].strip().split()))
                        points = [(coords[j] / scale_factor, coords[j+1] / scale_factor) for j in range(0, len(coords), 2)]
                        current_sample['strokes'].append(points)

                if char_label not in self.data_by_char:
                    self.data_by_char[char_label] = []
                self.data_by_char[char_label].append(current_sample)
            i += 1

        total_chars = sum(len(v) for v in self.data_by_char.values())
        print(f"Dataset cargado: {total_chars} muestras distribuidas en {len(self.data_by_char)} clases únicas.")

    def load_dictionary_from_txt(self, path_txt):
        print(f"Cargando diccionario desde {path_txt}...")
        try:
            with open(path_txt, 'r', encoding='utf-8') as f:
                todas = [line.strip() for line in f if line.strip()]
            # iter 9: cap a 8 caracteres — reduce varianza de longitud de secuencia
            # 8 chars × ~60 pasos/char = 480 pasos máx → T_MAX_TRAIN=300 recorta el exceso
            self.vocabulario = [w for w in todas if 1 <= len(w) <= 8]
            print(f"Vocabulario cargado: {len(self.vocabulario)} palabras (≤8 chars, de {len(todas)} totales).")
        except Exception as e:
            print(f"Error cargando diccionario: {e}")
            self.vocabulario = []

    def get_training_sample(self):
        if self.vocabulario and random.random() < 0.7:
            texto = random.choice(self.vocabulario)
        else:
            longitud = random.randint(3, 9)
            texto = "".join(random.choices(list(self.data_by_char.keys()), k=longitud))
        return self.get_random_word_sample(texto)

    def to_deltas(self, sample):
        if not sample['strokes']:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        strokes_validos = [s for s in sample['strokes'] if s and len(s) >= 1]
        if not strokes_validos:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        n_strokes = len(strokes_validos)
        deltas = []

        # global_last_{x,y}: rastrean la posición física al final del último
        # trazo válido emitido. Permiten calcular el delta de navegación entre
        # trazos consecutivos, que encode el cursor advance horizontal producido
        # por get_random_word_sample(). Sin este tracking ese desplazamiento
        # era invisible al LSTM (to_deltas reiniciaba last_x/y por trazo).
        global_last_x = None
        global_last_y = None

        for s_idx, stroke in enumerate(strokes_validos):
            is_ultimo_trazo = (s_idx == n_strokes - 1)
            stroke_deltas = []

            # ── Delta de navegación inter-trazo ──────────────────────────────
            # Emitir ANTES de los deltas internos del trazo, excepto en el
            # primero. pen=0: es reposicionamiento, no cierre de trazo.
            # Semántica de la secuencia resultante:
            #   [dx_last, dy_last, 1.0]  ← cierre del trazo anterior
            #   [dx_nav,  dy_nav,  0.0]  ← salto al inicio del nuevo trazo
            #   [dx_1,    dy_1,    0.0]  ← primer movimiento interno del trazo
            # El modelo aprende que pen=1 → nav→ intra. En inferencia genera
            # el token de navegación como cualquier otro — no requiere lógica
            # especial en generate.py (eliminar el reset fijo post pen=1).
            # Nota: estos deltas no se incluyen en _compute_stats (que itera
            # solo intra-trazo), por lo que std no se infla.
            if global_last_x is not None:
                dx_nav = stroke[0][0] - global_last_x
                dy_nav = stroke[0][1] - global_last_y
                if abs(dx_nav) >= 1e-4 or abs(dy_nav) >= 1e-4:
                    deltas.append([dx_nav, dy_nav, 0.0])
            # ─────────────────────────────────────────────────────────────────

            last_x, last_y = stroke[0]

            for i in range(1, len(stroke)):
                x, y = stroke[i]
                dx, dy = x - last_x, y - last_y

                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    last_x, last_y = x, y
                    continue

                stroke_deltas.append([dx, dy, 0.0])
                last_x, last_y = x, y

            # Trazo completamente degenerado (todos los puntos idénticos): no
            # emite nada, ni siquiera pen_lift. El delta de navegación ya fue
            # emitido antes de detectar la degeneración — se deja porque marca
            # correctamente la posición del inicio del trazo vacío; el LSTM
            # aprenderá que un nav sin deltas internos siguientes es infrecuente.
            # Actualizar global_last de todas formas para mantener coherencia.
            global_last_x, global_last_y = last_x, last_y

            if not stroke_deltas:
                continue

            # pen=1 se marca en el ÚLTIMO movimiento real del trazo no-final.
            if not is_ultimo_trazo:
                stroke_deltas[-1][2] = 1.0

            deltas.extend(stroke_deltas)
            # Actualizar posición global al final del trazo emitido
            global_last_x = last_x
            global_last_y = last_y

        if not deltas:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        # Post-proceso A: el primer delta nunca puede ser pen=1.
        if deltas[0][2] == 1.0:
            deltas[0][2] = 0.0

        # Post-proceso B: colapsar pen=1 consecutivos residuales.
        for i in range(len(deltas) - 1):
            if deltas[i][2] == 1.0 and deltas[i + 1][2] == 1.0:
                deltas[i + 1][2] = 0.0

        deltas = np.array(deltas, dtype=np.float32)
        deltas[:, 0] = (deltas[:, 0] - self.mean_dx) / (self.std_dx + 1e-6)
        deltas[:, 1] = (deltas[:, 1] - self.mean_dy) / (self.std_dy + 1e-6)
        deltas[:, :2] = np.clip(deltas[:, :2], -10.0, 10.0)

        sos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        return np.vstack([sos, deltas])

    def _compute_stats(self, num_samples=1000):
        print("Calculando estadísticas de normalización (solo movimientos intra-trazo)...")
        all_dx, all_dy = [], []
        caracteres_posibles = list(self.data_by_char.keys())

        if not caracteres_posibles:
            return 0.0, 1.0, 0.0, 1.0

        for _ in range(num_samples):
            texto = "".join(random.choices(caracteres_posibles, k=random.randint(3, 8)))
            sample = self.get_random_word_sample(texto, espaciado=20)

            for stroke in sample['strokes']:
                if len(stroke) < 2:
                    continue
                last_x, last_y = stroke[0]
                for x, y in stroke[1:]:
                    dx, dy = x - last_x, y - last_y
                    if abs(dx) >= 1e-4 or abs(dy) >= 1e-4:
                        all_dx.append(dx)
                        all_dy.append(dy)
                    last_x, last_y = x, y

        std_dx, std_dy = float(np.std(all_dx)), float(np.std(all_dy))
        print(f"Estadísticas calculadas: dx_std={std_dx:.2f} | dy_std={std_dy:.2f}")
        return float(np.mean(all_dx)), std_dx, float(np.mean(all_dy)), std_dy

    def get_random_word_sample(self, texto, espaciado=20):
        palabra_nueva = {'label': texto, 'strokes': [], 'origin': "sintetico_mix"}
        cursor_x = 0
        descenders = {'g', 'j', 'p', 'q', 'y'}

        for char in texto:
            if char == ' ':
                cursor_x += 150
                continue

            variantes = self.data_by_char.get(char)
            if not variantes:
                continue

            muestra_elegida = random.choice(variantes)

            all_xs = [p[0] for stroke in muestra_elegida['strokes'] for p in stroke]
            all_ys = [p[1] for stroke in muestra_elegida['strokes'] for p in stroke]

            if not all_xs:
                continue

            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            ancho_letra = max_x - min_x
            alto_letra  = max_y - min_y

            if char in descenders:
                baseline_shift = max_y - (alto_letra * 0.3)
            else:
                baseline_shift = max_y

            nuevos_trazos = []
            for stroke in muestra_elegida['strokes']:
                nuevo_stroke = [((x - min_x) + cursor_x, y - baseline_shift) for x, y in stroke]
                nuevos_trazos.append(nuevo_stroke)

            palabra_nueva['strokes'].extend(nuevos_trazos)
            cursor_x += ancho_letra + espaciado

        return palabra_nueva

    def visualize_training_samples(self, n=10):
        print(f"\n--- Iniciando auditoría de {n} muestras de entrenamiento ---")
        print(f"Estadísticas globales del dataset: std_dx={self.std_dx:.4f}, std_dy={self.std_dy:.4f}\n")

        filas, columnas = 2, 5
        fig, axes = plt.subplots(filas, columnas, figsize=(20, 8))
        axes = axes.flatten()

        for i in range(n):
            sample = self.get_training_sample()
            deltas = self.to_deltas(sample)

            all_xs = [p[0] for stroke in sample['strokes'] for p in stroke]
            all_ys = [p[1] for stroke in sample['strokes'] for p in stroke]
            min_x  = min(all_xs) if all_xs else 0
            max_x  = max(all_xs) if all_xs else 0
            min_y  = min(all_ys) if all_ys else 0
            max_y  = max(all_ys) if all_ys else 0

            dx_vals   = deltas[:, 0]
            dy_vals   = deltas[:, 1]
            pen_lifts = deltas[:, 2]

            num_points = len(deltas)
            num_lifts  = int(np.sum(pen_lifts))

            print(f"Muestra {i+1}/{n} | Texto: '{sample['label']}'")
            print(f"  Coordenadas absolutas: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")

            if num_points > 0:
                print(f"  Deltas normales: dx[min={np.min(dx_vals):.2f}, max={np.max(dx_vals):.2f}], "
                      f"dy[min={np.min(dy_vals):.2f}, max={np.max(dy_vals):.2f}]")
                print(f"  Drift acumulado: sum_dx={np.sum(dx_vals):.2f}, sum_dy={np.sum(dy_vals):.2f}")
                print(f"  Pen Lifts: {num_lifts}/{num_points} ({num_lifts/num_points*100:.1f}%)")
            else:
                print("  ¡ADVERTENCIA! Muestra sin puntos válidos.")
            print("-" * 50)

            ax = axes[i]
            for stroke in sample['strokes']:
                if len(stroke) > 0:
                    xs, ys = zip(*stroke)
                    ax.plot(xs, ys, "k-", linewidth=1.5)

            ax.set_title(f"'{sample['label']}'")
            ax.invert_yaxis()
            ax.axis('equal')
            ax.axis('off')

        for j in range(n, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        nombre_archivo = "auditoria_grid.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"\n--- Auditoría completada. Imagen guardada como '{nombre_archivo}' ---")


if __name__ == "__main__":
    dataset = UJIDataset('ujipenchars2.txt')
    dataset.load_dictionary_from_txt('words.txt')

    dataset.visualize_training_samples(10)

    errores = 0
    for _ in range(500):
        sample = dataset.get_training_sample()
        deltas = dataset.to_deltas(sample)
        pen = deltas[:, 2]
        for i in range(len(pen) - 1):
            if pen[i] > 0.5 and pen[i+1] > 0.5:
                errores += 1
                print(f"  CONSECUTIVO en '{sample['label']}' en posición {i}")
                break

    print(f"\nTotal muestras con pen_lifts consecutivos: {errores}/500")