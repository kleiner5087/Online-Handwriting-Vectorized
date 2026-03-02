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
                        # Optimización: Comprensión de listas para mayor velocidad
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
                self.vocabulario = [line.strip() for line in f if line.strip()]
            print(f"Vocabulario cargado: {len(self.vocabulario)} palabras.")
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
        deltas = []
        
        if not sample['strokes'] or not sample['strokes'][0]:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        # Aplanar con marca de inicio de trazo
        flat_points = []
        for s_idx, stroke in enumerate(sample['strokes']):
            for i, pt in enumerate(stroke):
                is_new_stroke = (i == 0 and s_idx > 0)
                flat_points.append((pt, is_new_stroke))

        last_x, last_y = flat_points[0][0]

        for (x, y), is_new_stroke in flat_points[1:]:
            dx, dy = x - last_x, y - last_y

            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                last_x, last_y = x, y
                continue

            # pen_lift=1 cuando el movimiento ocurre con pluma LEVANTADA
            pen_lift = 1.0 if is_new_stroke else 0.0
            deltas.append([dx, dy, pen_lift])
            last_x, last_y = x, y

        deltas = np.array(deltas, dtype=np.float32)

        if len(deltas) > 0:
            deltas[:, 0] /= (self.std_dx + 1e-6)
            deltas[:, 1] /= (self.std_dy + 1e-6)
            deltas[:, :2] = np.clip(deltas[:, :2], -20.0, 20.0)

        sos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        return np.vstack([sos, deltas]) if len(deltas) > 0 else sos

    def _compute_stats(self, num_samples=1000):
        print("Calculando estadísticas de normalización...")
        all_dx, all_dy = [], []
        caracteres_posibles = list(self.data_by_char.keys())
        
        if not caracteres_posibles: 
            return 0.0, 1.0, 0.0, 1.0

        for _ in range(num_samples):
            texto = "".join(random.choices(caracteres_posibles, k=random.randint(3, 8)))
            sample = self.get_random_word_sample(texto, espaciado=20)
            
            is_global_start = True
            last_x, last_y = 0, 0
            
            for stroke in sample['strokes']:
                for x, y in stroke:
                    if is_global_start:
                        last_x, last_y = x, y
                        is_global_start = False
                        continue
                    
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
            if not variantes: continue

            muestra_elegida = random.choice(variantes)
            
            all_xs = [p[0] for stroke in muestra_elegida['strokes'] for p in stroke]
            all_ys = [p[1] for stroke in muestra_elegida['strokes'] for p in stroke]

            if not all_xs: continue

            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            ancho_letra = max_x - min_x
            alto_letra = max_y - min_y
            
            # --- HEURÍSTICA DE LÍNEA BASE ---
            if char in descenders:
                # Si es una letra que "cuelga" (ej. g, p), empujamos su piso un 30% hacia abajo
                baseline_shift = max_y - (alto_letra * 0.3)
            else:
                # Para el resto, el piso (max_y) es la línea base
                baseline_shift = max_y

            nuevos_trazos = []
            for stroke in muestra_elegida['strokes']:
                nuevo_stroke = [((x - min_x) + cursor_x, y - baseline_shift) for x, y in stroke]
                nuevos_trazos.append(nuevo_stroke)
                
            palabra_nueva['strokes'].extend(nuevos_trazos)
            cursor_x += ancho_letra + espaciado 
            
        return palabra_nueva

    def plot_sample(self, sample):
        plt.figure(figsize=(12, 2))
        for stroke in sample['strokes']:
            xs, ys = zip(*stroke)
            plt.plot(xs, ys, "k-", linewidth=2)
        plt.title(f"Texto: {sample['label']}")
        plt.gca().invert_yaxis() 
        plt.axis('equal')
        plt.axis('off') # Limpiamos la gráfica visualmente
        plt.show()

# Bloque de prueba simplificado
if __name__ == "__main__":
    dataset = UJIDataset('ujipenchars2.txt')
    dataset.load_dictionary_from_txt('words.txt')
    
    # Probamos la nueva heurística de línea base
    test_word = dataset.get_random_word_sample("olaquease")
    dataset.plot_sample(test_word)