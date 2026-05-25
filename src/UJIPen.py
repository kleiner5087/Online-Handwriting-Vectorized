import json
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset

BAJOS = set("gjpqy")
FLOTANTES = set("'\"ºª")
INTERMEDIOS = set(";:<> -")
UJI_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúÁÉÍÓÚñÑüÜ.,;:?!'\"()%-@$<>¿¡ºª€")

class UJIDataset(Dataset):
    def __init__(self, jsonl_path='./data/ujipenchars2.jsonl', baselines_path='./data/writer_baselines.json', words_path='./data/words.txt', epoch_size=10000, spacing=150):
        self.epoch_size = epoch_size
        self.spacing = spacing
        self.uji_data = {}
        self.w_stats = {}
        self.vocabulario = []

        with open(baselines_path, 'r', encoding='utf-8') as f:
            self.w_stats = json.load(f)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                w_id, char = d['writer'], d['word']
                if w_id not in self.uji_data:
                    self.uji_data[w_id] = {}
                if char not in self.uji_data[w_id]:
                    self.uji_data[w_id][char] = []
                self.uji_data[w_id][char].append(d)

        self.escritores_validos = [w for w in self.uji_data.keys() if w in self.w_stats]

        try:
            with open(words_path, 'r', encoding='utf-8-sig') as f:
                todas = [w.strip() for w in f if w.strip()]
            self.vocabulario = [w for w in todas if 1 <= len(w) <= 8]
        except Exception:
            self.vocabulario = []

        self.mean_dx, self.std_dx, self.mean_dy, self.std_dy = 0.0, 1.0, 0.0, 1.0
        self.mean_dx, self.std_dx, self.mean_dy, self.std_dy = self._compute_stats(2000)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        word_points, texto = self._build_word_strokes()
        deltas = self._strokes_to_deltas(word_points)
        return torch.from_numpy(deltas), texto

    @staticmethod
    def aplicar_transformacion_afin(strokes):
        pts = [p for s in strokes for p in s]
        if not pts:
            return strokes
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (max(xs) + min(xs)) / 2.0
        cy = (max(ys) + min(ys)) / 2.0
        
        tipo = random.choice(['uniforme', 'diferencial_x', 'slant', 'rotacion'])
        
        if tipo == 'uniforme':
            s = random.uniform(0.96, 1.04)
            M = np.array([[s, 0.0], [0.0, s]])
        elif tipo == 'diferencial_x':
            sx = random.uniform(0.97, 1.03)
            M = np.array([[sx, 0.0], [0.0, 1.0]])
        elif tipo == 'slant':
            sh = random.uniform(-0.15, 0.15)
            M = np.array([[1.0, sh], [0.0, 1.0]])
        else:
            ang = math.radians(random.uniform(-1.5, 1.5))
            c, s = math.cos(ang), math.sin(ang)
            M = np.array([[c, -s], [s, c]])
            
        nuevos_trazos = []
        for s_ in strokes:
            nuevo_s = []
            for x, y in s_:
                v = np.array([x - cx, y - cy])
                v_trans = M.dot(v)
                nuevo_s.append([float(v_trans[0] + cx), float(v_trans[1] + cy)])
            nuevos_trazos.append(nuevo_s)
            
        return nuevos_trazos

    def _build_word_strokes(self):
        w_id = random.choice(self.escritores_validos)
        x_height = self.w_stats[w_id]["x_height"]
        
        if self.vocabulario and random.random() < 0.70:
            texto = random.choice(self.vocabulario)
        else:
            length = random.randint(3, 9)
            texto = "".join(random.choices(UJI_CHARS, k=length))

        word_points = []
        cursor_x = 0.0
        char_counts = {}

        for char in texto:
            if char not in self.uji_data[w_id]:
                continue
            
            char_counts[char] = char_counts.get(char, 0) + 1
            count = char_counts[char]
            variantes = self.uji_data[w_id][char]
            
            if count <= len(variantes):
                sample_trazos = variantes[count - 1]['strokes']
            else:
                sample_trazos = random.choice(variantes)['strokes']
                sample_trazos = self.aplicar_transformacion_afin(sample_trazos)
            
            all_xs = [p[0] for st in sample_trazos for p in st]
            all_ys = [p[1] for st in sample_trazos for p in st]
            if not all_xs:
                continue
            
            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            alto = max_y - min_y
            
            if char in BAJOS:
                anchor_y = max_y - (alto * 0.3)
            elif char in FLOTANTES:
                anchor_y = max_y + x_height
            elif char in INTERMEDIOS:
                anchor_y = (max_y - alto / 2.0) + (x_height / 2.0)
            else:
                anchor_y = max_y
            
            for stroke in sample_trazos:
                nuevo_trazo = []
                for x, y in stroke:
                    nuevo_trazo.append([(x - min_x) + cursor_x, y - anchor_y])
                word_points.append(nuevo_trazo)
            
            cursor_x += (max_x - min_x) + self.spacing

        return word_points, texto

    def _strokes_to_deltas(self, word_points):
        deltas = []
        global_last_x, global_last_y = None, None

        valid_strokes = [s for s in word_points if len(s) >= 1]
        if not valid_strokes:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        for s_idx, trazo in enumerate(valid_strokes):
            is_ultimo_trazo = (s_idx == len(valid_strokes) - 1)
            stroke_deltas = []

            if global_last_x is not None:
                dx_nav = trazo[0][0] - global_last_x
                dy_nav = trazo[0][1] - global_last_y
                if abs(dx_nav) >= 1e-4 or abs(dy_nav) >= 1e-4:
                    deltas.append([dx_nav, dy_nav, 0.0])

            last_x, last_y = trazo[0][0], trazo[0][1]

            for i in range(1, len(trazo)):
                x, y = trazo[i]
                dx, dy = x - last_x, y - last_y
                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    last_x, last_y = x, y
                    continue
                stroke_deltas.append([dx, dy, 0.0])
                last_x, last_y = x, y

            global_last_x, global_last_y = last_x, last_y

            if not stroke_deltas:
                continue

            if not is_ultimo_trazo:
                stroke_deltas[-1][2] = 1.0

            deltas.extend(stroke_deltas)

        if not deltas:
            return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        if deltas[0][2] == 1.0:
            deltas[0][2] = 0.0

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
            all_dx, all_dy = [], []

            for _ in range(num_samples):
                word_points, _ = self._build_word_strokes()
                
                for stroke in word_points:
                    if not stroke:
                        continue
                    
                    prev_pt_x, prev_pt_y = stroke[0]
                    for i in range(1, len(stroke)):
                        curr_x, curr_y = stroke[i]
                        dx, dy = curr_x - prev_pt_x, curr_y - prev_pt_y
                        if abs(dx) >= 1e-4 or abs(dy) >= 1e-4:
                            all_dx.append(dx)
                            all_dy.append(dy)
                            prev_pt_x, prev_pt_y = curr_x, curr_y

            mean_dx = float(np.mean(all_dx)) if all_dx else 0.0
            mean_dy = float(np.mean(all_dy)) if all_dy else 0.0
            std_dx = float(np.std(all_dx)) if all_dx else 1.0
            std_dy = float(np.std(all_dy)) if all_dy else 1.0
            
            return mean_dx, std_dx, mean_dy, std_dy
