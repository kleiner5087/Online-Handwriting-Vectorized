# Generación de Escritura a Mano en Español — LSTM + MDN + Soft Attention

Modelo generativo online (vectorial) de escritura manuscrita en español, entrenado sobre el dataset UJI Pen Characters v2. El sistema
sintetiza palabras completas a partir de caracteres aislados y las genera como secuencias de deltas (dx, dy, pen_lift) usando una LSTM
con ventana de atención suave y una Mixture Density Network con Pen Head como salida.

---

## Dataset

El modelo se entrena sobre el UJI Pen Characters v2, un corpus público compuesto por 11,640 muestras de caracteres manuscritos aislados.
El vocabulario abarca 97 clases únicas, incluyendo letras ASCII, caracteres del español (con acentos y diéresis), dígitos y símbolos
ortográficos. Los datos originales contienen exclusivamente información espacial (coordenadas X/Y cartesianas), careciendo de métricas
de presión o marcas de tiempo.

Nota arquitectónica: El uso actual de este corpus representa la fase fundacional del entrenamiento. El objetivo a largo plazo contempla
el fine-tuning con un dataset propio capturado en dispositivos de lápiz óptico (iPad) para asimilar las ligaduras continuas e intra-secuencia reales.

---

## Ingeniería de Datos (`UJIPen.py`)

El script UJIPen.py actua como el motor de preparacion de datos para un modelo generativo basado en vectores. Su funcion principal es
tomar coordenadas absolutas de trazos manuscritos, ensamblarlas dinamicamente en palabras completas y transformarlas en secuencias de
movimientos relativos, garantizando que los tensores resultantes esten estabilizados matematicamente para el entrenamiento. El flujo
completo de procesamiento de datos se detalla en procesamiento_ujipen.txt

---

## Arquitectura del Modelo (`model.py`)

El sistema implementa una red neuronal recurrente autorregresiva basada en la combinación de capas LSTM, un mecanismo de atención suave
para el condicionamiento de texto y una capa de salida probabilística (MDN) acoplada a un clasificador binario independiente para el control del lápiz.

1. El Codificador (Soft Attention)
   La ventana de atención suave (SoftAttentionWindow) se encarga de guiar al modelo a lo largo de la secuencia de caracteres del texto
   de entrada, permitiéndole mapear qué letra corresponde dibujar en cada paso temporal.

- Mecanismo de Ventana Dinámica: En lugar de utilizar una atención global basada en producto punto, este módulo proyecta el estado de
  la primera capa LSTM mediante una capa lineal para calcular tres parámetros por cada componente de mezcla (K=10): la fuerza de la atención
  (alpha), la precisión del enfoque (beta) y el avance del cursor espacial (delta).
- Acumulación de Posición: El parámetro delta se mantiene estrictamente positivo y acotado por un límite máximo (DELTA_MAX = 0.05). Al
  sumarse de manera acumulativa al parámetro kappa, la red garantiza matemáticamente un desplazamiento monótono hacia adelante a través
  de los embeddings de los caracteres, impidiendo que el mecanismo de atención retroceda o salte de forma errática durante la generación de texto continuo.

2. El Cerebro Secuencial (Capas LSTM)
   El núcleo del modelo procesa las trayectorias espaciales y mantiene el contexto secuencial mediante dos capas LSTM apiladas de 512
   unidades ocultas cada una, optimizadas con normalización de capa (LayerNorm).

- LSTM Capa 1: Recibe como entrada el punto anterior de la secuencia (un vector de dimensión 3 con coordenadas relativas dx, dy y el
  estado del lápiz) concatenado con la ventana de contexto del paso anterior. Su salida alimenta directamente al mecanismo de atención.
  Su aprendizaje se especializa en la alineación espacio-temporal. Analiza el trazo actual y decide cómo debe moverse la ventana de
  atención sobre el texto. Básicamente aprende cuándo y cómo avanzar el cursor de lectura.
- LSTM Capa 2 y Conexión Residual: Esta capa es el verdadero "dibujante" que recibe la salida de la primera capa (la inercia del movimiento
  general), el contexto de la atención (qué letra específica toca dibujar) y la conexión residual (el punto exacto donde se encuentra). Su
  aprendizaje se centra puramente en la morfología de la caligrafía.
  Esta estructura actúa como una conexión de salto (skip connection) implícita que ayuda a preservar la información geométrica fina y el
  contexto a lo largo de la pila recurrente, estabilizando el flujo de gradientes.

3. Las Cabezas de Salida (Split Heads)
   La arquitectura divide sus salidas en dos cabezales independientes que reciben la combinación de características de ambas capas recurrentes
   y el contexto de atención. Esta división previene la competencia destructiva de gradientes en la función de pérdida.

- Cabeza MDN (Mixture Density Network): Una capa lineal mapea las características hacia los parámetros de una mezcla de 10 gaussianas
  bivariadas (M=10). Esta cabeza predice 60 valores en total correspondientes a los coeficientes de mezcla (pi), las medias espaciales
  (mu_x, mu_y), las desviaciones estándar (sigma_x, sigma_y) y el factor de correlación (rho) de las nubes de probabilidad. Esto permite
  que el modelo aprenda las variaciones y estilos naturales de la escritura en lugar de una trayectoria rígida promediada.
- Cabeza Pen (Pen-Lift Head): Un bloque secuencial independiente compuesto por capas lineales y una activación ReLU procesa las mismas
  características para emitir un único logit binario (e_raw). Este valor pasa por una función sigmoide para determinar la probabilidad de
  levantar el lápiz del papel. Al aislar esta tarea de la cabeza MDN, el modelo logra optimizar la métrica de entropía cruzada binaria (BCE)
  con mayor estabilidad, eliminando artefactos visuales de trazos fantasma ("rayoteo") entre caracteres.

---

## Guia de uso

1. Preparacion del entorno
   El pipeline está diseñado para ejecutarse en entornos basados en Linux (vía WSL2 en Windows) para garantizar la correcta compilación y
   compatibilidad de las librerías de aceleración por hardware (CUDA).

  Nota: Instalar el driver de NVIDIA mas reciente para su gpu (Gameready o Studio)

    1.1 Abrir CMD como administrador y descargar e instalar WSL2 con la última versión de Ubuntu. Configurar usuario y contraseña desde la terminal de WSL.
    // wsl --install

    1.2 Actualizar repositorios del sistema
    // sudo apt update && sudo apt upgrade -y

    1.3 Instalar dependencias de Python
    // sudo apt install python3.13 python3.13-venv python3-pip -y

    1.4 Clonar el repositorio y acceder al directorio
    // git clone <https://github.com/kleiner5087/Online-Handwriting-Vectorized.git>
    // cd <NOMBRE_DEL_REPOSITORIO>

    1.5 Crear y activar el entorno virtual
    // python3.13 -m venv venv
    // source venv/bin/activate

    1.6 Instalar requerimientos del proyecto
    // pip install -r requirements.txt

    1.7 Abrir VSC
    // code .

2. Uso de entrenamiento (train.py)
    Para iniciar la fase de entrenamiento base, ejecuta el script principal. Durante la ejecución, la consola monitorea las métricas
    y se genera un archivo csv con las metricas de cada 25 epocas.
    // python -m src.debug_model

    Nota: Los pesos del modelo (checkpoints) se guardarán automáticamente en el directorio ./modelos/ a medida que la red mejore.

3. Uso de inferencia (generate.py)
   El script generate.py permite visualizar las secuencias espaciales generadas por la red. Dado que el modelo predice densidades
   de probabilidad y no puntos deterministas, es fundamental controlar el muestreo.

- Parámetros Críticos:

• --bias: Controla la varianza morfológica (los desplazamientos dx, dy). Un valor alto (ej. 3.0 o superior) reduce el tamaño de la
distribución, forzando al modelo a elegir la trayectoria más probable y limpia, pero con riesgo de colapso modal. Un valor bajo
(ej. 0.5) permite que la red explore distribuciones más amplias, introduciendo mayor aleatoriedad y variaciones orgánicas al trazo.

• --pen_bias: Ajusta la sensibilidad de la cabeza lineal binaria. Interviene directamente en el umbral que decide cuándo el lápiz
debe separarse del papel virtual.

- Casos de uso para evaluación:

• Validación morfológica + mapa de atención
Evalúa el trazo generado y despliega el mapa de calor que muestra cómo el vector de gravedad del mecanismo Soft Attention avanza sobre los caracteres.
// python -m src.generate --texto "escuela"

• Auditoría de varianza
Genera múltiples iteraciones probabilísticas de la misma palabra para comprobar la estabilidad morfológica del modelo.
// python -m src.generate --texto "hola" --mode grid --n 9

• Comparar comportamiento entre palabras
Renderiza varias palabras simultáneas en un solo pase.
// python -m src.generate --mode compare --textos "hola" "mundo" "España" "python"

• SVG limpio para inspección vectorial
Genera un archivo .svg estandarizado.
python -m src.generate --texto "mundo" --svg

---

## Hardware utilizado

- Asus Rog Zephyrus G15, Ryzen 9 6900HS, NVIDIA RTX 3060 Laptop GPU, vía WSL + CUDA.

## Referencia del Dataset

- UJIpenchars2: A Pen-Based Database with More Than 11K Isolated Handwritten Characters
- F. Prat, M. J. Castro, D. Llorens, A. Marzal, J. M. Vilar.
- Universitat Jaume I / Universidad Politécnica de Valencia, 2008.
- <http://www.lrec-conf.org/proceedings/lrec2008/summaries/658.html>.
