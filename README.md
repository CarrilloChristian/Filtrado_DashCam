# Analizador Automático de Alertas de Video (DMS)

Este proyecto es un sistema automatizado en Python que se conecta a una base de datos, descarga videos de alertas de vehículos y los clasifica utilizando diferentes módulos de análisis de visión por computadora para identificar falsos positivos.

---

## 🚀 Características

El sistema es modular y actualmente soporta el análisis de las siguientes alertas:

* **Acceleration alarm:** Detecta si una aceleración brusca ocurrió a baja velocidad (falso positivo) o a una velocidad considerable (alerta real) mediante OCR.
* **Lane departure:** Utiliza la detección de bordes y la transformada de Hough para identificar si las líneas del carril invaden una zona central, indicando una salida de carril real.
* **Forward Collision Warning:** Clasifica las alertas de colisión frontal basándose en la velocidad promedio del vehículo para descartar el seguimiento en tráfico lento.
* **Driver Distraction:** Similar al anterior, determina si una distracción ocurrió a una velocidad de maniobra (falso positivo) o a una velocidad de crucero (alerta real).
* **Following Distance Monitoring:** Usa un modelo de Deep Learning **(YOLOv4-tiny)** para detectar vehículos y estima la distancia para determinar si es insegura.
* **Pedestrian Collision Warning:** Utiliza **YOLOv4-tiny** para detectar peatones y determina si se encuentran dentro de una "zona de peligro" trapezoidal frente al vehículo.

---

## 🛠️ Prerrequisitos

Antes de ejecutar, asegúrate de tener lo siguiente instalado:

1.  **Python 3.8+**
2.  **Git** y **Git LFS** (si se manejan modelos grandes).
3.  **Tesseract OCR:** Es necesario para los módulos basados en velocidad. Sigue las instrucciones de instalación para tu sistema operativo.
4.  **Archivos del Modelo YOLO:** Los siguientes archivos deben estar en la carpeta raíz del proyecto.
    * `yolov4-tiny.weights`
    * `yolov4-tiny.cfg`
    * `coco.names.txt`

---

## ⚙️ Instalación y Configuración

1.  **Clona el repositorio:**
    ```bash
    git clone [URL_DE_TU_REPOSITORIO]
    cd [NOMBRE_DE_TU_CARPETA]
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura tus credenciales:**
    * Crea un archivo llamado `.env` en la carpeta raíz.
    * Añade las credenciales de tu base de datos, siguiendo el ejemplo de `env.example` (si lo creas).
    ```
    DB_HOST=tu_host
    DB_USER=tu_usuario
    DB_PASSWORD=tu_contraseña
    DB_NAME=nombre_de_la_bbdd
    ```
5.  **Crea el archivo de calibración:**
    * Asegúrate de tener el archivo `calibracion_distancia.csv` en la raíz para el módulo de distancia de seguimiento.

---

## ▶️ Uso

Para ejecutar el análisis completo, simplemente corre el script principal desde la terminal:

```bash
python main.py
