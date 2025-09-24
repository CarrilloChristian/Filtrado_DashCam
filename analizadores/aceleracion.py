import cv2
import pytesseract
import numpy as np
import re

# --- CONSTANTE DE OPTIMIZACIÓN ---
FRAME_SKIP = 10

# --- Lógica de Análisis (funciones auxiliares) ---
# Se les añade un guion bajo para indicar que son de uso interno del módulo
def _es_alerta_real_aceleracion(lista_de_velocidades):
    if not lista_de_velocidades: return False
    return not (lista_de_velocidades[0] < 25)

def _smooth_speeds(speed_list):
    if not speed_list: return []
    window_size, smoothed = 3, []
    padding = window_size // 2
    padded_list = [speed_list[0]] * padding + speed_list + [speed_list[-1]] * padding
    for i in range(len(speed_list)):
        window = sorted(padded_list[i : i + window_size])
        smoothed.append(window[padding])
    return smoothed

# --- FUNCIÓN PRINCIPAL DEL MÓDULO (NOMBRE CORREGIDO) ---
# Cambiamos 'analizar_video_headless' por 'analizar' para que coincida
def analizar(cap):
    """
    Analiza un video de aceleración y devuelve si es una alerta real.
    """
    y1, y2, x1, x2 = 40, 88, 1753, 1825
    raw_velocidades, frame_count, last_known_good_speed = [], 0, None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue
            
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        processed_roi = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 50, 255]))
        config = "-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=0123456789"
        texto_extraido = pytesseract.image_to_string(processed_roi, config=config).strip()
        numeros = re.sub(r'\D', '', texto_extraido)
        if numeros and len(numeros) >= 2:
            try: last_known_good_speed = int(numeros)
            except ValueError: pass
        if last_known_good_speed is not None:
            raw_velocidades.append(last_known_good_speed)

        frame_count += 1

    smoothed_velocidades = _smooth_speeds(raw_velocidades)
    final_velocidades = []
    if smoothed_velocidades:
        final_velocidades.append(smoothed_velocidades[0])
        for i in range(1, len(smoothed_velocidades)):
            if smoothed_velocidades[i] != smoothed_velocidades[i-1]:
                final_velocidades.append(smoothed_velocidades[i])
                
    return _es_alerta_real_aceleracion(final_velocidades)