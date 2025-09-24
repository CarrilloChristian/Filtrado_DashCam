import cv2
import pytesseract
import numpy as np
import re

#====================================================================================
# --- PARÁMETROS DE AJUSTE ---
#====================================================================================
VELOCIDAD_UMBRAL = 25
# --- ¡OPTIMIZACIÓN AÑADIDA! ---
# Analizará solo 1 de cada N fotogramas para acelerar el proceso.
FRAME_SKIP = 10

# --- Funciones auxiliares internas del módulo ---
def _extraer_velocidades(cap):
    """
    Procesa un video abierto para extraer la secuencia de velocidades usando OCR.
    ¡Ahora con salto de fotogramas para mayor velocidad!
    """
    y1, y2, x1, x2 = 40, 88, 1753, 1825
    raw_velocities = []
    last_known_good_speed = None
    frame_count = 0  # <--- INICIAMOS CONTADOR

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- LÓGICA DE SALTO DE FOTOGRAMAS AÑADIDA ---
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue
        # -------------------------------------------

        # La lógica de OCR es la misma
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        processed_roi = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 50, 255]))
        config = "-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=0123456789"
        texto_extraido = pytesseract.image_to_string(processed_roi, config=config).strip()
        numeros = re.sub(r'\D', '', texto_extraido)
        
        if numeros and len(numeros) >= 2:
            try:
                velocidad = int(numeros)
                last_known_good_speed = velocidad
            except ValueError: pass
        
        if last_known_good_speed is not None:
            raw_velocities.append(last_known_good_speed)

        frame_count += 1 # <--- INCREMENTAMOS CONTADOR
            
    return raw_velocities

# --- FUNCIÓN PRINCIPAL DEL MÓDULO ---
def analizar(cap):
    """
    Analiza un video de colisión frontal. Devuelve True si es una alerta real,
    o False si es un falso positivo.
    """
    lista_velocidades = _extraer_velocidades(cap)
    
    if not lista_velocidades:
        return False # Datos insuficientes se considera Falso Positivo
        
    velocidad_promedio = sum(lista_velocidades) / len(lista_velocidades)
    
    if velocidad_promedio < VELOCIDAD_UMBRAL:
        return False # Falso Positivo
    else:
        return True # Positivo Real