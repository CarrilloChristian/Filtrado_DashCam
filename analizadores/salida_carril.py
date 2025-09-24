import cv2
import numpy as np

#====================================================================================
# --- PARÁMETROS DE AJUSTE ESPECÍFICOS PARA ESTA ALERTA ---
#====================================================================================
GUIA_IZQUIERDA_X = 500
GUIA_DERECHA_X = 1420
ANCHO_ZONA_ALARMA = 100
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 100
SLOPE_THRESHOLD = 0.5

# --- Funciones auxiliares internas del módulo ---
def _detect_all_lane_segments(frame):
    """
    Toma un fotograma y devuelve una lista de los segmentos de línea de carril VÁLIDOS.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    
    # Define una Región de Interés (ROI) para enfocarse solo en la carretera
    roi_vertices = np.array([[(0, height), (int(width*0.45), int(height*0.6)), 
                              (int(width*0.55), int(height*0.6)), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_canny = cv2.bitwise_and(canny, mask)
    
    # Detecta líneas en la ROI
    lines = cv2.HoughLinesP(masked_canny, rho=2, theta=np.pi/180, threshold=HOUGH_THRESHOLD, 
                            lines=np.array([]), minLineLength=20, maxLineGap=100)
    
    # Filtra las líneas para quedarse solo con las que tienen una pendiente significativa (ignora horizontales)
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) == 0: continue # Evita división por cero en líneas verticales
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > SLOPE_THRESHOLD:
                filtered_lines.append(line[0])
    
    return filtered_lines

# --- FUNCIÓN PRINCIPAL DEL MÓDULO ---
def analizar(cap):
    """
    Analiza un video de salida de carril. Devuelve True si es una alerta real, False si no lo es.
    """
    alerta_detectada_en_video = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detecta todas las líneas de carril en el fotograma actual
        lineas_validas = _detect_all_lane_segments(frame)

        # Define la zona de alarma central
        eje_central_x = (GUIA_IZQUIERDA_X + GUIA_DERECHA_X) / 2
        limite_izquierdo_alarma = eje_central_x - (ANCHO_ZONA_ALARMA / 2)
        limite_derecho_alarma = eje_central_x + (ANCHO_ZONA_ALARMA / 2)

        # Comprueba si alguna línea detectada entra en la zona de alarma
        if lineas_validas:
            for x1, y1, x2, y2 in lineas_validas:
                # Si cualquier extremo de la línea (x1 o x2) está dentro de la zona, es una alerta
                if (x1 > limite_izquierdo_alarma and x1 < limite_derecho_alarma) or \
                   (x2 > limite_izquierdo_alarma and x2 < limite_derecho_alarma):
                    alerta_detectada_en_video = True
                    break # Salimos del bucle de líneas
        
        # Si ya detectamos una alerta, no necesitamos seguir procesando el video
        if alerta_detectada_en_video:
            break # Salimos del bucle de fotogramas
    
    return alerta_detectada_en_video