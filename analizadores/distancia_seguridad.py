import cv2
import numpy as np
import pandas as pd

#====================================================================================
# --- PARÁMETROS DE AJUSTE ---
#====================================================================================
UMBRAL_DISTANCIA_ALERTA = 4.0 # Metros
FRAME_SKIP = 5 # YOLO es pesado, saltamos más fotogramas

#====================================================================================
# --- CARGA DE MODELOS (SE EJECUTA UNA SOLA VEZ) ---
#====================================================================================

# --- 1. MODELO DE ESTIMACIÓN DE DISTANCIA ---
try:
    # Asume que el CSV está en la carpeta raíz desde donde se ejecuta main.py
    cal_data = pd.read_csv('calibracion_distancia.csv')
    model_coeffs = np.polyfit(cal_data['pixel_y'], cal_data['distancia_metros'], 2)
    def _estimar_distancia(y_pixel):
        A, B, C = model_coeffs
        return A * y_pixel**2 + B * y_pixel + C
    print("✅ Módulo 'Distancia': Modelo de calibración cargado.")
except FileNotFoundError:
    print("❌ Módulo 'Distancia': No se encontró 'calibracion_distancia.csv'. Las distancias no serán precisas.")
    def _estimar_distancia(y_pixel): return 999

# --- 2. MODELO DE DETECCIÓN DE OBJETOS YOLO ---
try:
    # Asume que los archivos del modelo están en la carpeta raíz
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("✅ Módulo 'Distancia': Modelo YOLO cargado.")
except cv2.error:
    print(f"❌ Módulo 'Distancia': Error al cargar el modelo YOLO. Asegúrate de tener los archivos .weights, .cfg y .names en la carpeta raíz.")
    net = None

#====================================================================================
# --- FUNCIÓN PRINCIPAL DEL MÓDULO ---
#====================================================================================
def analizar(cap):
    """
    Analiza un video para detectar vehículos demasiado cercanos usando YOLO.
    Devuelve True si es una alerta real.
    """
    if not net:
        print("  > Advertencia: El modelo YOLO no está cargado. No se puede analizar la distancia.")
        return False # No podemos analizar si el modelo no cargó

    alerta_detectada = False
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Filtramos por confianza y por tipo de objeto
                if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus"]:
                    center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Non-Max Suppression para eliminar cajas duplicadas
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                # Usamos la base de la caja (y+h) para estimar la distancia
                distancia_estimada = _estimar_distancia(y + h)
                
                if distancia_estimada < UMBRAL_DISTANCIA_ALERTA:
                    alerta_detectada = True
                    break # Salimos del bucle de cajas
        
        if alerta_detectada:
            break # Salimos del bucle de fotogramas
        
        frame_count += 1
        
    return alerta_detectada