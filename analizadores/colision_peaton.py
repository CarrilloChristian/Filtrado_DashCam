import cv2
import numpy as np

#====================================================================================
# --- PARÁMETROS DE AJUSTE ---
#====================================================================================
# Ajustan la forma del trapecio de peligro
ZONA_BOTTOM_WIDTH_PERCENT = 1.1   
ZONA_TOP_WIDTH_PERCENT = 0.07     
ZONA_HEIGHT_PERCENT = 0.45       

# Umbral de confianza de YOLO para detectar a una persona
YOLO_CONFIDENCE_THRESHOLD = 0.6 

# Optimización de Rendimiento
FRAME_SKIP = 5 

#====================================================================================
# --- CARGA DEL MODELO YOLO (SE EJECUTA UNA SOLA VEZ) ---
#====================================================================================
try:
    # Reutilizamos los mismos archivos de modelo que el módulo de distancia
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("✅ Módulo 'Peatón': Modelo YOLO cargado.")
except cv2.error:
    print(f"❌ Módulo 'Peatón': Error al cargar el modelo YOLO.")
    net = None

#====================================================================================
# --- FUNCIÓN PRINCIPAL DEL MÓDULO ---
#====================================================================================
def analizar(cap):
    """
    Analiza un video para detectar peatones en una zona de peligro.
    Devuelve True si es una alerta real.
    """
    if not net:
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
        
        # --- Definir la zona de peligro en trapecio ---
        top_y = int(height * (1 - ZONA_HEIGHT_PERCENT))
        bottom_y = height
        top_left_x = int((width / 2) - (width * ZONA_TOP_WIDTH_PERCENT / 2))
        top_right_x = int((width / 2) + (width * ZONA_TOP_WIDTH_PERCENT / 2))
        bottom_left_x = int((width / 2) - (width * ZONA_BOTTOM_WIDTH_PERCENT / 2))
        bottom_right_x = int((width / 2) + (width * ZONA_BOTTOM_WIDTH_PERCENT / 2))
        danger_zone_vertices = np.array([[(bottom_left_x, bottom_y), (top_left_x, top_y), 
                                          (top_right_x, top_y), (bottom_right_x, bottom_y)]], dtype=np.int32)

        # --- Detección de objetos con YOLO ---
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Buscamos específicamente a una persona con alta confianza
                if confidence > YOLO_CONFIDENCE_THRESHOLD and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    # Calculamos el punto base del peatón (centro en X, parte inferior en Y)
                    y, h = int(detection[1] * height - (detection[3] * height) / 2), int(detection[3] * height)
                    pedestrian_point = (center_x, y + h)
                    
                    # Verificamos si el punto base está dentro del polígono de peligro
                    if cv2.pointPolygonTest(danger_zone_vertices, pedestrian_point, False) >= 0:
                        alerta_detectada = True
                        break # Salimos del bucle de detecciones
        
        if alerta_detectada:
            break # Salimos del bucle de fotogramas
        
        frame_count += 1
        
    return alerta_detectada