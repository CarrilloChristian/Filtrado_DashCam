import os
import cv2
import mysql.connector
import pandas as pd
import requests
import urllib3
from mysql.connector import Error
from dotenv import load_dotenv
from tqdm import tqdm

# --- IMPORTAMOS NUESTROS MÓDULOS DE ANÁLISIS ---
from analizadores import aceleracion
from analizadores import salida_carril
from analizadores import colision_frontal
from analizadores import distraccion_conductor
from analizadores import distancia_seguridad
from analizadores import colision_peaton


# --- Configuración ---
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DB_CONFIG = { 'host': os.getenv('DB_HOST'), 'user': os.getenv('DB_USER'), 'password': os.getenv('DB_PASSWORD'), 'database': os.getenv('DB_NAME') }
BASE_URL = "https://strapi_cemex.gpstechserver.com/dispositivo/evidencias/vid/"
NOMBRE_ARCHIVO_TEMPORAL = "temp_video_analysis.mp4"
NOMBRE_CSV_SALIDA = "reporte_analisis_final.csv"

# --- MAPA DE ANALIZADORES ---
# Este diccionario nos permite elegir la función correcta dinámicamente.
ANALIZADORES = {
    'Acceleration alarm': aceleracion.analizar,
    'Lane departure': salida_carril.analizar,
    'Forward Collision Warning': colision_frontal.analizar,
    'Driver Distraction': distraccion_conductor.analizar,
    'Following Distance Monitoring': distancia_seguridad.analizar,
    'Pedestrian Collision Warning': colision_peaton.analizar,     
}


def procesar_alertas():
    # --- Carga de IDs ya procesados ---
    processed_ids = set()
    columnas_csv = ['dispositivo', 'idEvidencia', 'tipo_alarma', 'falso_positivo', 'resultado', 'url_video']
    if os.path.exists(NOMBRE_CSV_SALIDA):
        try:
            df_existente = pd.read_csv(NOMBRE_CSV_SALIDA)
            processed_ids = set(df_existente['idEvidencia'].astype(str))
            print(f"📄 Se encontraron {len(processed_ids)} videos ya analizados.")
        except pd.errors.EmptyDataError: pass
    else:
        pd.DataFrame(columns=columnas_csv).to_csv(NOMBRE_CSV_SALIDA, index=False, encoding='utf-8-sig')

    # --- Obtener lista de tareas de la BBDD ---
    print("🔌 Conectando a la base de datos...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT dispositivo, idEvidencia, alarma FROM evidencias;")
        todos_los_videos = cursor.fetchall()
    finally:
        if conn and conn.is_connected(): conn.close()

    # Filtramos los videos que ya hemos procesado
    videos_a_procesar = [v for v in todos_los_videos if str(v[1]) not in processed_ids]
    
    if not videos_a_procesar:
        print("✅ No hay videos nuevos para analizar.")
        return

    print(f"⚙️  Se analizarán {len(videos_a_procesar)} videos nuevos.")
    
    for dispositivo, id_evidencia, tipo_alarma in tqdm(videos_a_procesar, desc="Procesando Alertas"):
        
        # Si no tenemos un analizador para este tipo de alarma, lo saltamos
        if tipo_alarma not in ANALIZADORES:
            continue

        url_video = f"{BASE_URL}{dispositivo}/{id_evidencia}/1"
        headers = {'User-Agent': 'Mozilla/5.0'}
        video_descargado = False

        try:
            with requests.get(url_video, headers=headers, stream=True, verify=False, timeout=20) as r:
                if r.status_code == 200:
                    with open(NOMBRE_ARCHIVO_TEMPORAL, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                    video_descargado = True
        except requests.exceptions.RequestException:
            continue # Si falla la descarga, pasamos al siguiente

        if not video_descargado: continue

        cap = cv2.VideoCapture(NOMBRE_ARCHIVO_TEMPORAL)
        if not cap.isOpened(): continue
        
        # --- LLAMADA DINÁMICA AL ANALIZADOR CORRECTO ---
        funcion_de_analisis = ANALIZADORES[tipo_alarma]
        es_real = funcion_de_analisis(cap) # Le pasamos el video abierto
        cap.release()
        
        # --- Guardado de resultados ---
        resultado_texto = "ALERTA REAL" if es_real else "FALSO POSITIVO"
        falso_positivo_valor = 0 if es_real else 1
        
        nueva_fila = pd.DataFrame([{'dispositivo': dispositivo, 'idEvidencia': id_evidencia, 'tipo_alarma': tipo_alarma, 'falso_positivo': falso_positivo_valor, 'resultado': resultado_texto, 'url_video': url_video}])
        nueva_fila.to_csv(NOMBRE_CSV_SALIDA, mode='a', header=False, index=False, encoding='utf-8-sig')
        
    if os.path.exists(NOMBRE_ARCHIVO_TEMPORAL):
        os.remove(NOMBRE_ARCHIVO_TEMPORAL)
    print("\n🏁 Proceso finalizado.")

if __name__ == "__main__":
    procesar_alertas()