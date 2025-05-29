import cv2
import os
import numpy as np
from key import GEMINI_API_KEY
from IPython.display import clear_output

import google.generativeai as genai

# Configure the API key# Send to Gemini
genai.configure(api_key=GEMINI_API_KEY)
# Import the GenerativeModel class from the Google Generative AI library 

# Initialize the model (e.g., 'gemini-pro' for text)
model = genai.GenerativeModel('gemini-2.0-flash')

def extract_frames_from_video(video_path: str, indices: np.ndarray) -> list:
    """
    Abre un video y extrae num_frames fotogramas equiespaciados.

    Args:
        video_path: Ruta al archivo de video.
        num_frames: Número de fotogramas a extraer.

    Returns:
        Lista de fotogramas como arrays de OpenCV (BGR).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Índices de fotogramas equiespaciados
    indices = np.floor(indices * total_frames)
    indices = [int(i) for i in indices]  # Asegurarse de que son enteros
    frames = []
    try: clear_output()
    finally: pass
    os.system("clear")  # Limpiar la consola para mejor legibilidad
    print(f"Extrayendo {len(indices)} fotogramas de {video_path}...")

    # Extraer los fotogramas en los índices calculados
    for target in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        try: clear_output()
        finally: pass
        os.system("clear")  # Limpiar la consola para mejor legibilidad
        print(f"Fotograma {target} extraído correctamente.")

    cap.release()
    return frames, indices

def resize_image(image, scale: float):
    """
    Reduce la resolución de una imagen por un factor dado.

    Args:
        image: Array de imagen en formato OpenCV (BGR).
        scale: Factor de escala (por ejemplo, 0.5 para reducir a la mitad).

    Returns:
        Imagen redimensionada.
    """
    if scale <= 0 or scale > 1:
        raise ValueError("El factor de escala debe estar entre 0 y 1")
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def send_image_to_gemini(image, prompt: str):
    """
    Envía una imagen y prompt a la API de Gemini para extraer el valor de temperatura.

    Args:
        image: Ruta al archivo de imagen o array BGR de OpenCV.
        prompt: Texto con la instrucción para extraer la temperatura.

    Returns:
        Resultado de la generación del modelo (usualmente texto con la temperatura).
    """
    # Preparar la imagen como blob
    if isinstance(image, str):
        with open(image, "rb") as f:
            img_bytes = f.read()
    else:
        # image es un array de OpenCV
        _, buffer = cv2.imencode('.jpg', image)
        img_bytes = buffer.tobytes()

    try: clear_output()
    finally: pass
    os.system("clear")  # Limpiar la consola para mejor legibilidad
    print("Enviando imagen a Gemini para análisis...")

    # Inicializar modelo de Gemini
    response = model.generate_content([
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}}
    ])
    return response.text
    

# Ejemplo de uso:
# frames = extract_frames_from_video("video.mp4", 10)
# temp = send_image_to_gemini(frames[0], "Por favor extrae el valor de temperatura de este termómetro como entero.")
