import os
from google.cloud import speech_v1 as speech

# Corregir la ruta al archivo de credenciales (asegúrate de que esta ruta sea correcta)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\jsahonero\Desktop\Semestre 01 2025\lectorPdfIA\nimble-artwork-438818-k0-ade40f7834b6.json"

def transcribir_m4a(ruta_archivo):
    # El resto del código permanece igual
    cliente = speech.SpeechClient()
    
    print(f"Transcribiendo archivo: {ruta_archivo}")
    
    # Leer el archivo de audio
    with open(ruta_archivo, "rb") as audio_file:
        contenido = audio_file.read()
    
    # Configurar reconocimiento
    audio = speech.RecognitionAudio(content=contenido)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="es-ES",
        enable_automatic_punctuation=True
    )
    
    # Realizar transcripción
    print("Enviando a Google Speech API...")
    respuesta = cliente.recognize(config=config, audio=audio)
    
    # Extraer resultado
    texto_completo = ""
    for resultado in respuesta.results:
        texto_completo += resultado.alternatives[0].transcript + " "
    
    print("Transcripción completada")
    return texto_completo.strip()

if __name__ == "__main__":
    # Ruta al archivo .m4a (reemplaza con la ruta correcta)
    archivo_m4a = "C:\\Users\\jsahonero\\Desktop\\Semestre 01 2025\\lectorPdfIA\\prueba.m4a"
    
    # Verificar que el archivo existe
    if not os.path.exists(archivo_m4a):
        print(f"Error: El archivo {archivo_m4a} no existe")
        exit(1)
    
    # Transcribir
    texto = transcribir_m4a(archivo_m4a)
    
    # Mostrar resultado
    print("\nTexto transcrito:")
    print("-----------------")
    print(texto)
    
    # Guardar en archivo de texto
    nombre_salida = os.path.splitext(archivo_m4a)[0] + "_transcripcion.txt"
    with open(nombre_salida, "w", encoding="utf-8") as f:
        f.write(texto)
    
    print(f"\nTranscripción guardada en: {nombre_salida}")

    # Dirección de correo electrónico: topicos@nimble-artwork-438818-k0.iam.gserviceaccount.com
