import os
import logging
from google.cloud import speech_v1 as speech
from tempfile import NamedTemporaryFile

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleSpeechToText:
    """
    Clase para manejar la conversión de audio a texto utilizando Google Speech-to-Text API
    """
    
    def __init__(self, credentials_path=None):
        """
        Inicializa el cliente de Google Speech-to-Text
        
        Args:
            credentials_path: Ruta al archivo de credenciales JSON (opcional)
        """
        # Configurar credenciales si se proporcionan
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning("No se han configurado las credenciales de Google. " 
                         "Asegúrate de configurar GOOGLE_APPLICATION_CREDENTIALS.")
        
        try:
            self.cliente = speech.SpeechClient()
            logger.info("Cliente de Google Speech-to-Text inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el cliente de Google Speech-to-Text: {e}")
            raise
    
    def transcribir_audio(self, archivo_audio, idioma="es-ES", tasa_muestreo=16000, formato=None):
        """
        Transcribe un archivo de audio a texto
        
        Args:
            archivo_audio: Bytes del archivo de audio o ruta al archivo
            idioma: Código del idioma (por defecto 'es-ES')
            tasa_muestreo: Frecuencia de muestreo en Hz (por defecto 16000)
            formato: Formato del audio (None para autodetectar, o usar valores como 'MP3', 'WAV', etc.)
            
        Returns:
            str: Texto transcrito
        """
        try:
            # Verificar si es un archivo m4a y ajustar tasa de muestreo
            if isinstance(archivo_audio, str) and os.path.splitext(archivo_audio)[1].lower() == '.m4a':
                formato = 'MP3'
                tasa_muestreo = 44100
                logger.info(f"Detectado archivo m4a, usando formato MP3 y tasa de muestreo {tasa_muestreo}")
                

            # Determinar el formato de audio
            encoding = self._determinar_formato(archivo_audio, formato)
            
            # Manejar entrada como bytes o como ruta de archivo
            if isinstance(archivo_audio, bytes):
                contenido = archivo_audio
                logger.info(f"Transcribiendo audio desde bytes ({len(contenido)} bytes)")
            else:
                # Es una ruta de archivo
                with open(archivo_audio, "rb") as audio_file:
                    contenido = audio_file.read()
                logger.info(f"Transcribiendo archivo: {archivo_audio}")
            
            # Configurar reconocimiento
            audio = speech.RecognitionAudio(content=contenido)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                sample_rate_hertz=16000,
                language_code="es-419",
                enable_automatic_punctuation=True
            )
            
            # Realizar transcripción
            logger.info("Enviando a Google Speech API...")
            respuesta = self.cliente.recognize(config=config, audio=audio)
            
            # Extraer resultado
            texto_completo = ""
            for resultado in respuesta.results:
                texto_completo += resultado.alternatives[0].transcript + " "
            
            logger.info("Transcripción completada")
            return texto_completo.strip()
            
        except Exception as e:
            logger.error(f"Error durante la transcripción: {e}")
            raise
    
    def _determinar_formato(self, archivo, formato_indicado=None):
        """
        Determina el formato de codificación apropiado para el archivo de audio
        
        Args:
            archivo: Bytes del archivo o ruta al archivo
            formato_indicado: Formato indicado explícitamente (si se conoce)
            
        Returns:
            Formato de codificación para Google Speech API
        """
        if formato_indicado:
            formato = formato_indicado.upper()
        else:
            # Intentar detectar por extensión
            if isinstance(archivo, str) and os.path.exists(archivo):
                extension = os.path.splitext(archivo)[1].lower()
                if extension == '.wav':
                    formato = 'WAV'
                elif extension in ['.mp3', '.mp4', '.m4a']:
                    formato = 'MP3'
                else:
                    # Por defecto
                    formato = 'ENCODING_UNSPECIFIED'
                    logger.warning(f"No se pudo determinar el formato de audio para {extension}. "
                                 f"Usando {formato}")
            else:
                # Si es bytes, usar formato no especificado
                formato = 'ENCODING_UNSPECIFIED'
        
        # Mapear a constantes de Google Speech API
        formato_map = {
            'WAV': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'MP3': speech.RecognitionConfig.AudioEncoding.MP3,
            'FLAC': speech.RecognitionConfig.AudioEncoding.FLAC,
            'OGG': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            'ENCODING_UNSPECIFIED': speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
        }
        
        return formato_map.get(formato, speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)
    
    def guardar_transcripcion(self, texto, ruta_salida=None):
        """
        Guarda el texto transcrito en un archivo
        
        Args:
            texto: Texto a guardar
            ruta_salida: Ruta al archivo de salida (opcional)
            
        Returns:
            str: Ruta al archivo guardado
        """
        if not ruta_salida:
            # Crear un archivo temporal con extensión .txt
            with NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                ruta_salida = tmp.name
        
        try:
            with open(ruta_salida, "w", encoding="utf-8") as f:
                f.write(texto)
            logger.info(f"Transcripción guardada en: {ruta_salida}")
            return ruta_salida
        except Exception as e:
            logger.error(f"Error al guardar la transcripción: {e}")
            raise