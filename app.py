from flask import Flask, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv  # Para cargar variables de entorno
from core import AsistenteJuridico  # Importamos la clase que has creado
from core import GoogleSpeechToText  # Importar el nuevo servicio
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración para la carga de archivos
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar a 16 MB

# Inicializar el asistente jurídico
try:
    asistente = AsistenteJuridico()
    logger.info("Asistente jurídico inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el asistente jurídico: {e}")
    asistente = None

# Inicializar el servicio de transcripción
try:
    # Ruta al archivo de credenciales de Google
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 
                                    os.path.join(os.path.dirname(__file__), 
                                                'nimble-artwork-438818-k0-ade40f7834b6.json'))
    transcriptor = GoogleSpeechToText(credentials_path)
    logger.info("Servicio de transcripción inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el servicio de transcripción: {e}")
    transcriptor = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/consulta', methods=['POST'])
def procesar_consulta():
    if asistente is None:
        return jsonify({"error": "El asistente jurídico no se ha inicializado correctamente"}), 500
    
    try:
        datos = request.get_json()
        if 'pregunta' not in datos:
            return jsonify({"error": "No se proporcionó una pregunta"}), 400
        
        pregunta = datos['pregunta']
        tipo_modelo = datos['tipo-modelo']
        historial_conversacion = datos.get('historial-conversacion', [])
        
        # Convertir historial a string de manera simple
        historial_texto = json.dumps(historial_conversacion, ensure_ascii=False)
        
        # Debug: Imprimir el historial procesado
        # print(f"Historial procesado: {historial_texto}")
        
        # Llamar al asistente con el historial procesado
        respuesta = asistente.generar_respuesta(pregunta, tipo_modelo, historial_texto)
        
        return jsonify(respuesta)
        
    except Exception as e:
        print(f"Error en /api/consulta: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al procesar la consulta: {str(e)}"}), 500
@app.route('/api/audio', methods=['POST'])
def procesar_audio():
    """
    Endpoint para recibir un archivo de audio y transcribirlo a texto
    """
    if transcriptor is None:
        return jsonify({"error": "El servicio de transcripción no está disponible"}), 500
    
    try:
        # Verificar si hay archivo en la solicitud
        if 'archivo' not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400
        
        archivo = request.files['archivo']
        
        # Si el usuario no seleccionó un archivo
        if archivo.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
        if archivo and allowed_file(archivo.filename):
            # Guardar el archivo temporalmente
            filename = secure_filename(archivo.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            archivo.save(filepath)
            
            logger.info(f"Archivo guardado temporalmente en: {filepath}")
            
            # Obtener parámetros opcionales
            idioma = request.form.get('idioma', 'es-ES')
            
            # Transcribir el audio
            formato = 'MP3' if os.path.splitext(filename)[1][1:].lower() == 'm4a' else os.path.splitext(filename)[1][1:].upper()
            tasa_muestreo = 44100 if os.path.splitext(filename)[1][1:].lower() == 'm4a' else 16000

            texto_transcrito = transcriptor.transcribir_audio(
                filepath, 
                idioma=idioma,
                formato=formato,
                tasa_muestreo=tasa_muestreo
            )
            
            logger.info(f"Texto transcrito: {texto_transcrito}")
            
            # Eliminar el archivo temporal
            try:
                os.remove(filepath)
            except:
                pass
                
            # Devolver solo el texto transcrito
            return jsonify({"texto": texto_transcrito})
        else:
            return jsonify({"error": f"Formato de archivo no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
            
    except Exception as e:
        logger.error(f"Error al procesar el audio: {e}")
        return jsonify({
            "error": str(e)
        }), 500
if __name__ == '__main__':
    # Obtener puerto del entorno o usar 5001 por defecto
    port = int(os.environ.get('PORT', 5001))
    
    # Comprobar si estamos en entorno de desarrollo
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Iniciar la aplicación
    app.run(host='0.0.0.0', debug=debug, port=port)