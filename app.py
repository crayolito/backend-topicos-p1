from flask import Flask, request, jsonify
from flask_cors import CORS  # Para manejar CORS si tienes frontend separado
import os
import logging
from dotenv import load_dotenv  # Para cargar variables de entorno
from core import AsistenteJuridico  # Importamos la clase que has creado

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno (opcional)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Inicializar el asistente jurídico
try:
    asistente = AsistenteJuridico()
    logger.info("Asistente jurídico inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el asistente jurídico: {e}")
    asistente = None

@app.route('/api/consulta', methods=['POST'])
def procesar_consulta():
    if asistente is None:
        return jsonify({"error": "El asistente jurídico no se ha inicializado correctamente"}), 500
    
    try:
        datos = request.get_json()
        if 'pregunta' not in datos:
            return jsonify({"error": "No se proporcionó una pregunta"}), 400
        
        pregunta = datos['pregunta']
        logger.info(f"Consulta recibida: {pregunta}")
        
        # Verificar si la pregunta está en contexto
        esta_en_contexto = asistente.verificar_contexto(pregunta)
        logger.info(f"¿Está en contexto?: {esta_en_contexto}")
        
        # Generar respuesta
        respuesta = asistente.generar_respuesta(pregunta)
        return jsonify(respuesta)
    
    except Exception as e:
        logger.error(f"Error al procesar la consulta: {e}")
        return jsonify({
            "fueraDeContexto": False,
            "respuestaDirecta": "Ocurrió un error al procesar la consulta. Por favor, intenta nuevamente."
        }), 500



if __name__ == '__main__':
    # Obtener puerto del entorno o usar 5001 por defecto
    port = int(os.environ.get('PORT', 5001))
    
    # Comprobar si estamos en entorno de desarrollo
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Iniciar la aplicación
    app.run(host='0.0.0.0', debug=debug, port=port)