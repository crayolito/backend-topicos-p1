from flask import Flask, request, jsonify
from core.modelo_ia import AsistenteJuridico
import json

app = Flask(__name__)

# Crear instancia del asistente jurídico
print("Inicializando el asistente jurídico...")
asistente = AsistenteJuridico()

@app.route('/api/consulta', methods=['POST'])
def consulta_juridica():
    data = request.json
    
    if not data or 'pregunta' not in data:
        return jsonify({
            "explicarSituacion": "No se proporcionó una consulta válida",
            "mejoresOpciones": [
                "Formular una pregunta clara sobre tránsito", 
                "Incluir detalles específicos de su situación", 
                "Consultar sobre un artículo específico del código de tránsito"
            ],
            "peoresOpciones": [
                "Enviar consultas vacías", 
                "Proporcionar información incompleta", 
                "Consultar sobre temas no relacionados con tránsito"
            ],
            "respuestaLegal": "Oficial, en este momento estoy consultando con mi asesor legal sobre mi situación. Le solicito unos minutos para obtener la orientación adecuada."
        }), 400
    
    # Obtener la pregunta del usuario
    pregunta = data['pregunta']
    
    # Generar respuesta usando el asistente jurídico
    respuesta = asistente.generar_respuesta(pregunta)
    
    # Asegurar que la respuesta sea serializable a JSON
    return jsonify(respuesta)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "message": "El servicio está funcionando correctamente",
        "modelo_cargado": asistente.qa is not None
    })

@app.route('/')
def home():
    return jsonify({
        "mensaje": "API de asistente jurídico para defensa de conductores en Bolivia",
        "endpoints": {
            "/api/consulta": "POST - Realizar una consulta jurídica (requiere campo 'pregunta')",
            "/api/health": "GET - Verificar estado del servicio"
        },
        "ejemplo": {
            "pregunta": "¿Qué debo hacer si un policía me detiene por exceso de velocidad?"
        }
    })

# Configurar para una mejor visualización de JSON en respuestas
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.json.sort_keys = False  # Mantener el orden de los campos en el JSON

if __name__ == '__main__':
    print("Iniciando servidor en puerto 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)