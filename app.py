from flask import Flask, request, jsonify
from core import AsistenteJuridicoOpenAI, AsistenteJuridicoDeepSeek

app = Flask(__name__)

# Inicializa ambos asistentes
asistente_openai = AsistenteJuridicoOpenAI()
asistente_deepseek = AsistenteJuridicoDeepSeek()

@app.route('/api/consulta/openai', methods=['POST'])
def procesar_consulta_openai():
    datos = request.get_json()
    if 'pregunta' not in datos:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400
    
    respuesta = asistente_openai.generar_respuesta(datos['pregunta'])
    return jsonify(respuesta)

@app.route('/api/consulta/deepseek', methods=['POST'])
def procesar_consulta_deepseek():
    datos = request.get_json()
    if 'pregunta' not in datos:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400
    
    respuesta = asistente_deepseek.generar_respuesta(datos['pregunta'])
    return jsonify(respuesta)

if __name__ == '__main__':
    app.run(debug=True, port=5001)