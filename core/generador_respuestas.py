import json
import requests
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .formateador_json import FormateadorJSON

class OpenRouterLLM:
    """
    Clase para comunicarse con la API de OpenRouter directamente.
    """
    def __init__(self, api_key, model="deepseek-ai/deepseek-chat", temperature=0.05, max_tokens=4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def predict(self, prompt):
        """
        Envía una consulta a la API de OpenRouter y retorna la respuesta.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://asistentejuridico.app"  # Reemplaza con tu URL
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # Lanza excepción si hay error HTTP
            response_json = response.json()
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0]['message']['content']
            else:
                print(f"Error en la respuesta de OpenRouter: {response_json}")
                return "Error al generar respuesta"
        except Exception as e:
            print(f"Error en la solicitud a OpenRouter: {e}")
            return f"Error de comunicación con el modelo: {str(e)}"

class CustomQA:
    """
    Implementación personalizada de QA para usar con OpenRouter.
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, query):
        # Recuperar documentos relevantes
        documents = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Preparar prompt final con contexto y query
        prompt_with_context = f"Contexto legal:\n{context}\n\nPregunta: {query}"
        
        # Obtener respuesta del modelo
        return self.llm.predict(prompt_with_context)

class GeneradorRespuestas:
    def __init__(self, api_key, usar_openrouter=False):
        self.api_key = api_key
        self.usar_openrouter = usar_openrouter
        self.formateador = FormateadorJSON()
    
    def configurar_qa(self, base_conocimiento):
        """
        Configura el modelo de preguntas y respuestas.
        """
        if self.usar_openrouter:
            # Usar OpenRouter para DeepSeek
            modelo = OpenRouterLLM(
                api_key=self.api_key,
                model="deepseek-ai/deepseek-chat",
                temperature=0.05,
                max_tokens=4096
            )
            
            # Configurar retriever
            retriever = base_conocimiento.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )
            
            # Usar implementación personalizada para DeepSeek
            return CustomQA(retriever, modelo)
        else:
            # Usar OpenAI directamente con LangChain
            modelo = ChatOpenAI(
                temperature=0.05,
                model_name="gpt-3.5-turbo",
                api_key=self.api_key,
                max_tokens=4096
            )
            
            # Cadena de preguntas y respuestas
            print("Configurando motor de consultas jurídicas...")
            qa = RetrievalQA.from_chain_type(
                llm=modelo,
                chain_type="stuff",
                retriever=base_conocimiento.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 8,
                        "fetch_k": 15,
                        "lambda_mult": 0.7
                    }
                )
            )
            
            return qa
    
    def generar_respuesta(self, qa, pregunta):
        """
        Genera una respuesta jurídica basada en la pregunta del usuario.
        """
        # Instrucciones para respuesta tipo "abogado amigo"
        prompt = f"""
        CONTEXTO DE TRABAJO:
        Eres un abogado boliviano experto en tránsito, especialmente de Santa Cruz, actuando como AMIGO PERSONAL 
        del conductor. Hablas con modismos cruceños y de forma cercana pero precisa.
        
        INSTRUCCIONES FUNDAMENTALES:
        1. Habla como un AMIGO ABOGADO que está AHÍ MISMO ayudando.
        2. Distingue claramente entre SI ES CULPABLE y SI NO ES CULPABLE.
        3. Da consejos TÁCTICOS concretos sobre cómo COMPORTARSE y QUÉ DECIR.
        4. Todo consejo debe tener respaldo legal ESPECÍFICO (artículo y número).
        5. Incluye TRUCOS LEGALES que solo un abogado experto conocería.
        6. Si mencionas un artículo específico, incluye un breve resumen de lo que dice.
        7. Usa modismos bolivianos cuando sea apropiado.
        
        FORMATO DE RESPUESTA:
        Debes responder EXCLUSIVAMENTE en formato JSON válido con esta estructura simplificada:
        {{
            "situacionLegal": "Análisis BREVE y DIRECTO de la situación legal con artículos específicos",
            
            "siEresInocente": [
                "Acción táctica 1 si NO cometiste la infracción, con respaldo legal",
                "Acción táctica 2 si NO cometiste la infracción, con respaldo legal",
                "Acción táctica 3 si NO cometiste la infracción, con respaldo legal"
            ],
            
            "siEresCulpable": [
                "Acción táctica 1 si REALMENTE cometiste la infracción, con respaldo legal",
                "Acción táctica 2 si REALMENTE cometiste la infracción, con respaldo legal",
                "Acción táctica 3 si REALMENTE cometiste la infracción, con respaldo legal"
            ],
            
            "frasesClave": [
                "Frase EXACTA 1 para decir al policía (máximo 2 líneas) con respaldo legal",
                "Frase EXACTA 2 para decir al policía (máximo 2 líneas) con respaldo legal"
            ],
            
            "trucosLegales": [
                "Truco legal 1 que solo un ABOGADO EXPERTO conocería, con respaldo legal",
                "Truco legal 2 que solo un ABOGADO EXPERTO conocería, con respaldo legal"
            ],
            
            "derechosEsenciales": [
                "Derecho 1 que NUNCA te pueden quitar, con respaldo legal",
                "Derecho 2 que NUNCA te pueden quitar, con respaldo legal"
            ],
            
            "infoImportante": "Información sobre hora de detención, nombre del oficial, jurisdicción y documentación necesaria"
        }}
        
        CONSEJOS DE TU AMIGO ABOGADO:
        - Pregunta siempre la hora exacta de la detención
        - Verifica la jurisdicción de la carceleta a la que te llevarían
        - Pide siempre el nombre del oficial asignado al caso
        - Revisa atentamente el informe del oficial
        
        PREGUNTA DEL CLIENTE: {pregunta}
        """
        
        print(f"Procesando consulta personalizada: {pregunta}")
        respuesta_cruda = qa.run(prompt)
        print("Respuesta generada como abogado amigo")
        
        # Asegurar que la respuesta sea un JSON válido
        try:
            # Limpiar el texto si es necesario
            if not respuesta_cruda.strip().startswith('{'):
                inicio_json = respuesta_cruda.find('{')
                if inicio_json != -1:
                    respuesta_cruda = respuesta_cruda[inicio_json:]
            
            if not respuesta_cruda.strip().endswith('}'):
                fin_json = respuesta_cruda.rfind('}')
                if fin_json != -1:
                    respuesta_cruda = respuesta_cruda[:fin_json+1]
            
            # Convertir a objeto Python
            respuesta_json = json.loads(respuesta_cruda)
            
            # Verificar campos requeridos y aplicar valores por defecto si faltan
            respuesta_json = self.formateador.validar_completar_json(respuesta_json)
            
            return respuesta_json
                
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON: {e}")
            print(f"Respuesta cruda: {respuesta_cruda}")
            
            # Devolver formato de error con consejos útiles
            return self.formato_error("Error al formatear respuesta")
    
    def formato_error(self, mensaje):
        """
        Devuelve un objeto de error formateado como JSON de un abogado amigo,
        con consejos útiles generales de tránsito.
        """
        return {
            "situacionLegal": "No pude analizar completamente tu situación debido a un error técnico, pero aquí tienes información general sobre derechos en controles de tránsito en Bolivia.",
            "siEresInocente": [
                "Mantén la calma y solicita ver la evidencia de la infracción. El Art. 149 exige prueba fehaciente.",
                "Pregunta el nombre completo y número de placa del oficial. Anótalo para referencia futura.",
                "Toma nota de la hora exacta y ubicación del control. Esto es crucial para cualquier defensa posterior."
            ],
            
            "siEresCulpable": [
                "Solicita una boleta oficial en vez de pagar en efectivo. Según el Art. 225, todo pago debe ser en banco.",
                "Acepta la situación pero verifica que la categoría de la infracción sea la correcta según el Reglamento.",
                "No discutas ni te alteres. La actitud respetuosa siempre facilita la resolución legal del problema."
            ],
            
            "frasesClave": [
                "'Oficial, con respeto solicito recibir la boleta oficial para pagarla por los canales autorizados según el Art. 225.'",
                "'Entiendo la situación, ¿podría por favor indicarme el artículo específico que estoy infringiendo?'"
            ],
            
            "trucosLegales": [
                "La falta de especificación del artículo infringido es base legal para impugnar (Art. 73 Ley 2341).",
                "Las infracciones detectadas por equipos sin calibración reciente pueden ser impugnadas (Art. 149)."
            ],
            
            "derechosEsenciales": [
                "Derecho a conocer la infracción específica (Art. 16 de la Ley 2341).",
                "Derecho a no ser detenido por infracciones que no constituyan delito penal (Art. 231)."
            ],
            
            "infoImportante": "Siempre pregunta y anota: 1) Hora exacta de la detención, 2) Nombre completo del oficial, 3) Jurisdicción de la carceleta si aplica, 4) Solicita revisar el informe del oficial antes de firmar cualquier documento."
        }