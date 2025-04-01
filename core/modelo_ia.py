import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
import pickle  # Para guardar y cargar la base de conocimiento
import unicodedata  # Para normalización de texto
import re  # Para expresiones regulares
import numpy as np  # Para cálculos numéricos


# Intentar importar scikit-learn, con manejo de error si no está instalado
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.pairwise import cosine_similarity
    import joblib
    SKLEARN_DISPONIBLE = True
except ImportError:
    print("scikit-learn no está instalado. Se usarán métodos alternativos de verificación de contexto.")
    SKLEARN_DISPONIBLE = False


class AsistenteJuridico:
    def __init__(self):
        self.qa = None
        self.base_conocimiento = None
        self.clasificador = None
        
        # Ruta base para los archivos
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Archivo donde guardaremos la base de conocimiento procesada
        self.ruta_guardado = self.BASE_DIR / 'data' / 'base_conocimiento.pkl'
        self.ruta_clasificador = self.BASE_DIR / 'data' / 'clasificador_contexto.pkl'
        
        # Inicializamos comprobando si ya existe la base de conocimiento guardada
        self.inicializar_modelo()
        
        # Inicializar el clasificador de contexto si scikit-learn está disponible
        if SKLEARN_DISPONIBLE:
            self._inicializar_clasificador()
    
    def inicializar_modelo(self):
        """
        Inicializa el modelo, cargando la base de conocimiento desde un archivo guardado
        si existe, o creándola desde cero si no existe.
        """
        try:
            # Primero intentamos cargar la base de conocimiento ya procesada
            index_path = str(self.ruta_guardado) + ".faiss"
            if os.path.exists(index_path):
                print(f"Cargando base de conocimiento previamente procesada desde: {index_path}")
                
                # Cargar el índice FAISS
                vectores = OpenAIEmbeddings(
                    api_key=CLAVE_API,
                    model="text-embedding-ada-002"
                )
                
                base_dir = os.path.dirname(self.ruta_guardado)
                self.base_conocimiento = FAISS.load_local(
                    folder_path=base_dir,
                    index_name=os.path.basename(index_path),
                    embeddings=vectores
                )
                
                print("Base de conocimiento cargada exitosamente")
                
                # Una vez cargada la base, configuramos el modelo de QA
                self._configurar_qa()
                return True
            
            # Si no existe el archivo guardado, procesamos el texto desde cero
            return self._procesar_texto_inicial()
                
        except Exception as e:
            print(f"ERROR al inicializar el modelo jurídico: {e}")
            return False
        
    def _inicializar_clasificador(self):
        """
        Inicializa o carga un clasificador para determinar si una consulta está 
        dentro del contexto de tránsito.
        """
        try:
            if os.path.exists(self.ruta_clasificador):
                print("Cargando clasificador de contexto existente...")
                self.clasificador = joblib.load(self.ruta_clasificador)
                return True
            
            print("Creando nuevo clasificador de contexto...")
            # Datos de entrenamiento: ejemplos de consultas de tránsito y no tránsito
            X_train = [
                # Consultas de tránsito (etiqueta: 1)
                "me pararon por exceso de velocidad",
                "el poli me quiere quitar la licencia",
                "me multaron por estacionarme mal",
                "me agarraron manejando sin carnet",
                "me pararon en un control policial",
                "choqué mi auto qué hago",
                "no llevaba mi licencia de conducir",
                "me acusan de cruzar semáforo en rojo",
                "puedo manejar con licencia vencida",
                "el policia me pide coima que hago",
                # Variantes con modismos bolivianos
                "el tombo me quiere quitar el brevete",
                "me chaparon pasando la luz roja",
                "estaba al volante y me pararon los chapas",
                "choque mi movilidad que hago",
                "en la tranca me han coimeado",
                "el oficial me paró sin razón",
                "me quitaron los documentos del auto",
                "cuando manejo borracho qué me puede pasar",
                "estacioné en doble fila y me multaron",
                "no pagué el soat y me detuvieron",
                "tengo una multa pendiente puedo manejar",
                "me pasé el semáforo por accidente",
                "puedo circular sin placa",
                "me pararon por tener vidrios polarizados",
                # Consultas no relacionadas con tránsito (etiqueta: 0)
                "cómo puedo divorciarme",
                "quiero hacer un testamento",
                "me despidieron del trabajo",
                "compré una casa con problemas",
                "tengo problemas con mi arrendador",
                "necesito hacer un contrato",
                "cómo inicio un juicio",
                "cuáles son mis derechos laborales",
                "quiero adoptar un niño",
                "cómo abro una empresa"
            ]
            
            # Etiquetas: 1 para tránsito, 0 para no tránsito
            y_train = [1] * 24 + [0] * 10
            
            # Crear pipeline con TF-IDF y clasificador Naive Bayes
            self.clasificador = Pipeline([
                ('vectorizador', TfidfVectorizer(
                    lowercase=True,  # Convertir todo a minúsculas
                    analyzer='word',  # Analizar por palabras
                    ngram_range=(1, 2),  # Considerar unigramas y bigramas
                    stop_words=['de', 'la', 'el', 'en', 'por', 'con', 'a', 'y'],  # Palabras comunes a ignorar
                    max_features=1000  # Limitar características
                )),
                ('clasificador', MultinomialNB())
            ])
            
            # Entrenar el clasificador
            self.clasificador.fit(X_train, y_train)
            
            # Guardar el clasificador entrenado
            joblib.dump(self.clasificador, self.ruta_clasificador)
            return True
        except Exception as e:
            print(f"Error al inicializar el clasificador: {e}")
            return False
        
    def _procesar_texto_inicial(self):
        """
        Procesa el texto completo por primera vez y guarda la base de conocimiento
        para futuras ejecuciones.
        """
        try:
            # Obtener ruta al archivo de contexto
            ruta_txt = self.BASE_DIR / 'data' / 'completo.txt'
            
            print(f"Cargando conocimiento jurídico desde: {ruta_txt}")
            
            # Verificar que el archivo existe
            if not os.path.exists(ruta_txt):
                print(f"ERROR: El archivo {ruta_txt} no existe.")
                return False
                
            # Cargar el texto desde el archivo
            with open(ruta_txt, 'r', encoding='utf-8') as file:
                texto_completo = file.read()
                
            print(f"Texto cargado: {len(texto_completo)} caracteres")
            
            # Dividir en fragmentos más pequeños para mejor procesamiento
            divisor = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=200,
                length_function=len,
                separators=["\nArtículo", "\nArt.", "\n\n", "\n", ". ", " ", ""]
            )
            
            print("Creando fragmentos de conocimiento jurídico...")
            fragmentos = divisor.create_documents([texto_completo])
            print(f"Se crearon {len(fragmentos)} fragmentos")
            
            # Crear vectores y base de datos
            print("Generando embeddings especializados...")
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            self.base_conocimiento = FAISS.from_documents(fragmentos, vectores)
            
            # MODIFICACIÓN: Usar métodos de FAISS para guardar el índice
            print(f"Guardando base de conocimiento procesada en: {self.ruta_guardado}")
            
            # Creamos un directorio para los archivos de la base de conocimiento
            base_dir = os.path.dirname(self.ruta_guardado)
            os.makedirs(base_dir, exist_ok=True)
            
            # Guardar el índice FAISS y los metadatos de documentos por separado
            # Usamos .pkl para los metadatos y .faiss para el índice vectorial
            index_path = str(self.ruta_guardado) + ".faiss"
            metadata_path = str(self.ruta_guardado) + ".pkl"
            
            # Guardar el índice FAISS
            self.base_conocimiento.save_local(folder_path=base_dir, 
                                            index_name=os.path.basename(index_path))
            
            # Configurar el modelo de QA
            self._configurar_qa()
            
            print("Asistente jurídico especializado en tránsito boliviano inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"ERROR al procesar texto iniciaal: {e}")
            return False
    
    def _configurar_qa(self):
        """
        Configura el modelo de preguntas y respuestas (QA) usando la base de conocimiento.
        """
        try:
            # Modelo de chat optimizado para respuestas jurídicas
            modelo = ChatOpenAI(
                temperature=0.05,  # Temperatura más baja = respuestas más precisas y menos creativas
                # model_name="gpt-4",  # Modelo más avanzado de OpenAI
                model_name="gpt-3.5-turbo",  
                api_key=CLAVE_API,
                max_tokens=4096  # Límite de tokens para las respuestas
            )
            
            # Cadena de preguntas y respuestas
            print("Configurando motor de consultas jurídicas...")
            self.qa = RetrievalQA.from_chain_type(
                llm=modelo,
                chain_type="stuff",  # Método para combinar documentos recuperados
                retriever=self.base_conocimiento.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance - balanceo entre relevancia y diversidad
                    search_kwargs={
                        "k": 8,  # Recuperar 8 fragmentos relevantes (reducido de 12)
                        "fetch_k": 15,  # Considerar 15 documentos (reducido de 20)
                        "lambda_mult": 0.7  # Balance entre relevancia (1.0) y diversidad (0.0)
                    }
                )
            )
            return True
        except Exception as e:
            print(f"ERROR al configurar QA: {e}")
            return False

    def _normalizar_texto(self, texto):
        """
        Normaliza el texto para hacerlo más consistente: elimina tildes,
        convierte a minúsculas, etc.
        """
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar tildes
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                      if unicodedata.category(c) != 'Mn')
        
        # Reemplazar modismos comunes con sus equivalentes estándar
        modismos = {
            "brevete": "licencia",
            "tombo": "policia",
            "chapa": "policia",
            "movilidad": "auto",
            "tranca": "control",
            "chaparon": "detuvieron",
            "coimear": "sobornar",
            "coima": "soborno",
            "placa": "licencia",
            "soat": "seguro"
        }
        
        for modismo, estandar in modismos.items():
            texto = re.sub(r'\b' + modismo + r'\b', estandar, texto)
        
        return texto

    def _verificar_contexto_palabras_clave(self, pregunta_normalizada):
        """
        Verifica si la pregunta contiene palabras clave relacionadas con tránsito.
        """
        palabras_clave = [
            "transito", "policia", "multa", "infraccion", "licencia", 
            "vehiculo", "auto", "coche", "moto", "conducir", "manejar",
            "oficial", "carretera", "ruta", "semaforo", "estacionar",
            "velocidad", "alcoholemia", "choque", "accidente", "control",
            "detencion", "documento", "transporte", "seguro", "placa"
        ]
        
        for palabra in palabras_clave:
            if palabra in pregunta_normalizada:
                return True
        return False

    def _verificar_contexto_clasificador(self, pregunta_normalizada):
        """
        Verifica si la pregunta está en contexto usando el clasificador ML.
        """
        if not SKLEARN_DISPONIBLE or not self.clasificador:
            return None
        
        try:
            # Predecir si está en contexto (1) o fuera de contexto (0)
            esta_en_contexto = self.clasificador.predict([pregunta_normalizada])[0] == 1
            
            # Probabilidad de estar en contexto
            probabilidad = self.clasificador.predict_proba([pregunta_normalizada])[0][1]
            
            print(f"Verificación ML: {esta_en_contexto} (probabilidad: {probabilidad:.2f})")
            
            # Retornamos True solo si la confianza es alta
            if esta_en_contexto and probabilidad > 0.6:
                return True
            elif not esta_en_contexto and probabilidad < 0.3:
                return False
            # Si la confianza es media, devolvemos None para indicar incertidumbre
            return None
        
        except Exception as e:
            print(f"Error en verificación por clasificador: {e}")
            return None

    def _verificar_contexto_semantico(self, pregunta_normalizada):
        """
        Verifica si la pregunta está en contexto usando similitud semántica con embeddings.
        """
        try:
            # Ejemplos de consultas claramente sobre tránsito
            consultas_transito = [
                "infracciones de tránsito",
                "multas de tránsito",
                "licencia de conducir",
                "control policial en carretera",
                "derechos al ser detenido por la policía",
                "documentos de vehículo"
            ]
            
            # Usar el mismo motor de embeddings que usamos para la base de conocimiento
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            # Obtener embedding para la pregunta del usuario
            embedding_pregunta = vectores.embed_query(pregunta_normalizada)
            
            # Calcular embeddings para las consultas de referencia
            embeddings_referencia = [vectores.embed_query(consulta) for consulta in consultas_transito]
            
            # Calcular similitud coseno con cada consulta de referencia
            similitudes = [
                cosine_similarity(
                    np.array(embedding_pregunta).reshape(1, -1),
                    np.array(embedding_ref).reshape(1, -1)
                )[0][0]
                for embedding_ref in embeddings_referencia
            ]
            
            # Tomar la similitud máxima
            max_similitud = max(similitudes)
            print(f"Similitud semántica máxima: {max_similitud:.2f}")
            
            # Si la similitud es mayor que un umbral, consideramos que está en contexto
            return max_similitud > 0.65  # Umbral ajustable
        
        except Exception as e:
            print(f"Error en verificación semántica: {e}")
            return None

    def _verificar_contexto_llm(self, pregunta):
        """
        Verifica si la pregunta está en contexto usando una consulta específica al modelo LLM.
        Este método es el último recurso si los demás fallan.
        """
        try:
            # Modelo más ligero para clasificación
            clasificador_llm = ChatOpenAI(
                temperature=0,  # Sin creatividad para clasificación
                model_name="gpt-3.5-turbo",  # Modelo más económico y rápido
                api_key=CLAVE_API,
                max_tokens=50  # Respuesta corta
            )
            
            # Prompt de clasificación
            prompt_clasificacion = f"""
            Analiza si la siguiente consulta está relacionada con el tránsito vehicular, 
            infracciones de tránsito, trámites vehiculares, control policial en carreteras, 
            o derechos al ser detenido por la policía de tránsito en Bolivia.
            
            Responde solo con "SI" si está relacionada con estos temas o "NO" si no lo está.
            
            Consulta: {pregunta}
            """
            
            # Realizar la consulta
            respuesta = clasificador_llm.predict(prompt_clasificacion)
            
            # Verificar la respuesta
            esta_en_contexto = "SI" in respuesta.upper()
            print(f"Verificación LLM: {esta_en_contexto} - Respuesta: {respuesta}")
            
            return esta_en_contexto
        
        except Exception as e:
            print(f"Error en verificación por LLM: {e}")
            # En caso de error, asumimos que sí está en contexto para no rechazar consultas válidas
            return True

    def verificar_contexto(self, pregunta):
        """
        Verifica si la pregunta está dentro del contexto de tránsito combinando
        diferentes métodos para mayor precisión.
        """
        try:
            # Lista de palabras clave que indican temas claramente NO relacionados con tránsito
            palabras_clave_negativas = [
                "informatica", "programacion", "computadora", "software", "hardware",
                "internet", "web", "app", "aplicacion", "desarrollo", "sistema", "ingenieria",
                "universidad", "carrera", "estudios", "profesion", "cocinar", "receta", "comida",
                "deporte", "futbol", "basket", "tenis", "medicina", "enfermedad", "salud"
            ]
            
            # Normalizar la pregunta
            pregunta_normalizada = self._normalizar_texto(pregunta)
            
            print(f"Verificando contexto para: '{pregunta_normalizada}'")
            
            # Verificación rápida negativa - si contiene palabras clave que NO son de tránsito
            for palabra in palabras_clave_negativas:
                if palabra in pregunta_normalizada:
                    print(f"Fuera de contexto: contiene palabra clave negativa '{palabra}'")
                    return False
            
            # NIVEL 1: Verificación por palabras clave de tránsito (rápido y simple)
            if self._verificar_contexto_palabras_clave(pregunta_normalizada):
                print("Contexto confirmado por palabras clave")
                return True
            
            # NIVEL 2: Verificación por clasificador ML
            resultado_clasificador = self._verificar_contexto_clasificador(pregunta_normalizada)
            if resultado_clasificador is not None:
                print(f"Contexto determinado por clasificador ML: {resultado_clasificador}")
                return resultado_clasificador
            
            # NIVEL 3: Verificación por similitud semántica
            resultado_semantico = self._verificar_contexto_semantico(pregunta_normalizada)
            if resultado_semantico is not None:
                print(f"Contexto determinado por similitud semántica: {resultado_semantico}")
                return resultado_semantico
            
            # NIVEL 4: Verificación por LLM (último recurso)
            print("Usando verificación por LLM como último recurso")
            return self._verificar_contexto_llm(pregunta)
        
        except Exception as e:
            print(f"Error en verificación de contexto: {e}")
            # En caso de error en la verificación, es más seguro asumir que está fuera de contexto
            return False

    def generar_respuesta(self, pregunta):
        """
        Genera una respuesta jurídica basada en la pregunta del usuario.
        """
        if not self.qa:
            print("Error: Sistema no inicializado")
            return {"fueraDeContexto": True, "mensaje": "El sistema no está inicializado correctamente."}
        
        try:
            # Verificar si la pregunta está dentro del contexto jurídico de tránsito
            esta_en_contexto = self.verificar_contexto(pregunta)
            print(f"¿Está en contexto?: {esta_en_contexto}")
            
            if not esta_en_contexto:
                print("Devolviendo respuesta fuera de contexto")
                return {
                    "fueraDeContexto": True,
                    "mensaje": "Como tu amigo legal no tengo ese conocimiento esta fuera de mi contexto."
                }
            
            # Instrucciones mejoradas para respuesta tipo "abogado amigo" (simplificadas)
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
            respuesta_cruda = self.qa.run(prompt)
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
                self._validar_completar_json(respuesta_json)
                
                return respuesta_json
                    
            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON: {e}")
                print(f"Respuesta cruda: {respuesta_cruda}")
                
                # Devolver formato de error con consejos útiles
                return self._formato_error("Error al formatear respuesta")
                    
        except Exception as e:
            print(f"ERROR en generar_respuesta: {str(e)}")
            # Modificamos el formato de error para que incluya fueraDeContexto
            error_response = self._formato_error(f"Error al procesar consulta: {str(e)}")
            # Aseguramos que no se confunda con respuesta normal
            error_response["fueraDeContexto"] = False
            return error_response
            
    def _validar_completar_json(self, respuesta_json):
        """
        Valida que el JSON tenga todos los campos requeridos y los completa si faltan.
        """
        # Campos requeridos en el nuevo formato simplificado
        campos_requeridos = [
            "situacionLegal", 
            "siEresInocente", 
            "siEresCulpable", 
            "frasesClave", 
            "trucosLegales", 
            "derechosEsenciales",
            "infoImportante"
        ]
        
        # Verificar y completar campos faltantes
        for campo in campos_requeridos:
            if campo not in respuesta_json:
                if campo in ["siEresInocente", "siEresCulpable", "frasesClave", "trucosLegales", "derechosEsenciales"]:
                    respuesta_json[campo] = ["Información no disponible"]
                else:
                    respuesta_json[campo] = "Información no disponible"
        
        # Asegurar que los campos de listas sean listas
        campos_lista = ["siEresInocente", "siEresCulpable", "frasesClave", "trucosLegales", "derechosEsenciales"]
        for campo in campos_lista:
            if not isinstance(respuesta_json[campo], list):
                respuesta_json[campo] = [respuesta_json[campo]]
        
        # Asegurar un número mínimo de elementos por cada lista
        min_elementos = {
            "siEresInocente": 3,
            "siEresCulpable": 3,
            "frasesClave": 2,
            "trucosLegales": 2,
            "derechosEsenciales": 2
        }
        
        # Completar con valores predeterminados si no hay suficientes elementos
        self._completar_valores_predeterminados(respuesta_json, min_elementos)
        
        return respuesta_json

    def _completar_valores_predeterminados(self, respuesta_json, min_elementos):
        """
        Completa con valores predeterminados las listas que no tienen suficientes elementos.
        """
        valores_por_defecto = {
            "siEresInocente": [
                "Solicita ver la evidencia de la infracción. Según el Art. 149 del Código de Tránsito, tienen que mostrarte la prueba fehaciente.",
                "Pregunta específicamente qué artículo estás infringiendo. Si no pueden decirte, anótalo como evidencia.",
                "Pide el nombre completo del oficial y su número de placa. El Art. 13 del Reglamento Policial exige identificación visible."
            ],
            "siEresCulpable": [
                "Solicita una boleta oficial en vez de pagar en efectivo. Según el Art. 225, las multas se pagan en entidades bancarias.",
                "Si aceptas la infracción, pide la categoría correcta según el Reglamento para evitar sobrecargos.",
                "No discutas ni niegues lo evidente. Mantén la calma y busca la solución más proporcional según Art. 73 de la Ley 2341."
            ],
            "frasesClave": [
                "'Oficial, entiendo la situación. Por favor, emítame la boleta oficial para pagarla en el banco como establece el Art. 225 del Código de Tránsito.'",
                "'Con todo respeto oficial, necesito conocer el artículo específico que estoy infringiendo para entender la situación.'"
            ],
            "trucosLegales": [
                "Si el equipo de medición (radar) no está calibrado en las últimas 24 horas, puedes impugnar la multa (Art. 149).",
                "Si no especifican el artículo infringido en la boleta, tienes base legal para impugnar (Art. 73 Ley 2341)."
            ],
            "derechosEsenciales": [
                "Derecho a conocer la infracción específica (Art. 16 de la Ley 2341).",
                "Derecho a no ser detenido por infracciones que no constituyan delito penal (Art. 231)."
            ]
        }
        
        # Para cada tipo de lista, completar hasta el mínimo requerido
        for campo, min_valor in min_elementos.items():
            while len(respuesta_json[campo]) < min_valor:
                # Obtener valores predeterminados para este campo
                valores = valores_por_defecto.get(campo, ["Información no disponible"])
                # Agregar el siguiente valor predeterminado que no esté ya en la lista
                for valor in valores:
                    if valor not in respuesta_json[campo]:
                        respuesta_json[campo].append(valor)
                        break

    def _formato_error(self, mensaje):
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