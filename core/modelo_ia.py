import os
import json
import re
import unicodedata
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path

x = "sk-proj-"
y = "j5U7Bbt4OxN6OHL3TTidT3BlbkFJMRrveeFpdwbLQCoHqDNG"
z = x + y
CLAVE_API = z

class VerificadorContexto:
    def __init__(self):
        # Palabras clave relacionadas con tránsito
        self.palabras_clave_transito = [
            "transito", "policia", "multa", "infraccion", "licencia", 
            "vehiculo", "auto", "coche", "moto", "conducir", "manejar",
            "oficial", "carretera", "ruta", "semaforo", "estacionar",
            "velocidad", "alcoholemia", "choque", "accidente", "control",
            "detencion", "documento", "transporte", "seguro", "placa",
            "brevete", "tombo", "chapa", "patrulla", "soat", "volante",
            "carnet", "permiso", "papeles", "agente", "manejando", "circular",
            "estacionado", "transit", "codigo"
        ]
        
        # Palabras clave negativas (que indican no-tránsito)
        self.palabras_clave_negativas = [
            "informatica", "programacion", "computadora", "software", "hardware",
            "internet", "web", "app", "aplicacion", "desarrollo", "sistema", 
            "universidad", "carrera", "estudios", "profesion", "cocinar", "receta", 
            "comida", "deporte", "futbol", "tenis", "medicina", "enfermedad", 
            "salud", "divorcio", "herencia", "testamento", "adopcion", "matrimonio",
            "contrato", "alquiler", "propiedad", "hipoteca", "prestamo", "banco"
        ]
        
        # Frases comunes que indican consultas de tránsito
        self.frases_transito = [
            "me pararon", "me detuvieron", "control policial", "control de transito",
            "me multaron", "licencia vencida", "exceso de velocidad", "semaforo rojo",
            "estacionar mal", "sin documentos", "quitaron licencia", "quitaron placa",
            "accidente vehicular", "alcoholemia", "me chocaron", "choque vehicular"
        ]
        
        # Modismos comunes para normalizar
        self.modismos = {
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
    
    def normalizar_texto(self, texto):
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
        for modismo, estandar in self.modismos.items():
            texto = re.sub(r'\b' + modismo + r'\b', estandar, texto)
        
        return texto
    
    def verificar_contexto(self, pregunta):
        """
        Verifica si la pregunta está dentro del contexto de tránsito.
        """
        try:
            # Normalizar la pregunta
            pregunta_normalizada = self.normalizar_texto(pregunta)
            
            print(f"Verificando contexto para: '{pregunta_normalizada}'")
            
            # FASE 1: Verificación rápida por palabras clave de tránsito
            for palabra in self.palabras_clave_transito:
                if re.search(r'\b' + palabra + r'\b', pregunta_normalizada):
                    print(f"Palabra clave de tránsito encontrada: '{palabra}'")
                    # Verificar que no sea un falso positivo
                    return self.confirmar_contexto_transito(pregunta_normalizada)
            
            # FASE 2: Verificación negativa rápida
            for palabra in self.palabras_clave_negativas:
                if re.search(r'\b' + palabra + r'\b', pregunta_normalizada):
                    print(f"Palabra clave NO relacionada con tránsito: '{palabra}'")
                    return False
            
            # FASE 3: Verificación por frases comunes de tránsito
            for frase in self.frases_transito:
                if frase in pregunta_normalizada:
                    print(f"Frase de tránsito encontrada: '{frase}'")
                    return True
            
            # Si no hay suficientes indicios, asumir que no está en contexto
            return False
            
        except Exception as e:
            print(f"Error en verificación de contexto: {e}")
            # En caso de error, ser conservador y asumir fuera de contexto
            return False
    
    def confirmar_contexto_transito(self, pregunta_normalizada):
        """
        Confirma que una pregunta con palabras clave de tránsito realmente está en contexto.
        """
        # Palabras que indican que se está preguntando sobre crear algo
        palabras_creacion = ["crear", "desarrollar", "programar", "hacer", "diseñar", "construir"]
        palabras_tecnologia = ["sistema", "app", "programa", "software", "aplicacion"]
        
        # Si hay palabras de creación y tecnología juntas, probablemente no es de tránsito
        if (any(re.search(r'\b' + palabra + r'\b', pregunta_normalizada) for palabra in palabras_creacion) and
            any(re.search(r'\b' + palabra + r'\b', pregunta_normalizada) for palabra in palabras_tecnologia)):
            print("Falso positivo: pregunta sobre crear un sistema relacionado con tránsito")
            return False
        
        # Si no hay indicios claros en contra, asumir que está en contexto
        return True

class AsistenteJuridico:
    def __init__(self):
        self.qa = None
        self.base_conocimiento = None
        self.clasificador = None
        
        # En lugar de spaCy, usamos nuestro verificador personalizado
        self.verificador = VerificadorContexto()
        
        # Ruta base para los archivos
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Archivo donde guardaremos la base de conocimiento procesada
        self.ruta_guardado = self.BASE_DIR / 'data' / 'base_conocimiento.pkl'
        
        # Ruta a la carpeta de documentos fuente
        self.ruta_documentos = self.BASE_DIR / 'data' 
        
        # Inicializamos comprobando si ya existe la base de conocimiento guardada
        self.inicializar_modelo()
    
    def inicializar_modelo(self):
        """
        Inicializa el modelo, cargando la base de conocimiento desde un archivo guardado
        si existe, o creándola desde cero si no existe.
        """
        try:
            # Primero intentamos cargar la base de conocimiento ya procesada
            import pickle  # Añadir esta importación
    
            index_path = str(self.ruta_guardado) + ".faiss"
            if os.path.exists(index_path):
                print(f"Cargando base de conocimiento previamente procesada desde: {index_path}")
                
                # Cargar usando el método correcto de FAISS
                vectores = OpenAIEmbeddings(
                    api_key=CLAVE_API,
                    model="text-embedding-ada-002"
                )
                
                # Usar el método load_local de FAISS con el parámetro de seguridad
                self.base_conocimiento = FAISS.load_local(
                    folder_path=os.path.dirname(self.ruta_guardado),
                    index_name=os.path.basename(self.ruta_guardado),
                    embeddings=vectores,
                    allow_dangerous_deserialization=True  # Añadir este parámetro
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
    
    def _configurar_qa(self):
        """
        Configura el modelo de preguntas y respuestas basado en la base de conocimiento.
        """
        try:
            # Configurar el modelo de LLM (ChatOpenAI)
            llm = ChatOpenAI(
                api_key=CLAVE_API,
                # model_name="gpt-4-turbo",  # o "gpt-3.5-turbo" según disponibilidad
                model_name="gpt-3.5-turbo",
                temperature=0.97
            )
            
            # Crear la cadena de QA con recuperación
            self.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.base_conocimiento.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # Recuperar los 5 documentos más relevantes
                )
            )
            
            print("Modelo QA configurado exitosamente")
            return True
        except Exception as e:
            print(f"ERROR al configurar el modelo QA: {e}")
            return False
    
    def _procesar_texto_inicial(self):
        """
        Procesa el archivo completo.txt para crear la base de conocimiento.
        """
        try:
            print("Creando base de conocimiento desde los documentos fuente...")
            
            # Verificar que existe la carpeta de documentos
            if not os.path.exists(self.ruta_documentos):
                os.makedirs(self.ruta_documentos, exist_ok=True)
                print(f"Se ha creado la carpeta de documentos en: {self.ruta_documentos}")
                print("Por favor, añade el archivo completo.txt antes de continuar.")
                return False
            
            # Usar específicamente el archivo completo.txt
            archivo_completo = self.ruta_documentos / 'completo.txt'
            
            if not os.path.exists(archivo_completo):
                print(f"No se encontró el archivo completo.txt en: {self.ruta_documentos}")
                print("Por favor, añade el archivo completo.txt antes de continuar.")
                return False
            
            # Cargar y procesar el archivo completo.txt
            textos = []
            try:
                with open(archivo_completo, 'r', encoding='utf-8') as archivo:
                    contenido = archivo.read()
                    textos.append(
                        Document(
                            page_content=contenido,
                            metadata={"source": "completo.txt"}
                        )
                    )
                print(f"Archivo completo.txt cargado exitosamente")
            except Exception as e:
                print(f"Error al cargar el archivo completo.txt: {e}")
                return False
            
            if len(textos) == 0:
                print("No se pudo cargar el archivo completo.txt correctamente.")
                return False
            
            # Dividir los documentos en fragmentos más pequeños
            divisor_texto = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            fragmentos = divisor_texto.split_documents(textos)
            print(f"Se han creado {len(fragmentos)} fragmentos de texto para la base de conocimiento")
            
            # Crear los vectores de embeddings
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            # Crear y guardar la base de conocimiento vectorial
            self.base_conocimiento = FAISS.from_documents(fragmentos, vectores)
            
            # Asegurarse de que existe el directorio para guardar
            os.makedirs(os.path.dirname(self.ruta_guardado), exist_ok=True)
            
            # Guardar la base de conocimiento para futuros usos
            self.base_conocimiento.save_local(
                folder_path=os.path.dirname(self.ruta_guardado),
                index_name=os.path.basename(self.ruta_guardado)
            )
            
            print(f"Base de conocimiento creada y guardada exitosamente en: {self.ruta_guardado}")
            
            # Configurar el modelo QA
            self._configurar_qa()
            return True
            
        except Exception as e:
            print(f"ERROR al procesar texto inicial: {e}")
            return False
        
    def verificar_contexto(self, pregunta):
        """
        Verifica si la pregunta está dentro del contexto de tránsito.
        Ahora usa nuestro verificador personalizado en lugar de spaCy.
        """
        return self.verificador.verificar_contexto(pregunta)
    
    def generar_respuesta(self, pregunta):
        """
        Genera una respuesta jurídica en dos niveles: consejo rápido de amigo legal
        seguido de información técnica detallada con artículos específicos.
        """
        if not self.qa:
            print("Error: Sistema no inicializado")
            return {
                "fueraDeContexto": True,
                "respuestaDirecta": "El sistema no está inicializado correctamente."
            }
        
        try:
            # Verificar contexto
            esta_en_contexto = self.verificar_contexto(pregunta)
            print(f"¿Está en contexto?: {esta_en_contexto}")
            
            if not esta_en_contexto:
                return {
                    "fueraDeContexto": True,
                    "respuestaDirecta": "Como tu asistente legal, no tengo esa información. Puedo ayudarte con temas de (codigo de transito) en Bolivia."
                }
            
            # Instrucciones para respuesta en dos niveles: amigo rápido + legal profundo
            prompt = f"""
                    CONTEXTO: Eres un abogado boliviano experto en codigo de transito de , actuando como ABOGADO LEGAL del conductor.

                    FORMATO DE RESPUESTA JSON:
                    {{
                        "respuestaAmigo": "CONSEJO PRÁCTICO COMPLETO EN UN SOLO PÁRRAFO: Análisis directo sobre la situación legal del conductor, en tono amigable pero profesional. Incluye consejos sobre qué decir, cómo comportarse, qué documentación reunir, si debe grabar el encuentro (audio/video), qué derechos invocar, y cómo actuar ante un posible caso de corrupción. Adaptado específicamente al caso y considerando si el conductor es culpable o inocente.",

                        "accionesInmediatas": [
                            "Acción inmediata específica con pasos detallados",
                            "Segunda acción prioritaria que debe realizar", 
                            "Tercera acción recomendada con justificación legal"
                        ],

                        "análisisLegal": "Análisis jurídico técnico y detallado con referencias a normativas bolivianas específicas aplicables al caso. Incluye fundamentos legales, posibles defensas y consecuencias según el código de tránsito.",

                        "articulosAplicables": [
                            "Artículo específico y en que ley se encuentra el articulo texto exacto y explicación de cómo se aplica al caso",
                            "Segundo artículo relevante  disposición legal precisa y su interpretación favorable para el conductor"
                        ],

                        "derechosFundamentales": [
                            "Derecho específico base legal que lo respalda donde esta para que lo mencione y forma correcta de invocarlo ante las autoridades sobre el procso que hara el policiasl y por que",
                            "Segundo derecho fundamental donde esta y cual es cómo debe ser respetado y qué hacer si es vulnerado "
                        ],

                        "defensaLegal": "Estrategia jurídica específica considerando si el conductor es culpable o no. Incluye argumentos técnicos, precedentes favorables y tácticas procesales para minimizar consecuencias.",

                        "antiCorrupción": "Métodos prácticos para identificar intentos de soborno, cómo documentarlos (grabaciones, testigos), a qué entidades denunciar y procedimiento específico para hacerlo sin exponerse a represalias."
                    }}

                    PREGUNTA DEL CLIENTE: {pregunta}

                    INSTRUCCIONES ADICIONALES:
                    - Analiza si el conductor es CULPABLE o INOCENTE según la pregunta
                    - Ofrece consejos prácticos sobre cómo usar tecnología (grabaciones, fotos) como evidencia
                    - Enfatiza la importancia de mantener RESPETO hacia las autoridades
                    - Indica claramente QUÉ DECIR y QUÉ NO DECIR ante un oficial
                    - Explica cómo identificar y manejar situaciones de posible CORRUPCIÓN
                    - Incluye ARTÍCULOS ESPECÍFICOS en que ley estan del código de tránsito boliviano
                    - Usa lenguaje Detallado en la respuestaAmigo, como si hablaras con un amigo de derecho que te va ayudar
                    - Usa lenguaje TÉCNICO PRESICO DETALLADO en el análisis legal
                    - Toma encuenta que el texto plano de las leyes ya el modelo lo tiene pero si ve necesario adjunta cosas
                """
            print(f"Procesando consulta: {pregunta}")
            respuesta_cruda = self.qa.run(prompt)
            
            # Extraer y procesar JSON
            try:
                import re
                json_match = re.search(r'\{.*\}', respuesta_cruda, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0).replace('\n', ' ').replace('\r', '')
                    return json.loads(json_str)
                else:
                    # Si no podemos extraer JSON, intentar crear una respuesta estructurada
                    partes = respuesta_cruda.split('\n\n')
                    if len(partes) >= 2:
                        return {
                            "respuestaAmigo": partes[0],
                            "análisisLegal": '\n\n'.join(partes[1:]),
                            "fueraDeContexto": False
                        }
                    else:
                        return {
                            "respuestaAmigo": respuesta_cruda,
                            "fueraDeContexto": False
                        }
                        
            except json.JSONDecodeError as e:
                print(f"Error JSON: {e}")
                # Intentar rescatar al menos la respuesta rápida
                primeras_lineas = '\n'.join(respuesta_cruda.split('\n')[:5])
                return {
                    "fueraDeContexto": False,
                    "respuestaAmigo": primeras_lineas,
                    "análisisLegal": "Disculpa, tuve un problema al generar el análisis legal detallado."
                }
                        
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return {
                "fueraDeContexto": False,
                "respuestaAmigo": "Disculpa, ocurrió un error. Intenta con otra pregunta."
            }
            
    def cargar_documentos_texto(self, ruta_directorio):
            """
            Carga todos los documentos de texto (.txt) de un directorio.
            Útil para añadir documentos a la base de conocimiento.
            """
            try:
                documentos = []
                rutas_archivos = list(Path(ruta_directorio).glob('*.txt'))
                
                if len(rutas_archivos) == 0:
                    print(f"No se encontraron archivos .txt en: {ruta_directorio}")
                    return []
                
                for ruta_archivo in rutas_archivos:
                    try:
                        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                            contenido = archivo.read()
                            documentos.append(
                                Document(
                                    page_content=contenido,
                                    metadata={"source": str(ruta_archivo.name)}
                                )
                            )
                        print(f"Documento cargado: {ruta_archivo.name}")
                    except Exception as e:
                        print(f"Error al cargar el documento {ruta_archivo.name}: {e}")
                
                return documentos
            
            except Exception as e:
                print(f"ERROR al cargar documentos de texto: {e}")
                return []
        
    def actualizar_base_conocimiento(self, nuevos_documentos=None, ruta_directorio=None):
        """
        Actualiza la base de conocimiento con nuevos documentos.
        
        Args:
            nuevos_documentos: Lista de objetos Document de langchain
            ruta_directorio: Ruta al directorio con documentos de texto para cargar
        """
        try:
            documentos = []
            
            # Si se proporcionan documentos directamente
            if nuevos_documentos:
                documentos.extend(nuevos_documentos)
            
            # Si se proporciona una ruta de directorio
            if ruta_directorio:
                documentos_cargados = self.cargar_documentos_texto(ruta_directorio)
                documentos.extend(documentos_cargados)
            
            if not documentos:
                print("No se proporcionaron documentos para actualizar la base de conocimiento")
                return False
            
            # Dividir los documentos en fragmentos
            divisor_texto = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            fragmentos = divisor_texto.split_documents(documentos)
            print(f"Se han creado {len(fragmentos)} fragmentos de texto para añadir a la base de conocimiento")
            
            # Crear los vectores de embeddings
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            # Si ya existe una base de conocimiento, añadir los nuevos documentos
            if self.base_conocimiento:
                self.base_conocimiento.add_documents(fragmentos)
                print("Documentos añadidos a la base de conocimiento existente")
            else:
                # Crear nueva base de conocimiento
                self.base_conocimiento = FAISS.from_documents(fragmentos, vectores)
                print("Nueva base de conocimiento creada")
            
            # Guardar la base de conocimiento actualizada
            os.makedirs(os.path.dirname(self.ruta_guardado), exist_ok=True)
            self.base_conocimiento.save_local(
                folder_path=os.path.dirname(self.ruta_guardado),
                index_name=os.path.basename(self.ruta_guardado)
            )
            
            print(f"Base de conocimiento actualizada y guardada en: {self.ruta_guardado}")
            
            # Asegurarse de que el modelo QA esté configurado con la base actualizada
            self._configurar_qa()
            
            return True
            
        except Exception as e:
            print(f"ERROR al actualizar la base de conocimiento: {e}")
            return False