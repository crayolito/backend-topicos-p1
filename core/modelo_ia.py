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

class VerificadorContexto:
    """
    Verificador de contexto especializado para consultas sobre tránsito en Bolivia.
    
    Características:
    - Léxico especializado con terminología boliviana de tránsito
    - Reconocimiento de entidades geográficas bolivianas (ciudades, rutas)
    - Detección de referencias a normativa boliviana específica
    - Sistema avanzado de puntuación con análisis de contexto regional
    - Patrones lingüísticos del español boliviano
    """
    def __init__(self):
        # TERMINOLOGÍA ESPECÍFICA DE BOLIVIA
        
        # Términos de alta relevancia específicos de Bolivia (peso 4)
        self.terminos_bolivia_alta = {
            # Entidades y autoridades
            "transito bolivia": 4, "policia caminera": 4, "transito la paz": 4, 
            "transito santa cruz": 4, "transito cochabamba": 4, "diprove": 4,
            "abc": 4, "vias bolivia": 4, "transito el alto": 4,
            
            # Documentos específicos
            "roseta": 4, "inspeccion tecnica vehicular": 4, "itv": 4,
            "b-sisa": 4, "soat boliviano": 4, "ruat": 4,
            
            # Normativa
            "codigo de transito boliviano": 4, "ley 3988": 4,
            "decreto supremo 23027": 4, "resolucion administrativa 010": 4
        }
        
        # Términos y modismos bolivianos de tránsito (peso 3)
        self.modismos_bolivianos = {
            # Jerga boliviana para policías
            "verde": 3, "caminero": 3, "transistero": 3, "paco": 3,
            
            # Términos locales para vehículos/situaciones
            "trameaje": 3, "trufi": 3, "micro": 3, "flotas": 3,
            "minibus": 3, "taxi trufi": 3, "mototaxi": 3, "surubi": 3,
            "chatarra": 3, "tranca": 3, "control": 3,
            
            # Términos para infracciones/sobornos
            "coima": 3, "mordida": 3, "pisacola": 3, "pasada": 3,
            "invitacion": 3, "colaboracion": 3, "para el refresco": 3,
            
            # Documentos (jerga)
            "carton": 3, "placa": 3, "chapa": 3, "licencia de conducir": 3
        }
        
        # Términos geográficos bolivianos relevantes (peso 2)
        self.geografia_bolivia = {
            # Principales ciudades
            "la paz": 2, "el alto": 2, "cochabamba": 2, "santa cruz": 2, 
            "oruro": 2, "potosi": 2, "sucre": 2, "tarija": 2, "trinidad": 2,
            "cobija": 2,
            
            # Rutas importantes
            "ruta la paz-oruro": 2, "carretera al norte": 2, "carretera nueva": 2,
            "autopista la paz-el alto": 2, "doble via sacaba": 2, "doble via a montero": 2,
            "carretera a los yungas": 2, "carretera a cochabamba": 2,
            "carretera a santa cruz": 2, "ruta bioceánica": 2, "ruta del chaco": 2,
            
            # Puntos de control comunes
            "achica arriba": 2, "huarina": 2, "konani": 2, "patacamaya": 2,
            "panduro": 2, "kilometer 17": 2, "senkata": 2, "rio seco": 2,
            "peaje": 2, "retén": 2
        }
        
        # Palabras clave relacionadas con tránsito con pesos de relevancia
        self.palabras_clave_transito = {
            # Términos de alta relevancia (peso 3)
            "transito": 3, "policia": 3, "multa": 3, "infraccion": 3, 
            "licencia": 3, "vehiculo": 3, "oficial": 3, "transporte": 3,
            "accidente": 3, "choque": 3, "atropello": 3,
            
            # Términos de relevancia media (peso 2)
            "auto": 2, "coche": 2, "moto": 2, "conducir": 2, "manejar": 2,
            "carretera": 2, "ruta": 2, "semaforo": 2, "estacionar": 2,
            "velocidad": 2, "alcoholemia": 2, "control": 2, "detencion": 2,
            "camioneta": 2, "camion": 2, "minibus": 2, "microbus": 2,
            
            # Términos de relevancia baja (peso 1)
            "documento": 1, "seguro": 1, "volante": 1, "carnet": 1, 
            "permiso": 1, "papeles": 1, "agente": 1, "manejando": 1, 
            "circular": 1, "estacionado": 1, "transit": 1, "codigo": 1,
            "asiento": 1, "pasajero": 1, "conductor": 1, "via": 1, "calle": 1
        }
        
        # Términos de categorías específicas
        self.categorias_especificas = {
            # Términos sobre alcoholemia
            "alcoholemia": 3, "alcoholimetro": 3, "test de alcotest": 3,
            "ebriedad": 3, "borracho": 3, "conductor ebrio": 3, "aliento": 2,
            
            # Términos sobre exceso de velocidad
            "radar": 3, "exceso de velocidad": 3, "fotomulta": 3,
            "limite de velocidad": 3, "velocimetro": 2, "kilometros por hora": 2,
            "acelerando": 2,
            
            # Términos sobre documentación
            "documentos vehiculares": 3, "papeles del auto": 3, "tarjeta": 2,
            "inspeccion tecnica": 3, "revision tecnica": 3, "soat": 3,
            "seguro obligatorio": 3, "seguro contra accidentes": 3
        }
        
        # Palabras clave negativas (que indican no-tránsito) con pesos
        self.palabras_clave_negativas = {
            # Tecnología e informática
            "informatica": 3, "programacion": 3, "computadora": 3, "software": 3, 
            "hardware": 3, "internet": 2, "web": 2, "app": 2, "aplicacion": 2, 
            "desarrollo": 2, "sistema": 1, "codigo python": 3, "javascript": 3,
            "programador": 3, "frontend": 3, "backend": 3, "base de datos": 3,
            
            # Educación
            "universidad": 2, "carrera": 1, "estudios": 1, "profesion": 2,
            "umsa": 3, "umss": 3, "upb": 3, "ucb": 3, "unifranz": 3,
            
            # Otros dominios
            "cocinar": 2, "receta": 2, "comida": 2, "deporte": 2, "futbol": 2, 
            "tenis": 2, "medicina": 2, "enfermedad": 2, "salud": 2,
            
            # Derecho no relacionado con tránsito
            "divorcio": 3, "herencia": 3, "testamento": 3, "adopcion": 3, 
            "matrimonio": 3, "contrato": 2, "alquiler": 2, "propiedad": 2, 
            "hipoteca": 2, "prestamo": 2, "banco": 1
        }
        
        # Patrones sintácticos bolivianos específicos de tránsito
        self.patrones_bolivianos = [
            # Patrones de detención/control
            (r'me (pararon|detuvieron|chaparon) (los|el|la) (transito|policia|verde|caminero)', 5),
            (r'me (hicieron|levantaron) un(a)? (acta|boleta|infraccion|multa)', 5),
            (r'(estaba|estuve) en (la|el) (tranca|control|puesto|reten)', 4),
            
            # Patrones de coima/soborno
            (r'(me pidio|me pidieron|queria|querian) (coima|mordida|plata|para el refresco)', 5),
            (r'(me dijo|me dijeron) que (podiamos|podemos) (arreglar|solucionar)', 4),
            (r'(me ofrecio|me ofrecieron) (ayudarme|solucionarlo) por un(a)? (monto|cantidad)', 4),
            
            # Patrones de documentación
            (r'no tenia (licencia|soat|roseta|ruat|itv|b-sisa|carton)', 4),
            (r'(me retuvieron|me quitaron) (mi|el|la) (licencia|placa|auto|moto)', 4),
            (r'(vencio|esta vencido|caduco) (mi|el|la) (licencia|soat|roseta|itv)', 4),
            
            # Patrones de infracción
            (r'(pase|cruce|me pase) (el|un) semaforo en rojo', 4),
            (r'estacion(e|ado) en (lugar|zona) prohibid(o|a)', 4),
            (r'(iba|estaba|me encontraron) (excediendo|pasando) el limite de velocidad', 4)
        ]
        
        # Frases comunes que indican consultas de tránsito
        self.frases_transito = {
            # Situaciones de control policial
            "me pararon": 4, "me detuvieron": 4, "control policial": 4, 
            "control de transito": 4, "me multaron": 4, "licencia vencida": 3,
            "me chaparon": 4, "me agarraron los verdes": 4, "me pararon en la tranca": 4,
            
            # Situaciones de infracción
            "exceso de velocidad": 3, "semaforo rojo": 3, "estacionar mal": 3,
            "sin documentos": 3, "sin soat": 4, "sin roseta": 4, "sin ruat": 4,
            
            # Procedimientos
            "quitaron licencia": 4, "quitaron placa": 4, "secuestraron el auto": 4,
            "decomisaron el vehiculo": 4, "remolcaron mi auto": 4,
            
            # Accidentes
            "accidente vehicular": 3, "alcoholemia": 3, "me chocaron": 3, 
            "choque vehicular": 3, "atropellé a": 4, "me atropellaron": 4,
            
            # Sobornos
            "dar coima": 4, "pedir coima": 4, "me pidio plata": 4,
            "arreglar con el policia": 4, "mordida": 4, "colaboracion": 4
        }
        
        # Modismos comunes para normalizar (especializados para Bolivia)
        self.modismos = {
            # Términos para policía
            "verde": "policia",
            "caminero": "policia",
            "transistero": "policia",
            "paco": "policia",
            
            # Términos para vehículos
            "trufi": "vehiculo",
            "micro": "vehiculo",
            "minibus": "vehiculo",
            "surubi": "vehiculo",
            "movilidad": "auto",
            
            # Términos para documentos
            "brevete": "licencia",
            "carton": "licencia",
            "chapa": "placa",
            "roseta": "itv",
            
            # Términos para control
            "tranca": "control",
            "reten": "control",
            "punto de control": "control",
            
            # Términos para situaciones
            "chaparon": "detuvieron",
            "agarraron": "detuvieron",
            "levantaron": "multaron",
            
            # Términos para soborno
            "coimear": "sobornar",
            "coima": "soborno",
            "mordida": "soborno",
            "refresco": "soborno",
            "colaboracion": "soborno",
            "gastos": "soborno",
            "arreglar": "sobornar",
            "ayudar": "sobornar"
        }
        
        # Umbral de puntuación para considerar contexto de tránsito
        self.umbral_puntaje = 3
        
        # Historial de preguntas para análisis de contexto conversacional
        self.historial_preguntas = []
        self.max_historial = 5  # Máximo de preguntas a mantener en el historial
    
    def normalizar_texto(self, texto):
        """
        Normaliza el texto para hacerlo más consistente: elimina tildes,
        convierte a minúsculas y reemplaza modismos bolivianos por términos estándar.
        
        Args:
            texto (str): Texto a normalizar
            
        Returns:
            str: Texto normalizado
        """
        import unicodedata
        import re
        
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar tildes
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                      if unicodedata.category(c) != 'Mn')
        
        # Reemplazar modismos comunes con sus equivalentes estándar
        for modismo, estandar in self.modismos.items():
            texto = re.sub(r'\b' + modismo + r'\b', estandar, texto)
        
        return texto
    
    def detectar_ngramas(self, texto, ngramas_dict):
        """
        Detecta la presencia de n-gramas (frases) específicos en el texto.
        
        Args:
            texto (str): Texto normalizado a analizar
            ngramas_dict (dict): Diccionario de n-gramas con sus pesos
            
        Returns:
            list: Lista de tuplas (n-grama, peso) encontrados en el texto
        """
        ngramas_encontrados = []
        
        for ngrama, peso in ngramas_dict.items():
            if ngrama in texto:
                ngramas_encontrados.append((ngrama, peso))
                
        return ngramas_encontrados
    
    def calcular_puntaje_texto(self, texto):
        """
        Calcula un puntaje para determinar si el texto está en contexto de tránsito boliviano.
        
        Args:
            texto (str): Texto normalizado para analizar
            
        Returns:
            tuple: (puntaje total, detalles de puntuación)
        """
        import re
        
        puntaje = 0
        detalles = {
            "palabras_positivas": [],
            "ngramas_positivos": [],
            "frases_positivas": [],
            "terminos_bolivia": [],
            "modismos_bolivia": [],
            "geografia_bolivia": [],
            "patrones_bolivia": [],
            "categorias_especificas": [],
            "palabras_negativas": []
        }
        
        # 1. Verificar términos específicos de Bolivia (mayor peso)
        for termino, peso in self.terminos_bolivia_alta.items():
            if termino in texto:
                puntaje += peso
                detalles["terminos_bolivia"].append((termino, peso))
        
        # 2. Verificar modismos bolivianos
        for modismo, peso in self.modismos_bolivianos.items():
            if re.search(r'\b' + modismo + r'\b', texto):
                puntaje += peso
                detalles["modismos_bolivia"].append((modismo, peso))
        
        # 3. Verificar términos geográficos
        for lugar, peso in self.geografia_bolivia.items():
            if re.search(r'\b' + lugar + r'\b', texto):
                puntaje += peso
                detalles["geografia_bolivia"].append((lugar, peso))
        
        # 4. Verificar palabras clave generales de tránsito
        for palabra, peso in self.palabras_clave_transito.items():
            if re.search(r'\b' + palabra + r'\b', texto):
                puntaje += peso
                detalles["palabras_positivas"].append((palabra, peso))
        
        # 5. Verificar categorías específicas (alcoholemia, velocidad, etc.)
        for categoria in self.categorias_especificas.values():
            for termino, peso in categoria.items() if isinstance(categoria, dict) else []:
                if re.search(r'\b' + termino + r'\b', texto):
                    puntaje += peso
                    detalles["categorias_especificas"].append((termino, peso))
        
        # 6. Verificar frases comunes de tránsito
        frases_encontradas = self.detectar_ngramas(texto, self.frases_transito)
        for frase, peso in frases_encontradas:
            puntaje += peso
            detalles["frases_positivas"].append((frase, peso))
        
        # 7. Verificar patrones sintácticos bolivianos
        for patron, peso in self.patrones_bolivianos:
            if re.search(patron, texto):
                puntaje += peso
                detalles["patrones_bolivia"].append((patron, peso))
        
        # 8. Verificar palabras clave negativas (resta puntos)
        for palabra, peso in self.palabras_clave_negativas.items():
            if re.search(r'\b' + palabra + r'\b', texto):
                puntaje -= peso
                detalles["palabras_negativas"].append((palabra, peso))
        
        return puntaje, detalles
    
    def analizar_historial(self):
        """
        Analiza el historial de preguntas para determinar el contexto conversacional.
        
        Returns:
            float: Factor de ajuste para el puntaje basado en el historial (entre -2 y 2)
        """
        if not self.historial_preguntas:
            return 0
        
        # Calcular promedio de puntajes de preguntas anteriores
        puntajes_previos = [p for p, _ in self.historial_preguntas]
        promedio = sum(puntajes_previos) / len(puntajes_previos)
        
        # Dar más peso a las preguntas más recientes
        puntajes_ponderados = [(puntajes_previos[i] * (i + 1)) for i in range(len(puntajes_previos))]
        promedio_ponderado = sum(puntajes_ponderados) / sum(range(1, len(puntajes_previos) + 2))
        
        # Normalizar a un factor entre -2 y 2
        factor = max(min(promedio_ponderado / 10, 2), -2)
        
        return factor
    
    def detectar_falsos_positivos(self, texto, puntaje, detalles):
        """
        Verifica si una pregunta que parece de tránsito podría ser un falso positivo.
        
        Args:
            texto (str): Texto normalizado
            puntaje (float): Puntaje calculado
            detalles (dict): Detalles de la puntuación
            
        Returns:
            tuple: (es_falso_positivo, factor_ajuste)
        """
        import re
        
        # Palabras que indican creación de sistemas o aplicaciones
        palabras_creacion = ["crear", "desarrollar", "programar", "hacer", 
                            "diseñar", "construir", "implementar", "generar"]
        
        palabras_tecnologia = ["sistema", "app", "programa", "software", 
                              "aplicacion", "plataforma", "pagina", "web"]
        
        # Verificar combinaciones de creación + tecnología + tránsito
        if (any(re.search(r'\b' + palabra + r'\b', texto) for palabra in palabras_creacion) and
            any(re.search(r'\b' + palabra + r'\b', texto) for palabra in palabras_tecnologia) and
            len(detalles["palabras_positivas"]) > 0):
            
            # Es probable que sea una pregunta sobre crear un sistema relacionado con tránsito
            return True, -puntaje * 0.8  # Reducir significativamente el puntaje
        
        # Verificar preguntas de programación que mencionan tránsito
        lenguajes_programacion = ["python", "java", "javascript", "c++", 
                                 "codigo", "programacion", "funcion", "clase"]
        
        if (any(re.search(r'\b' + lenguaje + r'\b', texto) for lenguaje in lenguajes_programacion) and
            len(detalles["palabras_positivas"]) < 3):
            return True, -puntaje * 0.7
        
        # Verificar consultas sobre trabajos o profesiones relacionadas con tránsito
        palabras_profesion = ["trabajo", "empleo", "profesion", "contratacion", 
                             "oferta", "vacante", "requisitos", "curriculum"]
        
        if (any(re.search(r'\b' + palabra + r'\b', texto) for palabra in palabras_profesion) and
            (len(detalles.get("terminos_bolivia", [])) == 0)):
            return True, -puntaje * 0.5
        
        return False, 0
    
    def verificar_contexto(self, pregunta):
        """
        Verifica si la pregunta está dentro del contexto de tránsito boliviano.
        
        Args:
            pregunta (str): La pregunta o consulta del usuario
            
        Returns:
            tuple: (está_en_contexto, puntaje, confianza, diagnostico)
                - está_en_contexto (bool): True si está en contexto de tránsito
                - puntaje (float): Puntaje calculado
                - confianza (float): Nivel de confianza en la decisión (0-1)
                - diagnostico (dict): Información detallada sobre la clasificación
        """
        try:
            # Normalizar la pregunta
            pregunta_normalizada = self.normalizar_texto(pregunta)
            
            # Calcular puntaje inicial
            puntaje, detalles = self.calcular_puntaje_texto(pregunta_normalizada)
            
            # Verificar falsos positivos
            es_falso_positivo, ajuste_falso_positivo = self.detectar_falsos_positivos(
                pregunta_normalizada, puntaje, detalles
            )
            
            if es_falso_positivo:
                puntaje += ajuste_falso_positivo
            
            # Considerar el historial conversacional
            factor_historial = self.analizar_historial()
            ajuste_historial = factor_historial
            
            puntaje_final = puntaje + ajuste_historial
            
            # Añadir la pregunta actual al historial
            self.historial_preguntas.append((puntaje, pregunta_normalizada))
            
            # Mantener solo las últimas N preguntas
            if len(self.historial_preguntas) > self.max_historial:
                self.historial_preguntas.pop(0)
            
            # Calcular confianza basada en la distancia al umbral
            confianza = min(abs(puntaje_final - self.umbral_puntaje) / 5, 1.0)
            
            # Determinar si está en contexto basado en el umbral
            esta_en_contexto = puntaje_final >= self.umbral_puntaje
            
            # Información detallada de diagnóstico
            diagnostico = {
                "puntajes": {
                    "terminos_bolivia": sum(peso for _, peso in detalles.get("terminos_bolivia", [])),
                    "modismos_bolivia": sum(peso for _, peso in detalles.get("modismos_bolivia", [])),
                    "geografia_bolivia": sum(peso for _, peso in detalles.get("geografia_bolivia", [])),
                    "palabras_transito": sum(peso for _, peso in detalles.get("palabras_positivas", [])),
                    "frases_transito": sum(peso for _, peso in detalles.get("frases_positivas", [])),
                    "patrones_bolivia": sum(peso for _, peso in detalles.get("patrones_bolivia", [])),
                    "categorias_especificas": sum(peso for _, peso in detalles.get("categorias_especificas", [])),
                    "palabras_negativas": -sum(peso for _, peso in detalles.get("palabras_negativas", [])),
                },
                "ajustes": {
                    "falso_positivo": ajuste_falso_positivo,
                    "historial": ajuste_historial
                },
                "resultados": {
                    "puntaje_bruto": puntaje,
                    "puntaje_final": puntaje_final,
                    "umbral": self.umbral_puntaje,
                    "confianza": confianza,
                    "decision": "en_contexto" if esta_en_contexto else "fuera_contexto"
                },
                "terminos_encontrados": {
                    "bolivia": [t for t, _ in detalles.get("terminos_bolivia", [])],
                    "modismos": [m for m, _ in detalles.get("modismos_bolivia", [])],
                    "geografia": [g for g, _ in detalles.get("geografia_bolivia", [])],
                    "palabras_clave": [p for p, _ in detalles.get("palabras_positivas", [])],
                    "frases": [f for f, _ in detalles.get("frases_positivas", [])],
                    "patrones": [p for p, _ in detalles.get("patrones_bolivia", [])],
                    "negativos": [n for n, _ in detalles.get("palabras_negativas", [])]
                }
            }
            
            # Registrar para debugging (opcional)
            print(f"Verificación de contexto para: '{pregunta_normalizada}'")
            print(f"Resultado: {'En contexto' if esta_en_contexto else 'Fuera de contexto'}")
            print(f"Puntaje final: {puntaje_final}, Confianza: {confianza:.2f}")
            
            return esta_en_contexto, puntaje_final, confianza, diagnostico
            
        except Exception as e:
            print(f"Error en verificación de contexto: {e}")
            # En caso de error, ser conservador y asumir fuera de contexto
            return False, 0, 0, {"error": str(e)}
            
    def procesar_pregunta_para_desarrollo(self, pregunta):
        """
        Método auxiliar para desarrollo y depuración que muestra información
        detallada sobre la clasificación de una pregunta.
        
        Args:
            pregunta (str): La pregunta a analizar
            
        Returns:
            None: Imprime información detallada en consola
        """
        # Normalizar y clasificar
        pregunta_normalizada = self.normalizar_texto(pregunta)
        esta_en_contexto, puntaje, confianza, diagnostico = self.verificar_contexto(pregunta)
        
        # Mostrar resultados detallados
        print("="*50)
        print(f"ANÁLISIS DE PREGUNTA: '{pregunta}'")
        print("="*50)
        print(f"Pregunta normalizada: '{pregunta_normalizada}'")
        print(f"Decisión: {'EN CONTEXTO ✓' if esta_en_contexto else 'FUERA DE CONTEXTO ✗'}")
        print(f"Puntaje: {puntaje:.2f} (umbral: {self.umbral_puntaje})")
        print(f"Confianza: {confianza:.2f}")
        print("-"*50)
        
        # Mostrar desglose de puntajes
        print("DESGLOSE DE PUNTAJES:")
        for categoria, puntos in diagnostico["puntajes"].items():
            if puntos != 0:
                print(f"  - {categoria}: {puntos:.2f}")
        
        # Mostrar ajustes
        if any(diagnostico["ajustes"].values()):
            print("\nAJUSTES APLICADOS:")
            for tipo_ajuste, valor in diagnostico["ajustes"].items():
                if valor != 0:
                    print(f"  - {tipo_ajuste}: {valor:.2f}")
        
        # Mostrar términos encontrados
        print("\nTÉRMINOS DETECTADOS:")
        for categoria, terminos in diagnostico["terminos_encontrados"].items():
            if terminos:
                print(f"  - {categoria}: {', '.join(terminos)}")
        
        print("="*50)

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
                    api_key=z,
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
                # self._configurar_qa()
                return True
            
            # Si no existe el archivo guardado, procesamos el texto desde cero
            return self._procesar_texto_inicial()
                
        except Exception as e:
            print(f"ERROR al inicializar el modelo jurídico: {e}")
            return False
    
    def _configurar_qa(self,tipo_modelo):
        """
        Configura el modelo de preguntas y respuestas basado en la base de conocimiento.
        """
        try:
            # Configurar el modelo de LLM (ChatOpenAI)
            if(tipo_modelo == "basico"):
                llm = ChatOpenAI(
                api_key=z,
                # model_name="gpt-4-turbo",  # o "gpt-3.5-turbo" según disponibilidad
                model_name="gpt-3.5-turbo",
                temperature=0.3
            )
            elif(tipo_modelo == "avanzado"):
                llm = ChatOpenAI(
                api_key=z,
                # model_name="gpt-4-turbo",  # o "gpt-3.5-turbo" según disponibilidad
                model_name="gpt-4",
                temperature=0.3
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
                api_key=z,
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
    
    def generar_respuesta(self, pregunta, tipo_modelo,historial_conversacion):
        """
        Genera una respuesta jurídica en dos niveles: consejo rápido de amigo legal
        seguido de información técnica detallada con artículos específicos.
        """
        self._configurar_qa(tipo_modelo)

        if not self.qa:
            print("Error: Sistema no inicializado")
            return {
                "fueraDeContexto": True,
                "respuestaDirecta": "El sistema no está inicializado correctamente."
            }
        
        try:
            # Verificar contexto
            esta_en_contexto = self.verificar_contexto(pregunta)
            # historial_conversacion = json.dumps({})
            print(f"¿Está en contexto?: {esta_en_contexto}")
            
            if not esta_en_contexto:
                return {
                    "fueraDeContexto": True,
                    "respuestaDirecta": "Como tu asistente legal, no tengo esa información. Puedo ayudarte con temas de (codigo de transito) en Bolivia."
                }
            
            # Instrucciones para respuesta en dos niveles: amigo rápido + legal profundo
            prompt = f"""
            ### CONTEXTO:
            Eres un abogado boliviano experto en código de tránsito, especializado en defender los derechos de conductores frente a situaciones de control policial, infracciones y posibles abusos. Debes proporcionar asesoramiento legal preciso, práctico y específico.

            ### INSTRUCCIONES CRÍTICAS:
            - SIEMPRE especifica los artículos exactos con su número
            - SIEMPRE incluye montos exactos de multas en Bolivianos
            - SIEMPRE menciona cómo actuar ante intentos de soborno
            - SIEMPRE incluye consejos para documentar la situación (grabar, testigos, etc.)
            - NUNCA des respuestas vagas o genéricas

            ### INFORMACIÓN OBLIGATORIA PARA INCLUIR:
            1. Para TODA consulta sobre infracciones:
            - Artículo exacto infringido con número
            - Monto específico de la multa en Bs
            - Si amerita o no retención de vehículo/licencia
            - Procedimiento correcto de emisión de boleta

            2. Para TODA situación de control policial:
            - Derechos específicos del conductor
            - Documentos que legalmente pueden exigirte
            - Procedimiento legal que debe seguir el policía
            - Cómo documentar discretamente (grabar, anotar placa, etc.)
            - Frases específicas para afirmar derechos sin confrontación

            3. Para TODA situación donde pueda existir soborno:
            - Mencionar que es delito de cohecho.
            - Indicar monto legal de la multa para comparación
            - Recomendar grabar discretamente la interacción
            - Frases para rechazar soborno sin provocar confrontación
            - Vías legales para denunciar posteriormente

            ### ESTRUCTURA DE RESPUESTA:
            - Primer párrafo: Explicación legal específica (artículos y multas)
            - Segundo párrafo: Acciones prácticas inmediatas (5-7 puntos concretos)
            - Tercer párrafo: Consejos para situaciones de posible abuso/soborno
            - Todo debe ser específico, directo y útil en situaciones reales
            - No uses muchas palabras tecnicas tiene que llegar a entender cualquier persona

            ### FORMATO DE RESPUESTA JSON:
            {{
                "respuesta": "Respuesta detallada, específica y práctica que incluya TODOS los elementos requeridos: artículos exactos, montos de multas, acciones específicas, consejos ante sobornos y cómo documentar. Debe ser información útil para una situación real.",
                "sugerencia_temas_relacionados": ["Tema relacionado específico 1", "Tema relacionado específico 2"]
            }}

            ### DATOS ACTUALES:
            HISTORIAL DE CONVERSACIÓN: {historial_conversacion}
            PREGUNTA DEL CLIENTE: {pregunta}
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
                api_key=z,
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