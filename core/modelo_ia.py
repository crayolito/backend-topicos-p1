import os
import json
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import re

x = "sk-proj-"
y = "macETBBxiqF74MwjeFXSjRb4FINl5GyhKK-qIWYJxPOE_5MeAKTtTcuzK6VnJNR4q1g79T4dpGT3BlbkFJr17fqDwBf_xEmv3y0ztA1SQ3kST3Sifn1NAdht-gUgBae7AkiQhbO-VhNQ19YTn7cfMPBL9VkA"
z = x + y
CLAVE_API = z

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
            "hipoteca": 2, "prestamo": 2, "banco": 1,

            # Términos de conversación general
            "buenos dias": 4, "hola": 3, "como estas": 4, "buen dia": 4,
            
            # Solicitudes de tareas escolares
            "tarea": 4, "escuela": 4, "trabajo escolar": 5, "actividad escolar": 5,
            "deberes": 4, "sociales": 4, "historia": 4, "investigacion escolar": 5,
            
            # Términos de contenido violento o inapropiado
            "matar": 8, "asesinar": 8, "lastimar": 7, "herir": 7, "violencia": 7,
            "robar": 7, "estafar": 7, "hackear": 7, "ilegal": 5, "trampa": 5,
            "fraude": 7, "dañar": 6, "violento": 7, "arma": 7, "drogas": 7,
        }
        
        # Patrones sintácticos bolivianos específicos de tránsito
        self.patrones_bolivianos = [
            (r'(que|qué|cuál|cual|cuanto|cuánto).*(licencia|placa|soat|multa|infraccion|sancion)', 5),

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
            (r'(iba|estaba|me encontraron) (excediendo|pasando) el limite de velocidad', 4),

            # === DOCUMENTACIÓN ===
            (r'(no|me olvide|deje|se me quedo|perdi|falta).*(licencia|brevet|carnet|registro|permiso|papeles)', 5),
            (r'(sin|falta|no tengo|no llevo).*(licencia|papeles|documentos|permiso|carnet)', 5),
            (r'(licencia|papeles|carnet|permiso).*(olvide|perdi|deje|no traje|no tengo|falta)', 5),
            (r'(vencio|paso|expiro|caduco).*(licencia|carnet|permiso|soat|placa)', 4),

            # === INFRACCIONES ===
            (r'(pase|cruce|me pase|ignore).*(luz|semaforo|señal|pare).*(rojo|roja|stop|alto)', 5),
            (r'(iba|estaba|me encontraron|me agarraron).*(rapido|veloz|a toda|volando|acelerado)', 5),
            (r'(exceso|excedi|sobrepase).*(limite|velocidad|rapidez|permitido)', 5),
            (r'(estacione|deje|pare).*(mal|donde no|prohibido|indebido|no permiten)', 5),
            (r'(tome|bebi|estaba|iba|me encontraron).*(alcohol|bebidas|cerveza|trago|chupado|borracho)', 5),
            (r'(iba|me meti|circule|maneje).*(sentido contrario|contramano|direccion prohibida|marcha atras)', 5),
            (r'(sin|no|falta|falla).*(luces|luz|faro|focos|frenos|cinturon|seguridad)', 5),
            (r'(uso|usando|con|hablando).*(celular|telefono|movil|smartphone).*(conducir|manejar|manejando)', 5),

            # === ACCIDENTES ===
            (r'(tuve|sufri|ocurrio|paso|me paso).*(accidente|choque|colision|impacto|volcadura)', 5),
            (r'(choque|impacte|golpee|atropelle|me lleve).*(auto|persona|peaton|vehiculo|moto|ciclista)', 5),
            (r'(me chocaron|me golpearon|me impactaron|me atropellaron)', 5),
            (r'(hubo|hay|con|causo|ocasiono).*(heridos|victimas|lesionados|daños|muertos|fallecidos)', 5),
            (r'(daños|perjuicios|costo|precio|valor).*(reparacion|arreglo|compostura)', 5),
            (r'(culpa|culpable|responsable|responsabilidad).*(accidente|choque|siniestro)', 5),
            (r'(seguro|cobertura|poliza).*(cubre|paga|responsabilidad|daños)', 5),

            # === AUTORIDADES Y PROCEDIMIENTOS ===
            (r'(me|acaban).*(par(o|aron)|agarr(o|aron)|detuv(o|ieron)|frena(ron)|pesc(o|aron))', 5),
            (r'(policia|transito|verde|autoridad|agente).*(par(o|aron)|agarr(o|aron)|detuv(o|ieron))', 5),
            (r'(me|van|quieren|acaban).*(multar|sancionar|infraccionar|cobrar|poner parte)', 5),
            (r'(multa|infraccion|sancion|ticket|boleta).*(cuanto|valor|monto|precio|pagar)', 5),
            (r'(me|van|quieren|pueden).*(quitar|retener|sacar|llevar|decomisar).*(licencia|auto|placa)', 5),
            (r'(me|van|quieren|han).*(arrestar|detener|encerrar|llevar preso|meter preso)', 5),
            (r'(me|pidio|quiso|planteo).*(coima|mordida|arreglo|plata).*(policia|agente|transito)', 5),
            (r'(como|que hacer|debo|tengo).*(evitar|rechazar|negar|denunciar).*(coima|soborno|mordida)', 5),

            # === TRÁMITES Y CONSULTAS ===
            (r'(como|donde|que necesito|requisitos|tramite).*(sacar|renovar|obtener).*(licencia|brevet)', 5),
            (r'(como|donde|que necesito|requisitos|tramite).*(transferir|traspasar|cambiar).*(auto|vehiculo)', 5),
            (r'(revision|inspeccion|control).*(tecnico|tecnica|vehicular|anual)', 4),
            (r'(pago|impuesto|impositivo|tasa).*(vehicular|municipal|circulacion|propiedad)', 4),
            (r'(que|cuales|donde|como).*(documentos|papeles|requisitos).*(llevar|circular|conducir|traer)', 5)
        ]
        
        self.preguntas_basicas_transito = [
            r'(que|qué).*(pasa|ocurre|sucede).*(sin|no).*(licencia|placa|soat)',
            r'(multa|sancion|pena).*(licencia|placa|soat|conducir)',
            r'(cuanto|cuánto).*(cuesta|vale|paga).*(multa|infraccion|sancion)'

            # === DOCUMENTACIÓN ===
            r'(que|cuanto).*(pasa|cobran|multan|hacen).*sin.*(licencia|papeles|permiso|carnet)',
            r'(no tengo|me faltan|sin|olvide|olvidado).*(licencia|papeles|carnet|permiso).*que.*(hago|puedo|debo)',
            r'(se me vencio|caduco|expiro).*(licencia|registro|permiso).*que.*(hago|puedo|debo)',

            # === INFRACCIONES ===
            r'(que|cuanto).*(pasa|multa|sancion|cobran).*(luz|semaforo).*(rojo|roja)',
            r'(que|cuanto).*(pasa|multa|sancion).*(exceso|velocidad|rapido|acelerar)',
            r'(que|cuanto).*(pasa|multa|sancion).*(estacion|parar).*(prohibido|mal|donde no)',
            r'(que|cuanto).*(pasa|multa|sancion).*(alcohol|ebrio|borracho|tomado)',
            r'(que|cuanto).*(pasa|multa|sancion).*(contramano|sentido contrario|marcha atras)',
            r'(que|cuanto).*(pasa|multa|sancion).*(sin|no).*(luces|frenos|cinturon)',
            r'(que|cuanto).*(pasa|multa|sancion).*(usar|celular|telefono).*(conducir|manejar)',

            # === ACCIDENTES ===
            r'(que|como).*(hago|hacer|debo|proceder).*(accidente|choque|atropello)',
            r'(quien|como).*(paga|responsable|culpable).*(accidente|daños|victimas)',
            r'(que|cuales).*(derechos|obligaciones|responsabilidades).*(accidente|choque)',
            r'(cubre|que cubre|cuanto cubre).*(seguro|poliza|soat).*(accidente|daños)',

            # === AUTORIDADES Y PROCEDIMIENTOS ===
            r'(que|cuanto|cuantos).*(dias|tiempo).*(quitan|retienen|decomisan).*(vehiculo|auto|licencia)',
            r'(como|que).*(hago|hacer|debo|proceder).*(paro|detuvo|freno).*(policia|transito)',
            r'(como|puedo).*(evitar|rechazar|denunciar).*(coima|soborno|mordida)',
            r'(donde|como).*(pago|cancelo|abono).*(multa|infraccion|sancion)',
            r'(puedo|se puede|como).*(apelar|reclamar|impugnar).*(multa|sancion|infraccion)',

            # === TRÁMITES Y CONSULTAS ===
            r'(como|donde|que necesito).*(sacar|renovar|obtener).*(licencia|brevet|permiso)',
            r'(como|donde|que necesito).*(transferir|traspasar|cambiar).*(auto|vehiculo)',
            r'(cuanto|cual|cada cuanto).*(revision|inspeccion).*(tecnica|vehicular)',
            r'(cuanto|como|donde).*(pagar|pago).*(impuesto|impositivo).*(auto|vehiculo)',
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
            "arreglar con el policia": 4, "mordida": 4, "colaboracion": 4,

            # === DOCUMENTACIÓN ===
            "no tengo licencia": 4,
            "no traje licencia": 4,
            "me olvide licencia": 4,
            "deje licencia": 4,
            "no tengo papeles": 4,
            "no andan mis papeles": 4,
            "no encuentro papeles": 4,
            "se me perdio licencia": 4,
            "se me vencio la licencia": 4,
            "sin documentos": 4,
            "sin papeles": 4,

            # === INFRACCIONES ===
            "me pase luz roja": 4,
            "cruce en rojo": 4,
            "pase semaforo": 4,
            "iba rapido": 3,
            "exceso velocidad": 4,
            "muy rapido": 3,
            "limite velocidad": 4,
            "estacionado mal": 4,
            "mal estacionado": 4,
            "lugar prohibido": 3,
            "alcoholemia": 4,
            "control alcoholemia": 4,
            "prueba alcoholemia": 4,
            "tome bebidas": 3,
            "estaba tomado": 4,
            "maneje borracho": 4,
            "contra el transito": 4,
            "sentido contrario": 4,
            "contramano": 4,
            "marcha atras": 3,
            "reversa prohibida": 3,
            "escape ruidoso": 3,
            "luces mal": 3,
            "sin luces": 4,
            "usando celular": 4,
            "haciendo maniobras": 3,
            "pasando doble linea": 4,
            "sin cinturon": 4,
            "pasajeros parados": 3,
            "exceso pasajeros": 3,
            "mucha carga": 3,

            # === ACCIDENTES ===
            "tuve accidente": 4,
            "me accidente": 4,
            "choque vehiculo": 4,
            "choque auto": 4,
            "me chocaron": 4,
            "atropelle": 4,
            "me atropellaron": 4,
            "volcadura": 4,
            "me volque": 4,
            "daños vehiculo": 3,
            "daños materiales": 3,
            "lesiones": 4,
            "heridos": 4,
            "victimas": 4,
            "seguro cobertura": 3,
            "responsabilidad": 3,
            "culpa accidente": 4,

            # === SITUACIONES CON AUTORIDADES ===
            "me paro policia": 4,
            "me paro transito": 4,
            "me han parado": 4,
            "me han detenido": 4,
            "acaban de pararme": 4,
            "me acaban de agarrar": 4,
            "apenas me pararon": 4,
            "recien me pararon": 4,
            "me multaron": 4,
            "me pusieron multa": 4,
            "me quieren multar": 4,
            "me van a multar": 4,
            "me sancionaron": 4,
            "me infraccionaron": 4,
            "me quitaron licencia": 4,
            "me retuvieron licencia": 4,
            "se llevaron mi auto": 4,
            "me decomisaron": 4,
            "me arrestaron": 4,
            "me llevaron detenido": 4,
            "me pidieron coima": 4,
            "me quisieron coimear": 4,
            "quisieron arreglar": 4,
            "sin boleta": 4,

            # === TRÁMITES ===
            "sacar licencia": 3,
            "renovar licencia": 3,
            "licencia vencida": 3,
            "transferencia vehiculo": 3,
            "cambiar propietario": 3,
            "inscribir auto": 3,
            "placa nueva": 3,
            "cambio placa": 3,
            "revision tecnica": 3,
            "inspeccion vehicular": 3,
            "pagar impuestos": 3,
        }
        
        # Modismos comunes para normalizar (especializados para Bolivia)
        self.modismos = {
            # Términos para policía
            "verde": "policia",
            "caminero": "policia",
            "transistero": "policia",
            "paco": "policia",
            "transito": "policia",
            "vigilante": "policia",
            "agente": "policia",
            "uniformado": "policia",
            
            # Términos para vehículos
            "trufi": "vehiculo",
            "micro": "vehiculo",
            "minibus": "vehiculo",
            "surubi": "vehiculo",
            "movilidad": "auto",
            "taxi trufi": "vehiculo",
            "vagoneta": "auto",
            "jeep": "auto",
            "camioneta": "vehiculo",
            "motorizado": "vehiculo",
            "chatarra": "vehiculo viejo",
            "nave": "auto",
            "moto": "motocicleta",
            "cuadratrack": "motocicleta",
            
            # Términos para documentos
            "carton": "licencia",
            "chapa": "placa",
            "roseta": "itv",
            "librillo": "licencia",
            "papeles": "documentos",
            "permiso": "licencia",
            "itv": "inspeccion tecnica",
            "b-sisa": "documento vehicular",
            "ruat": "documento vehicular",
            "matricula": "placa",
            
            # Términos para control
            "tranca": "control",
            "reten": "control",
            "punto de control": "control",
            "puesto": "control",
            "caseta": "control",
            "barrera": "control",
            "peaje": "control",
            "alcabala": "control",
            
            # Términos para situaciones
            "chaparon": "detuvieron",
            "agarraron": "detuvieron",
            "levantaron": "multaron",
            "pillaron": "detuvieron",
            "pescaron": "detuvieron",
            "atraparon": "detuvieron",
            "cortaron": "detuvieron",
            "pararon": "detuvieron",
            "notificaron": "multaron",
            "sancionaron": "multaron",
            "sacaron parte": "multaron",
            
            # Términos para soborno
            "coimear": "sobornar",
            "coima": "soborno",
            "mordida": "soborno",
            "refresco": "soborno",
            "colaboracion": "soborno",
            "gastos": "soborno",
            "arreglar": "sobornar",
            "ayudar": "sobornar",
            "pacto": "soborno",
            "arreglo": "soborno",
            "por lo bajo": "soborno",
            "por debajo": "soborno",
            "sin boleta": "soborno",
            "ayudita": "soborno",
            "propina": "soborno",
            "para el cafecito": "soborno",
            "para la gaseosa": "soborno",
            "negociar": "sobornar",
            "sin papeles": "sobornar",
            
            # Más términos para vehículos
            "taxi trufi": "vehiculo",
            "vagoneta": "auto",
            "jeep": "auto",
            "camioneta": "vehiculo",
            "motorizado": "vehiculo",
            "chatarra": "vehiculo viejo",
            "nave": "auto",
            "moto": "motocicleta",
            "cuadratrack": "motocicleta",
            
            # Más términos para documentos
            "librillo": "licencia",
            "papeles": "documentos",
            "permiso": "licencia",
            "itv": "inspeccion tecnica",
            "b-sisa": "documento vehicular",
            "ruat": "documento vehicular",
            "matricula": "placa",
            
            # Más términos para control
            "puesto": "control",
            "caseta": "control",
            "barrera": "control",
            "peaje": "control",
            "alcabala": "control",
            
            # Más términos para situaciones
            "pillaron": "detuvieron",
            "pescaron": "detuvieron",
            "atraparon": "detuvieron",
            "cortaron": "detuvieron",
            "pararon": "detuvieron",
            "notificaron": "multaron",
            "sancionaron": "multaron",
            "infraccion": "multa",
            "sacaron parte": "multaron",
            
            # Más términos para soborno
            "pacto": "soborno",
            "arreglo": "soborno",
            "por lo bajo": "soborno",
            "por debajo": "soborno",
            "sin boleta": "soborno",
            "ayudita": "soborno",
            "propina": "soborno",
            "para el cafecito": "soborno",
            "para la gaseosa": "soborno",
            "negociar": "sobornar",
            "sin papeles": "sobornar",
            
            # Términos para infracciones
            "pasarse": "infraccion",
            "cruzarse": "infraccion",
            "meterse": "infraccion",
            "colarse": "infraccion",
            "excederse": "exceso velocidad",
            "estar cebado": "exceso velocidad",
            "ir a fondo": "exceso velocidad",
            "ir quemando": "exceso velocidad",
            "pasarse": "infraccion",
            "cruzarse": "infraccion",
            "meterse": "infraccion",
            "colarse": "infraccion",
            "excederse": "exceso velocidad",
            "estar cebado": "exceso velocidad",
            "ir a fondo": "exceso velocidad",
            "ir quemando": "exceso velocidad",
            
            # Términos para accidentes
            "choque": "accidente",
            "topón": "accidente menor",
            "raspón": "accidente menor",
            "encontronazo": "accidente",
            "volcadura": "accidente",
            "estrellarse": "accidente",
            "choque": "accidente",
            "topón": "accidente menor",
            "raspón": "accidente menor",
            "encontronazo": "accidente",
            "volcadura": "accidente",
            "estrellarse": "accidente",
            
            # Términos para estado del conductor
            "chupado": "ebrio",
            "cocido": "ebrio",
            "picado": "ebrio",
            "tomado": "ebrio",
            "volteado": "ebrio",

            # === DOCUMENTACIÓN ===
            "no traer": "no portar",
            "no traje": "no portar",
            "me olvide": "no portar",
            "olvide": "no portar",
            "deje": "no portar",
            "se quedo": "no portar",
            "no tengo": "no portar",
            "no cargo": "no portar",
            "no llevo": "no portar",
            "no ando con": "no portar",
            "sin": "no portar",
            "no traigo": "no portar",
            "me faltan": "no portar",
            "faltan": "no portar",
            "perdi": "no portar",
            "extravie": "no portar",

            "papeles": "documentos",
            "papelitos": "documentos",
            "carton": "licencia",
            "cartoncito": "licencia",
            "permiso": "licencia",
            "credencial": "licencia",
            "identificacion": "licencia",
            "brevet": "licencia",
            "carnet de conducir": "licencia",
            "registro": "licencia",
            "pase": "licencia",
            "seguro": "soat",
            "chapa": "placa",
            "numero": "placa",
            "patente": "placa",
            "matricula": "placa",

            # === INFRACCIONES COMUNES ===
            "pase": "cruzar",
            "me cruce": "cruzar",
            "me pase": "cruzar",
            "no respete": "infringir",
            "no hice caso": "infringir",
            "rompi": "infringir",
            "viole": "infringir",
            "salte": "infringir",
            "ignore": "infringir",
            "me meti": "invasión",

            "luz roja": "semáforo en rojo",
            "luz en rojo": "semáforo en rojo",
            "rojo": "semáforo en rojo",
            "semaforo": "semáforo",

            "rapido": "exceso de velocidad",
            "veloz": "exceso de velocidad",
            "corriendo": "exceso de velocidad",
            "acelerado": "exceso de velocidad",
            "volando": "exceso de velocidad",
            "a toda": "exceso de velocidad",
            "a fondo": "exceso de velocidad",
            "quemando": "exceso de velocidad", 
            "al palo": "exceso de velocidad",

            "mal estacionado": "estacionamiento prohibido",
            "en doble fila": "estacionamiento prohibido",
            "donde no debia": "estacionamiento prohibido",
            "en zona prohibida": "estacionamiento prohibido",
            "donde no se puede": "estacionamiento prohibido",

            "borracho": "estado de ebriedad",
            "tomado": "estado de ebriedad",
            "bebido": "estado de ebriedad",
            "chupado": "estado de ebriedad",
            "con tragos": "estado de ebriedad",
            "con copas": "estado de ebriedad",
            "picado": "estado de ebriedad",
            "con alcohol": "estado de ebriedad",

            "sentido contrario": "contra el sentido de circulación",
            "contramano": "contra el sentido de circulación",
            "en contra": "contra el sentido de circulación",
            "direccion prohibida": "contra el sentido de circulación",
            "en reversa": "marcha atrás prohibida",
            "retrocediendo": "marcha atrás prohibida",

            # === ACCIDENTES ===
            "choque": "accidente",
            "choqué": "accidente",
            "me chocaron": "accidente",
            "colisión": "accidente",
            "impacto": "accidente", 
            "golpeé": "accidente",
            "golpeado": "accidente",
            "topé": "accidente",
            "topón": "accidente",
            "raspón": "accidente",
            "rayón": "accidente",
            "atropellé": "atropello",
            "atropellado": "atropello",
            "pisé": "atropello",
            "me llevé": "atropello",
            "volcadura": "volcamiento",
            "volcada": "volcamiento",
            "vuelco": "volcamiento",
            "di vuelta": "volcamiento",
            "me volqué": "volcamiento",

            # === AUTORIDADES ===
            "me paro": "me detuvo",
            "me agarraron": "me detuvieron",
            "me atraparon": "me detuvieron",
            "me frenaron": "me detuvieron",
            "me pesco": "me detuvo",
            "me encontraron": "me detuvieron",
            "me pillaron": "me detuvieron",
            "me sacaron": "me multaron",
            "me pusieron": "me multaron",
            "me dieron": "me multaron",
            "me cobraron": "me multaron",
            "me llevaron": "me arrestaron",
            "me quitaron": "decomisaron",
            "me secuestraron": "decomisaron",
            "me sacaron el auto": "decomisaron",
            "me lo llevaron al auto": "decomisaron",
            "me anotaron": "me multaron",
            "me ficharon": "me multaron",
            "me levantaron": "me multaron",

            "verde": "policía de tránsito",
            "caminero": "policía de tránsito",
            "transistero": "policía de tránsito",
            "paco": "policía",
            "rati": "policía",
            "vigilante": "policía",
            "agente": "policía",
            "autoridad": "policía",

            # === VEHÍCULOS ===
            "auto": "vehículo",
            "carro": "vehículo", 
            "coche": "vehículo",
            "movilidad": "vehículo",
            "nave": "vehículo",
            "cacharro": "vehículo",
            "maquina": "vehículo",
            "moto": "motocicleta",
            "motoca": "motocicleta",
            "motorizado": "vehículo motorizado",
            "cuadratrack": "vehículo todo terreno",

            # === CONSECUENCIAS ===
            "arrestaron": "detención",
            "me encerraron": "detención",
            "calabozo": "detención",
            "cárcel": "detención",
            "preso": "detención",
            "detenido": "detención",
            "encerrado": "detención",

            "me llevaron el auto": "retención del vehículo",
            "me quitaron el auto": "retención del vehículo",
            "me dejaron a pie": "retención del vehículo",
            "se llevaron mi": "retención del vehículo",
            "decomisaron mi": "retención del vehículo",
            "me secuestraron el": "retención del vehículo",

            "cuanto pago": "multa",
            "que me cobran": "multa",
            "tengo que pagar": "multa",
            "me multaron": "multa",
            "boleta": "multa",
            "ticket": "multa",
            "sancion": "multa",
            "castigo": "sanción",
            "infraccion": "infracción",

            # === SOBORNO ===
            "coima": "soborno",
            "mordida": "soborno",
            "refresco": "soborno",
            "gastos": "soborno",
            "colaboracion": "soborno",
            "ayudita": "soborno",
            "propina": "soborno",
            "para el cafecito": "soborno",
            "para la gaseosa": "soborno",
            "arreglar por fuera": "soborno",
            "arreglar sin papeles": "soborno",
            "sin boleta": "soborno",
            "sin recibo": "soborno",
            "por lo bajo": "soborno",
            "por debajo": "soborno",

        }
        
        # Umbral de puntuación para considerar contexto de tránsito
        self.umbral_puntaje = 0
    
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
        
        texto = texto.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        texto = texto.replace("ñ", "n").replace("ü", "u")
        
        # Normalizar múltiples espacios
        texto = re.sub(r'\s+', ' ', texto)
        
        # Quitar caracteres especiales excepto letras, números y espacios
        texto = re.sub(r'[^a-z0-9\s]', ' ', texto)

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
            # Buscar la palabra exacta
            if re.search(r'\b' + palabra + r'\b', texto):
                puntaje += peso
                detalles["palabras_positivas"].append((palabra, peso))
            # También buscar el plural simple (añadir 's')
            elif re.search(r'\b' + palabra + r's\b', texto):
                puntaje += peso
                detalles["palabras_positivas"].append((palabra + 's', peso))
        
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
        
        # saludos = ["hola", "buenos dias", "buenas tardes", "buen dia", "que tal", "como estas", 
        #    "saludos", "cuentame", "dime", "ayudame"]
        saludos = []
        
    
        if any(re.search(r'\b' + saludo + r'\b', texto) for saludo in saludos) and len(detalles["palabras_positivas"]) < 2:
            return True, -puntaje * 0.9  # Reducir casi todo el puntaje
        
        # Detectar contenido violento o inapropiado
        contenido_inapropiado = []
        # "matar", "asesinar", "herir", "lastimar", "violencia", "robar", 
        #                     "hackear", "ilegal", "fraude", "drogas", "arma"
        
        if any(re.search(r'\b' + termino + r'\b', texto) for termino in contenido_inapropiado):
            return True, -puntaje * 1.0  # Eliminar todo el puntaje
        
        # Detectar peticiones de tareas escolares
        tareas_escolares = ["tarea", "deberes", "trabajo escolar", "actividad", "investigacion",
                        "sociales", "historia", "geografia", "exposicion"]
        
        if any(re.search(r'\b' + tarea + r'\b', texto) for tarea in tareas_escolares):
            return True, -puntaje * 0.8  # Reducir significativamente el puntaje
            
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

            for patron in self.preguntas_basicas_transito:
                if re.search(patron, pregunta_normalizada, re.IGNORECASE):
                    print(f"Pregunta básica de tránsito detectada: '{pregunta_normalizada}'")
                    # Si es una pregunta básica, está automáticamente en contexto con alta confianza
                    return True, 10, 0.9, {
                        "puntajes": {"pregunta_basica_transito": 10},
                        "resultados": {
                            "puntaje_bruto": 10,
                            "puntaje_final": 10,
                            "umbral": self.umbral_puntaje,
                            "confianza": 0.9,
                            "decision": "en_contexto"
                        }
                    }

            if len(pregunta_normalizada.split()) < 4:
                print(f"Pregunta demasiado corta, considerada fuera de contexto: '{pregunta_normalizada}'")
                return False, 0, 0, {"error": "Pregunta demasiado corta"}
                
            # Calcular puntaje inicial
            puntaje, detalles = self.calcular_puntaje_texto(pregunta_normalizada)
            
            # Verificar falsos positivos
            es_falso_positivo, ajuste_falso_positivo = self.detectar_falsos_positivos(
                pregunta_normalizada, puntaje, detalles
            )
            
            if es_falso_positivo:
                puntaje += ajuste_falso_positivo
            
            puntaje_final = puntaje
            
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
                    "falso_positivo": ajuste_falso_positivo
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
            
            # Verificar si hay error en el diagnóstico
            if "error" in diagnostico:
                print(f"ERROR DE ANÁLISIS: {diagnostico['error']}")
                print("="*50)
                return
            
            # Mostrar desglose de puntajes (solo si existe la clave)
            if "puntajes" in diagnostico:
                print("DESGLOSE DE PUNTAJES:")
                for categoria, puntos in diagnostico["puntajes"].items():
                    if puntos != 0:
                        print(f"  - {categoria}: {puntos:.2f}")
            
            # Mostrar ajustes (solo si existe la clave)
            if "ajustes" in diagnostico and any(diagnostico["ajustes"].values()):
                print("\nAJUSTES APLICADOS:")
                for tipo_ajuste, valor in diagnostico["ajustes"].items():
                    if valor != 0:
                        print(f"  - {tipo_ajuste}: {valor:.2f}")
            
            # Mostrar términos encontrados (solo si existe la clave)
            if "terminos_encontrados" in diagnostico:
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
        
        # Ruta a la carpeta de documentos fuente
        self.ruta_documentos = self.BASE_DIR / 'data' 

        self.db_config = {
            'dbname': 'BDRodalex', 
            'user': 'postgres',           
            'password': 'clave123',    
            'host': 'localhost',
            'port': '5432'
        }

        # Inicializar la base de datos
        self.inicializar_db()

        # Inicializamos comprobando si ya existe la base de conocimiento guardada
        self.inicializar_modelo()

    def obtener_conexion_BaseDatos(self):
        """Establece una conexión a la base de datos PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Error al conectar a PostgreSQL: {e}")
            return None

    def inicializar_modelo(self):
        """
        Inicializa el modelo, verificando si los fragmentos ya existen en PostgreSQL.
        Si no existen, procesa el texto inicial.
        """
        try:
            # Comprobar si ya hay fragmentos en PostgreSQL
            conn = self.obtener_conexion_BaseDatos()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fragmentos_texto")
                count = cursor.fetchone()[0]
                conn.close()
                
                if count > 0:
                    print(f"Ya existen {count} fragmentos en PostgreSQL, reconstruyendo base de conocimiento...")
                    # Cargar fragmentos desde PostgreSQL y construir FAISS en memoria
                    return self._cargar_desde_postgresql()
            
            # Si no existen fragmentos en PostgreSQL, procesamos el texto desde cero
            return self._procesar_texto_inicial()
                
        except Exception as e:
            print(f"ERROR al inicializar el modelo jurídico: {e}")
            return False

    def inicializar_db(self):
        """Crea las tablas necesarias en PostgreSQL si no existen."""
        conn = self.obtener_conexion_BaseDatos()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Crear tabla para fragmentos de texto
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fragmentos_texto (
                    id SERIAL PRIMARY KEY,
                    contenido TEXT NOT NULL,
                    embedding BYTEA,
                    metadata JSONB,
                    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            print("Base de datos inicializada correctamente")
            return True
        except Exception as e:
            print(f"Error al inicializar la base de datos: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def _cargar_desde_postgresql(self):
        """
        Carga los fragmentos desde PostgreSQL y reconstruye el índice FAISS en memoria
        usando el método from_embeddings que es más seguro.
        """
        try:
            conn = self.obtener_conexion_BaseDatos()
            if not conn:
                return False
                    
            cursor = conn.cursor()
            cursor.execute("SELECT id, contenido, embedding, metadata FROM fragmentos_texto")
            resultados = cursor.fetchall()
            conn.close()
            
            if not resultados:
                print("No se encontraron fragmentos en PostgreSQL")
                return False
                    
            # Preparar documentos y embeddings
            documentos = []
            embeddings_list = []
            metadatas = []
            texts = []
            
            for id_frag, contenido, embedding_bytes, metadata_json in resultados:
                # Deserializar el embedding
                embedding = pickle.loads(embedding_bytes) if embedding_bytes else None
                
                if embedding is not None:
                    # Manejar el metadata adecuadamente según su tipo
                    metadata = {}
                    if metadata_json is not None:
                        if isinstance(metadata_json, str):
                            metadata = json.loads(metadata_json)
                        elif isinstance(metadata_json, dict):
                            metadata = metadata_json
                        else:
                            try:
                                metadata = json.loads(metadata_json)
                            except:
                                print(f"No se pudo parsear metadata para fragmento {id_frag}")
                    
                    # Guardar el texto y metadatos
                    texts.append(contenido)
                    metadatas.append(metadata)
                    
                    # Guardar el embedding
                    embeddings_list.append(embedding)
            
            print(f"Cargados {len(texts)} fragmentos desde PostgreSQL")
            
            # Crear objeto de embeddings
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            if embeddings_list and texts:
                # Crear la base de conocimiento utilizando from_embeddings
                # Este método es más seguro y maneja la creación de docstore correctamente
                from langchain_community.vectorstores import FAISS
                self.base_conocimiento = FAISS.from_embeddings(
                    text_embeddings=list(zip(texts, embeddings_list)),
                    embedding=vectores,
                    metadatas=metadatas if metadatas else None
                )
                
                print("Base de conocimiento reconstruida exitosamente desde PostgreSQL")
                
                # Verificar que la base de conocimiento tiene el método as_retriever
                if hasattr(self.base_conocimiento, 'as_retriever'):
                    print("Verificado: base de conocimiento tiene método as_retriever")
                    self._configurar_qa("basico")
                    return True
                else:
                    print("ERROR: base de conocimiento no tiene método as_retriever")
                    return False
            else:
                print("No se pudieron reconstruir los embeddings")
                return False
                    
        except Exception as e:
            print(f"ERROR al cargar desde PostgreSQL: {e}")
            import traceback
            traceback.print_exc()  # Imprime el stack trace completo para mejor diagnóstico
            return False

    def _configurar_qa(self,tipo_modelo):
        """
        Configura el modelo de preguntas y respuestas basado en la base de conocimiento.
        """
        try:
            # Configurar el modelo de LLM (ChatOpenAI)
            # if(tipo_modelo == "basico"):
            #     llm = ChatOpenAI(
            #     api_key=CLAVE_API,
            #     model_name="gpt-4o",
            #     temperature=0.3
            # )
            # elif(tipo_modelo == "avanzado"):
            #     llm = ChatOpenAI(
            #     api_key=CLAVE_API,
            #     model_name="gpt-4",
            #     temperature=0.3
            # )

            if not hasattr(self, 'llm_actual') or self.llm_actual != tipo_modelo:
                if tipo_modelo == "basico":
                    self.llm = ChatOpenAI(api_key=CLAVE_API, model_name="gpt-4-turbo", temperature=0.2)
                elif tipo_modelo == "avanzado":
                    self.llm = ChatOpenAI(api_key=CLAVE_API, model_name="gpt-4o", temperature=0.3)
                self.llm_actual = tipo_modelo
            
            
            # Verificar que self.base_conocimiento existe y es del tipo correcto
            if not hasattr(self, 'base_conocimiento') or self.base_conocimiento is None:
                print("Error: La base de conocimiento no está inicializada")
                return False
                
            # Verificar que as_retriever es un método disponible
            if not hasattr(self.base_conocimiento, 'as_retriever'):
                print(f"Error: La base de conocimiento no tiene método as_retriever. Tipo: {type(self.base_conocimiento)}")
                return False
            
            # Crear la cadena de QA con recuperación
            # self.qa = RetrievalQA.from_chain_type(
            #     llm=llm,
            #     chain_type="stuff",
            #     retriever=self.base_conocimiento.as_retriever(
            #         search_type="similarity",
            #         search_kwargs={"k": 10} 
            #     )
            # )

            # self.qa = RetrievalQA.from_chain_type(
            #     llm=self.llm,
            #     chain_type="stuff",
            #     retriever=self.base_conocimiento.as_retriever(
            #         search_type="mmr",  
            #         search_kwargs={"k": 10, "fetch_k": 15}  
            #     )
            # )

            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                
                chain_type="stuff",
                
                retriever=self.base_conocimiento.as_retriever(
                    # MMR equilibra relevancia y diversidad
                    
                    search_kwargs={
                        "k": 10,
                        
                        "fetch_k": 20,
                        
                        "lambda_mult": 0.8
                    }
                ),
                
                return_source_documents=True
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
            print("Creando base de conocimiento desde el documento fuente...")
            
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
            


            divisor_texto = RecursiveCharacterTextSplitter(
                # - Facilita mantener el contexto completo de un artículo sin fragmentarlo
                chunk_size=2000,
                
                # chunk_overlap=500: Overlap sustancial para preservar contexto entre fragmentos
                # - Evita perder información en los límites entre artículos relacionados
                # - Mantiene conexiones entre artículos y sus referencias cruzadas
                # - Proporciona suficiente contexto en casos como las infracciones donde un artículo menciona a otro
                chunk_overlap=500,
                
                # Separadores específicos para la estructura de la normativa de tránsito boliviana:
                separators=[
                    # Secciones principales
                    "DECRETO SUPREMO", "Decreto_Supremo_", 
                    "TÍTULO ", "Título ", "TITULO ", "Titulo ", 
                    "CAPÍTULO ", "Capítulo ", "CAPITULO ", "Capitulo ", 
                    
                    # Artículos
                    "\nArtículo ", "\nARTÍCULO ", "\nArticulo ", "\nARTICULO ", 
                    "\nArt. ", "\nART. ", 
                    
                    # Especial énfasis en incisos de infracciones
                    # Patrones más específicos para las infracciones numeradas
                    "\n +[0-9]+\. ", 
                    "\n[0-9]+\. Por ", 
                    "\n  [0-9]+\. Por ", 
                    "\n [0-9]+\. Por ", 
                    "\n  [a-z]\) ", "\n [a-z]\) ", "\n[a-z]\) ", 
                    "\n  [ivxIVX]+\. ", "\n [ivxIVX]+\. ", 
                    
                    # Patrones para sanciones y multas
                    "\n.*multa de [A-Z]+", 
                    "\n.*sanción de [A-Z]+", 
                    "con [A-Z]+ PESOS BOLIVIANOS", 
                    "inhabilitación", 
                    "suspensión de", 
                    "arresto de", 
                    
                    # Patrones para procedimientos legales
                    "El procedimiento", 
                    "La autoridad", 
                    "deberá presentar", 
                    "podrá apelar", 
                    "en un plazo de", 
                    "bajo responsabilidad de", 
                    
                    # Patrones para derechos y obligaciones
                    "Todo conductor tiene derecho a", 
                    "Es obligación del conductor", 
                    "Es obligación del peatón", 
                    "Se prohíbe", 
                    "Queda prohibido", 
                    "Es deber de", 
                    
                    # Patrones para definiciones legales:
                    "Se entiende por", 
                    "Se define como", 
                    "Para los efectos de", 
                    "Para los fines del presente", 
                    "Se considera", 
                    
                    # Patrones específicos para vehículos y circulación:
                    "Los vehículos de transporte público", 
                    "La velocidad máxima", 
                    "En las intersecciones", 
                    "Derecho de vía", 
                    "señalización", 
                    "tránsito", 
                    
                    # Términos clave del contador de palabras
                    # Documentación y requisitos legales
                    "licencia", "SOAT", "identificación", "placas", "documentos", "brevet", "autorización",
                    
                    # Infracciones y sanciones específicas
                    "multa", "infracción", "bolivianos", "velocidad", "vidrios oscurecidos", 
                    "polarizados", "alcohol", "alcoholemia", "embriaguez",
                    
                    # Autoridades y proceso de control
                    "policía", "control", "autoridad", "inspección", "retención", "secuestro",
                    "ATT", "SEGIP", "superintendencia",
                    
                    # Actores viales principales
                    "conductor", "conductores", "peatones", "pasajeros", "peatón", "choferes",
                    
                    # Accidentes y seguridad
                    "accidentes", "seguridad", "vial", "muerte", "heridos", "víctimas",
                    "lesiones", "daños", "indemnización", "SOAT", "cobertura",
                    
                    # Términos específicos de vehículos
                    "circulación", "tránsito", "vehículos", "vehículo", "transporte", "servicio público",
                    
                    # NUEVOS TÉRMINOS AÑADIDOS
                    
                    # Categorías de infracciones
                    "infracción leve", "infracción grave", "infracción gravísima",
                    "primera infracción", "reincidencia", 
                    
                    # Sanciones específicas 
                    "inhabilitación por", "suspensión de licencia", "suspensión definitiva",
                    "alcoholemia positiva", "estado de embriaguez", "conducción peligrosa",
                    "multa de [0-9]+", "sanción de arresto", "sanción económica",
                    "decomiso del", "retención del", "secuestro del vehículo",
                    
                    # Documentación y requisitos extendidos
                    "licencia de conducir", "placa de control", "roseta de inspección", 
                    "identificación vehicular", "cédula de identidad", "documentos obligatorios",
                    "vigencia", "renovación", "caducidad", "trámite", "solicitud", "certificado",
                    
                    # Autoridades adicionales
                    "policía caminera", "policía de tránsito", "comando", "jefatura", 
                    "dirección", "administradora", "órgano ejecutivo", "juzgado",
                    
                    # Accidentes y términos relacionados
                    "accidente de tránsito", "colisión", "atropello", "choque",
                    "daños materiales", "lesiones graves", "lesiones leves",
                    "muerte instantánea", "víctimas fatales", "personas heridas",
                    "indemnización por", "cobertura del seguro", "pago de gastos",
                    "asegurado", "damnificado", "póliza",
                    
                    # Vehículos - características y condiciones
                    "vidrios polarizados", "luces reglamentarias", "frenos deficientes", 
                    "cinturón de seguridad", "límite de velocidad", "exceso de velocidad", 
                    "carga peligrosa", "sobrecarga", "peso máximo", "dimensiones",
                    
                    # Procedimientos específicos
                    "procedimiento de fiscalización", "audiencia", "apelación", "recurso",
                    "prueba de alcoholemia", "test de drogas", "inspección técnica",
                    "peritaje", "declaración jurada", "dictamen", "sentencia", "resolución",
                    
                    # Infraestructura vial
                    "carretera", "autopista", "vía pública", "calzada", "intersección",
                    "semáforo", "señal de tránsito", "paso peatonal", "cruces", "puentes",
                    
                    # Terminología específica boliviana
                    "CRPVA", "SEGELIC", "ED3", "FISO", "RUI", "APS", "SRUI",
                    
                    # Resto de separadores
                    "\nDISPOSICIONES ", "\nDisposiciones ",
                    "\n\n", "\n", ". ", ", ", " ", ""
                ],
                # Función de medición estándar (conteo de caracteres)
                length_function=len,
                
                # is_separator_regex=True: Habilita expresiones regulares en los separadores
                # - CRUCIAL para reconocer patrones numéricos (1., 2., 3.) e incisos (a), b), c))
                # - Permite usar los patrones [0-9]+, [a-z], [ivxIVX]+ para capturar cualquier número o letra
                # - Sin esto, no se detectarían correctamente las listas numeradas de infracciones
                # - Facilita el manejo de diferentes niveles de indentación (\n  [0-9]+\., \n [0-9]+\.)
                # - Mejora significativamente la segmentación de artículos con múltiples incisos
                is_separator_regex=True
            )



            fragmentos = divisor_texto.split_documents(textos)
            print(f"Se han creado {len(fragmentos)} fragmentos de texto para la base de conocimiento")
            
            # Crear los vectores de embeddings
            vectores = OpenAIEmbeddings(
                api_key=CLAVE_API,
                model="text-embedding-ada-002"
            )
            
            # Crear la base de conocimiento vectorial en memoria
            self.base_conocimiento = FAISS.from_documents(fragmentos, vectores)
            
            # Guardar fragmentos en PostgreSQL
            try:
                conn = self.obtener_conexion_BaseDatos()
                if conn:
                    cursor = conn.cursor()
                    
                    # Limpiar tabla existente si ya hay datos
                    cursor.execute("TRUNCATE TABLE fragmentos_texto RESTART IDENTITY")
                    print("🔄 Tabla fragmentos_texto limpiada, insertando nuevos fragmentos...")
                    
                    # Preparar datos para inserción masiva
                    datos = []
                    for fragmento in fragmentos:
                        # Generar embedding para el fragmento
                        embedding = vectores.embed_query(fragmento.page_content)
                        embedding_bytes = pickle.dumps(embedding)
                        
                        datos.append((
                            fragmento.page_content,
                            psycopg2.Binary(embedding_bytes),
                            json.dumps(fragmento.metadata)
                        ))
                    
                    print(f"🔄 Preparando {len(datos)} fragmentos para inserción en PostgreSQL...")
                    
                    # Insertar todos los fragmentos de una vez
                    execute_values(
                        cursor,
                        "INSERT INTO fragmentos_texto (contenido, embedding, metadata) VALUES %s",
                        datos,
                        template="(%s, %s, %s)"
                    )
                                        
                    # Verificar que se guardaron correctamente
                    cursor.execute("SELECT COUNT(*) FROM fragmentos_texto")
                    count = cursor.fetchone()[0]
                    
                    conn.commit()
                    conn.close()
                    
                    print(f"✅ {count} fragmentos guardados exitosamente en PostgreSQL")
                else:
                    print("❌ No se pudo conectar a PostgreSQL para guardar fragmentos")
            except Exception as e:
                print(f"❌ Error al guardar fragmentos en PostgreSQL: {e}")
                if 'conn' in locals() and conn:
                    conn.rollback()
                    conn.close()
        except Exception as e:
            print(f"ERROR al procesar texto inicial: {e}")
            return False
        
        self._configurar_qa("basico")
        return True

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
        # Configurar el modelo QA y verificar que se haya configurado correctamente
        if not self._configurar_qa(tipo_modelo):
            print("Error: No se pudo configurar el modelo QA")
            return {
                "fueraDeContexto": True,
                "respuestaDirecta": "El sistema no está inicializado correctamente."
            }

        if not self.qa:
            print("Error: Sistema no inicializado")
            return {
                "fueraDeContexto": True,
                "respuestaDirecta": "El sistema no está inicializado correctamente."
            }
        
        try:
            # Agregar esta línea ↓
            self.verificador.procesar_pregunta_para_desarrollo(pregunta)
            
            # Verificar contexto
            resultado_contexto = self.verificar_contexto(pregunta)
            esta_en_contexto = resultado_contexto[0] if isinstance(resultado_contexto, tuple) else resultado_contexto
            print(f"¿Está en contexto?: {resultado_contexto}")

            if not esta_en_contexto:
                return {
                    "fueraDeContexto": True,
                    "respuestaDirecta": "Como tu asistente legal, no tengo esa información. Puedo ayudarte con temas de (codigo de transito) en Bolivia."
                }
            
            # Instrucciones para respuesta en dos niveles: amigo rápido + legal profundo

            # print(f"Generando respuesta para: {historial_conversacion}")

            # prompt = f"""
            # ### CONTEXTO: 
            # Eres un abogado boliviano experto en código de tránsito, especializado en defender los derechos de conductores frente a situaciones de control policial, infracciones y posibles abusos. Debes proporcionar asesoramiento legal preciso, práctico y específico.

            # ### INSTRUCCIONES CRÍTICAS:
            # - SIEMPRE especifica los artículos exactos con su número, sección, subsección, inciso o numeral
            # - SIEMPRE incluye montos exactos de multas en Bolivianos PARA CADA ARTÍCULO mencionado, SIN EXCEPCIÓN, indicando tanto el valor numérico como escrito
            # - SIEMPRE ante cualquier situación, menciona cómo actuar ante intentos de presión o soborno
            # - SIEMPRE incluye consejos para documentar la situación (grabar, testigos, etc.)
            # - NUNCA des respuestas vagas o genéricas
            # - SIEMPRE presenta MÍNIMO 3 y MÁXIMO 4 situaciones/artículos legales relacionados con la consulta inicial
            # - SIEMPRE usa lenguaje SENCILLO y DIRECTO, como si hablaras con un amigo
            # - SIEMPRE especifica el tiempo de retención del vehículo o licencia cuando aplique
            # - SIEMPRE explica las CONSECUENCIAS PRÁCTICAS para el cliente: montos exactos a pagar, procesos completos, tiempos de espera, y requisitos de documentación

            # ### INFORMACIÓN OBLIGATORIA PARA INCLUIR:
            # 1. Para TODA consulta sobre infracciones:
            # - Artículo exacto infringido con número, sección y numeral
            # - Monto específico de la multa en Bs (SIEMPRE expresado en números y letras, sin importar el tipo de infracción)
            # - Si amerita o no retención de vehículo/licencia y por CUÁNTOS DÍAS exactamente
            # - Categoría de la infracción (leve, grave, muy grave)
            # - Procedimiento correcto de emisión de boleta y pasos posteriores
            # - SIEMPRE explicar claramente TODAS las consecuencias para el conductor: cuánto deberá pagar exactamente, qué proceso deberá seguir, cuánto tiempo tomará, y qué documentos necesitará presentar
            # - SIEMPRE especificar los plazos legales para pagar multas, recuperar vehículos/licencias, o presentar apelaciones

            # 2. Para TODA situación de control policial:
            # - Derechos específicos del conductor (mínimo 3)
            # - Documentos que legalmente pueden exigirte (listar todos)
            # - Procedimiento legal que debe seguir el policía paso por paso
            # - Cómo documentar discretamente (grabar, anotar placa, etc.)
            # - Frases específicas para afirmar derechos sin confrontación

            # 3. Para TODA situación donde pueda existir soborno:
            # - Mencionar que es delito de cohecho según el artículo específico del Código Penal
            # - Indicar monto legal de la multa para comparación
            # - Recomendar grabar discretamente la interacción
            # - Proporcionar 2-3 frases exactas para rechazar soborno sin provocar confrontación
            # - Vías legales específicas para denunciar posteriormente (nombres de instituciones y procedimiento)

            # ### ESTRUCTURA DE RESPUESTA:
            # - Si es inicio de conversación (historial vacío):
            # * ANALIZAR la consulta y presentar SIEMPRE entre 3-4 situaciones/artículos legales relevantes a la situacion
            # * MÍNIMO 3 situaciones/artículos legales SIEMPRE (incluso si una parece más obvia)
            # * MÁXIMO 4 situaciones/artículos legales (solo cuando el análisis indique cuarta situación relevante)
            # * Para cada situación/artículo dar una explicación sencilla
            # * Incluir palabras clave diferenciadoras muy claras en el campo "diferencias"
            # * Si la pregunta es ambigua, presentar las opciones más probables e invitar a aclarar
            # * EXPLICAR CADA SITUACIÓN COMPLETAMENTE en la respuesta, incluyendo para cada artículo mencionado:
            #     - Qué establece exactamente el artículo (texto legal simplificado)
            #     - Monto específico de la multa (en número y letras)
            #     - Si amerita retención de vehículo/licencia y por cuántos días exactamente
            #     - Procedimiento correcto que debe seguir el policía
            #     - Consejos para el conductor
            #     - Qué hacer ante intentos de presión o soborno
            #     - Derechos específicos que puede invocar

            # - Si es continuación de conversación:
            # * REVISAR lo que se mencionó anteriormente en el historial de conversación, especialmente las situaciones y artículos legales que ya se discutieron
            # * RECORDAR y REFERIRSE ESPECÍFICAMENTE a las situaciones/artículos mencionados anteriormente
            # * Ya no mostrar múltiples opciones
            # * Proporcionar una respuesta legal precisa específica a la situación elegida o mencionada previamente
            # * Primer párrafo: Explicación legal específica (artículos y multas)
            # * Segundo párrafo: Acciones prácticas inmediatas (5-7 puntos concretos)
            # * Tercer párrafo: Consejos para situaciones de posible abuso/soborno
            # * Si el usuario hace una pregunta sobre una "situación anterior" o compara con algo "anterior", SIEMPRE buscar en el historial a qué situación específica se refiere y responder en ese contexto
            # * Si el usuario menciona un nuevo tema relacionado con un tema anterior, ESTABLECER EXPLÍCITAMENTE la relación: "Sobre tu pregunta anterior de [tema] y esta nueva consulta sobre [nuevo tema]..."
            # * Si el usuario no ha elegido claramente una de las opciones o no está claro a qué se refiere, REVISAR todo el historial y preguntar: "¿Te refieres a la situación del [artículo/tema específico] que mencionamos antes?"

            # ### IMPORTANTE SOBRE CONTINUIDAD DEL CONTEXTO:
            # * SIEMPRE mantener presentes en la memoria TODAS las situaciones y artículos legales que se han discutido previamente en la conversación
            # * Cuando el usuario mencione "comparar con lo anterior", "misma categoría", o términos similares, IDENTIFICAR ESPECÍFICAMENTE a qué situación anterior se refiere basándose en todo el historial de la conversación
            # * NUNCA responder que "no entiendes a qué situación anterior se refiere" sin antes revisar exhaustivamente el historial completo
            # * Si realmente no puedes determinar a qué situación anterior se refiere, LISTAR EXPLÍCITAMENTE las situaciones anteriores que has mencionado: "Anteriormente hablamos de estas situaciones: 1) [situación 1], 2) [situación 2]... ¿A cuál te refieres específicamente?"

            # ### FORMATO DE RESPUESTA JSON:
            # {{
            # "diferencias": "SOLO incluir en la primera interacción. SIEMPRE listar ENTRE 3-4 situaciones legales diferentes con sus palabras clave diferenciadoras en MAYÚSCULAS. NUNCA MENOS DE 3. Cada situación debe incluir artículo, sección y numeral específico. Ejemplo: 'Artículo 380, Sección II, Numeral 3: aplica cuando NO PORTAS FÍSICAMENTE la licencia pero la tienes vigente. Artículo 381, Sección II, Numeral 4: aplica cuando NO TIENES LICENCIA VÁLIDA (nunca obtenida o vencida). Artículo 382, Sección II, Numeral 5: aplica cuando tu licencia está RETENIDA por otra infracción.' Las diferencias deben ser MUY CLARAS para que el usuario identifique exactamente cuál se aplica a su caso.",

            # "respuesta": "Con tono de confianza y seguridad, mostrando que estamos ayudándole. EXPLICAR DETALLADAMENTE CADA SITUACIÓN mencionada en 'diferencias' de manera NATURAL y FLUIDA, no como una lista mecánica de puntos. Para cada artículo, incluir en el flujo natural de la conversación:
            # - La explicación completa del artículo y cuándo aplica exactamente
            # - SIEMPRE ser preciso al mencionar los artículos: incluir sección, subsección, inciso o numeral específico donde se encuentra
            # - SIEMPRE especificar el monto EXACTO de cada multa o sanción en Bolivianos (Bs.) con el valor numérico y escrito (ejemplo: 'multa de CINCUENTA BOLIVIANOS (Bs. 50)')
            # - SIEMPRE aclarar la categoría de la sanción (grave, leve, etc.)
            # - SIEMPRE indicar si amerita retención de vehículo/licencia y por CUÁNTOS DÍAS exactamente
            # - El procedimiento correcto que debe seguir el oficial de tránsito paso a paso
            # - Consejos prácticos y específicos para el conductor en esa situación
            # - SIEMPRE detallar las CONSECUENCIAS PRÁCTICAS para el conductor: cuánto deberá pagar, dónde, en qué plazo, qué documentos necesitará para recuperar su vehículo/licencia, y si hay opciones de apelación
            # - Recomendaciones sobre cómo actuar ante intentos de soborno, con frases exactas para usar
            # - Sugerencias para documentar correctamente la situación
            # - Mencionar instituciones específicas donde reclamar o denunciar si es necesario

            # Dividir claramente la explicación de cada artículo usando subtítulos naturales o transiciones conversacionales. Todo el texto debe fluir como si fuera una conversación real con un amigo que es abogado, evitando el formato de lista de viñetas o numeración. USAR LENGUAJE SENCILLO Y DIRECTO."
            # }}


            # ### DATOS ACTUALES:
            # HISTORIAL DE CONVERSACIÓN: {historial_conversacion}
            # PREGUNTA DEL CLIENTE: {pregunta}
            # """
            
            prompt = f"""
            ### CONTEXTO: 
            Eres un abogado boliviano experto en código de tránsito, especializado en defender los derechos de conductores frente a situaciones de control policial, infracciones y posibles abusos. Debes proporcionar asesoramiento legal preciso, práctico y específico.

            ### INSTRUCCIONES CRÍTICAS:
            - SIEMPRE especifica los artículos exactos con su número, sección, subsección, inciso o numeral
            - SIEMPRE incluye montos exactos de multas en Bolivianos PARA CADA ARTÍCULO mencionado, SIN EXCEPCIÓN, indicando tanto el valor numérico como escrito
            - SIEMPRE ante cualquier situación, menciona cómo actuar ante intentos de presión o soborno
            - SIEMPRE incluye consejos para documentar la situación (grabar, testigos, etc.)
            - NUNCA des respuestas vagas o genéricas
            - SIEMPRE presenta MÍNIMO 3 y MÁXIMO 4 situaciones/artículos legales relacionados con la consulta inicial
            - SIEMPRE usa lenguaje SENCILLO y DIRECTO, como si hablaras con un amigo
            - SIEMPRE especifica el tiempo de retención del vehículo o licencia cuando aplique
            - SIEMPRE explica las CONSECUENCIAS PRÁCTICAS para el cliente: montos exactos a pagar, procesos completos, tiempos de espera, y requisitos de documentación

            ### INFORMACIÓN OBLIGATORIA PARA INCLUIR:
            1. Para TODA consulta sobre infracciones:
            - Artículo exacto infringido con número, sección y numeral
            - Monto específico de la multa en Bs (SIEMPRE expresado en números y letras, sin importar el tipo de infracción)
            - Si amerita o no retención de vehículo/licencia y por CUÁNTOS DÍAS exactamente
            - Categoría de la infracción (leve, grave, muy grave)
            - Procedimiento correcto de emisión de boleta y pasos posteriores
            - SIEMPRE explicar claramente TODAS las consecuencias para el conductor: cuánto deberá pagar exactamente, qué proceso deberá seguir, cuánto tiempo tomará, y qué documentos necesitará presentar
            - SIEMPRE especificar los plazos legales para pagar multas, recuperar vehículos/licencias, o presentar apelaciones

            2. Para TODA situación de control policial:
            - Derechos específicos del conductor (mínimo 3)
            - Documentos que legalmente pueden exigirte (listar todos)
            - Procedimiento legal que debe seguir el policía paso por paso
            - Cómo documentar discretamente (grabar, anotar placa, etc.)
            - Frases específicas para afirmar derechos sin confrontación

            3. Para TODA situación donde pueda existir soborno:
            - Mencionar que es delito de cohecho según el artículo específico del Código Penal
            - Indicar monto legal de la multa para comparación
            - Recomendar grabar discretamente la interacción
            - Proporcionar 2-3 frases exactas para rechazar soborno sin provocar confrontación
            - Vías legales específicas para denunciar posteriormente (nombres de instituciones y procedimiento)

            ### ESTRUCTURA DE RESPUESTA:
            - Si es inicio de conversación (historial vacío):
            * ANALIZAR la consulta y presentar SIEMPRE entre 3-4 situaciones/artículos legales relevantes a la situacion
            * MÍNIMO 3 situaciones/artículos legales SIEMPRE (incluso si una parece más obvia)
            * MÁXIMO 4 situaciones/artículos legales (solo cuando el análisis indique cuarta situación relevante)
            * Para cada situación/artículo dar una explicación sencilla
            * Incluir palabras clave diferenciadoras muy claras en el campo "diferencias"
            * Si la pregunta es ambigua, presentar las opciones más probables e invitar a aclarar
            * EXPLICAR CADA SITUACIÓN COMPLETAMENTE en la respuesta, incluyendo para cada artículo mencionado:
                - Qué establece exactamente el artículo (texto legal simplificado)
                - Monto específico de la multa (en número y letras)
                - Si amerita retención de vehículo/licencia y por cuántos días exactamente
                - Procedimiento correcto que debe seguir el policía
                - Consejos para el conductor
                - Qué hacer ante intentos de presión o soborno
                - Derechos específicos que puede invocar

            - Si es continuación de conversación:
            * REVISAR lo que se mencionó anteriormente en el historial de conversación, especialmente las situaciones y artículos legales que ya se discutieron
            * RECORDAR y REFERIRSE ESPECÍFICAMENTE a las situaciones/artículos mencionados anteriormente
            * Ya no mostrar múltiples opciones
            * Proporcionar una respuesta legal precisa específica a la situación elegida o mencionada previamente
            * Primer párrafo: Explicación legal específica (artículos y multas)
            * Segundo párrafo: Acciones prácticas inmediatas (5-7 puntos concretos)
            * Tercer párrafo: Consejos para situaciones de posible abuso/soborno
            * Si el usuario hace una pregunta sobre una "situación anterior" o compara con algo "anterior", SIEMPRE buscar en el historial a qué situación específica se refiere y responder en ese contexto
            * Si el usuario menciona un nuevo tema relacionado con un tema anterior, ESTABLECER EXPLÍCITAMENTE la relación: "Sobre tu pregunta anterior de [tema] y esta nueva consulta sobre [nuevo tema]..."
            * Si el usuario no ha elegido claramente una de las opciones o no está claro a qué se refiere, REVISAR todo el historial y preguntar: "¿Te refieres a la situación del [artículo/tema específico] que mencionamos antes?"

            ### IMPORTANTE SOBRE CONTINUIDAD DEL CONTEXTO:
            * SIEMPRE mantener presentes en la memoria TODAS las situaciones y artículos legales que se han discutido previamente en la conversación
            * Cuando el usuario mencione "comparar con lo anterior", "misma categoría", o términos similares, IDENTIFICAR ESPECÍFICAMENTE a qué situación anterior se refiere basándose en todo el historial de la conversación
            * NUNCA responder que "no entiendes a qué situación anterior se refiere" sin antes revisar exhaustivamente el historial completo
            * Si realmente no puedes determinar a qué situación anterior se refiere, LISTAR EXPLÍCITAMENTE las situaciones anteriores que has mencionado: "Anteriormente hablamos de estas situaciones: 1) [situación 1], 2) [situación 2]... ¿A cuál te refieres específicamente?"

            ### FORMATO DE RESPUESTA JSON:
            {{
            "diferencias": "SOLO incluir en la primera interacción. SIEMPRE listar ENTRE 3-4 situaciones legales diferentes con sus palabras clave diferenciadoras en MAYÚSCULAS. NUNCA MENOS DE 3. Cada situación debe incluir artículo, sección y numeral específico. Ejemplo: 'Artículo 380, Sección II, Numeral 3: aplica cuando NO PORTAS FÍSICAMENTE la licencia pero la tienes vigente. Artículo 381, Sección II, Numeral 4: aplica cuando NO TIENES LICENCIA VÁLIDA (nunca obtenida o vencida). Artículo 382, Sección II, Numeral 5: aplica cuando tu licencia está RETENIDA por otra infracción.' Las diferencias deben ser MUY CLARAS para que el usuario identifique exactamente cuál se aplica a su caso.",

            "respuesta": "Con tono de confianza y seguridad, mostrando que estamos ayudándole. EXPLICAR DETALLADAMENTE CADA SITUACIÓN mencionada en 'diferencias' de manera NATURAL y FLUIDA, usando MÚLTIPLES PÁRRAFOS CON SALTOS DE LÍNEA para mejorar la legibilidad.\\n\\nSIEMPRE usar SALTOS DE LÍNEA DOBLES (\\n\\n) entre párrafos distintos y SALTOS DE LÍNEA SIMPLES (\\n) entre ideas relacionadas dentro del mismo párrafo.\\n\\nPara cada artículo, incluir en el flujo natural de la conversación:\\n
            - La explicación completa del artículo y cuándo aplica exactamente\\n
            - SIEMPRE ser preciso al mencionar los artículos: incluir sección, subsección, inciso o numeral específico donde se encuentra\\n
            - SIEMPRE especificar el monto EXACTO de cada multa o sanción en Bolivianos (Bs.) con el valor numérico y escrito (ejemplo: 'multa de CINCUENTA BOLIVIANOS (Bs. 50)')\\n
            - SIEMPRE aclarar la categoría de la sanción (grave, leve, etc.)\\n
            - SIEMPRE indicar si amerita retención de vehículo/licencia y por CUÁNTOS DÍAS exactamente\\n
            - El procedimiento correcto que debe seguir el oficial de tránsito paso a paso\\n
            - Consejos prácticos y específicos para el conductor en esa situación\\n
            - SIEMPRE detallar las CONSECUENCIAS PRÁCTICAS para el conductor: cuánto deberá pagar, dónde, en qué plazo, qué documentos necesitará para recuperar su vehículo/licencia, y si hay opciones de apelación\\n
            - Recomendaciones sobre cómo actuar ante intentos de soborno, con frases exactas para usar\\n
            - Sugerencias para documentar correctamente la situación\\n
            - Mencionar instituciones específicas donde reclamar o denunciar si es necesario\\n\\n

            INICIAR UN NUEVO PÁRRAFO al cambiar de tema o de artículo legal.\\n\\n

            USAR SUBTÍTULOS O TITULARES en mayúsculas seguidos de dos puntos para separar claramente los distintos artículos o secciones relevantes.\\n\\n

            Todo el texto debe fluir como si fuera una conversación real con un amigo que es abogado, evitando el formato de lista de viñetas o numeración. USAR LENGUAJE SENCILLO Y DIRECTO."
            }}


            ### DATOS ACTUALES:
            HISTORIAL DE CONVERSACIÓN: {historial_conversacion}
            PREGUNTA DEL CLIENTE: {pregunta}
            """

            # "sugerencia_temas_relacionados": ["Tema relacionado específico 1", "Tema relacionado específico 2"]
            print(f"Procesando consulta: {pregunta}")
            # Usar invoke() en lugar de run()
            from langchain_core.prompts import PromptTemplate
            
            # Normal trae respuesta generales digamos 
            

            # Crear un prompt template
            prompt_template = PromptTemplate.from_template(prompt)
            
            # Invocar la cadena con el prompt
            respuesta = self.qa.invoke({"query": prompt})
            
            # La respuesta puede venir en diferentes formatos dependiendo de la versión de langchain
            # Intentar extraer el resultado
            if isinstance(respuesta, dict) and "result" in respuesta:
                respuesta_cruda = respuesta["result"]
            elif hasattr(respuesta, "result"):
                respuesta_cruda = respuesta.result
            else:
                respuesta_cruda = str(respuesta)
            
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
            import traceback
            traceback.print_exc()  # Imprime el stack trace completo para mejor diagnóstico
            return {
                "fueraDeContexto": False,
                "respuestaAmigo": "Disculpa, ocurrió un error. Intenta con otra pregunta."
            }
            
