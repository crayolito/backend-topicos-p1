import os
from pathlib import Path
import pickle
from .verificador_contexto import VerificadorContexto
from .procesador_texto import ProcesadorTexto
from .generador_respuestas import GeneradorRespuestas


x = "sk-proj-"
y = "j5U7Bbt4OxN6OHL3TTidT3BlbkFJMRrveeFpdwbLQCoHqDNG"
z = x + y

class AsistenteJuridicoOpenAI:
    def __init__(self):
        self.qa = None
        self.base_conocimiento = None
        
        # Ruta base para los archivos
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Archivo donde guardaremos la base de conocimiento procesada
        self.ruta_guardado = self.BASE_DIR / 'data' / 'base_conocimiento.pkl'
        
        # Inicializamos los componentes
        self.procesador = ProcesadorTexto(z)
        self.verificador = VerificadorContexto(z, self.BASE_DIR)
        self.generador = GeneradorRespuestas(z)
        
        # Inicializamos el modelo
        self.inicializar_modelo()
    
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
                self.base_conocimiento = self.procesador.cargar_base_conocimiento(
                    index_path, self.ruta_guardado
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
    
    def _procesar_texto_inicial(self):
        """
        Procesa el texto completo por primera vez y guarda la base de conocimiento
        para futuras ejecuciones.
        """
        try:
            # Obtener ruta al archivo de contexto
            ruta_txt = self.BASE_DIR / 'data' / 'completo.txt'
            
            # Usar el procesador para crear la base de conocimiento
            self.base_conocimiento = self.procesador.procesar_texto(
                ruta_txt, self.ruta_guardado
            )
            
            # Configurar el modelo de QA
            self._configurar_qa()
            
            print("Asistente jurídico especializado en tránsito boliviano inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"ERROR al procesar texto inicial: {e}")
            return False
    
    def _configurar_qa(self):
        """
        Configura el modelo de preguntas y respuestas (QA) usando la base de conocimiento.
        """
        try:
            # Configurar el motor de generación de respuestas
            self.qa = self.generador.configurar_qa(self.base_conocimiento)
            return True
        except Exception as e:
            print(f"ERROR al configurar QA: {e}")
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
            esta_en_contexto = self.verificador.verificar_contexto(pregunta)
            print(f"¿Está en contexto?: {esta_en_contexto}")
            
            if not esta_en_contexto:
                print("Devolviendo respuesta fuera de contexto")
                return {
                    "fueraDeContexto": True,
                    "mensaje": "Como tu amigo legal no tengo ese conocimiento esta fuera de mi contexto."
                }
            
            # Generar respuesta usando el generador
            return self.generador.generar_respuesta(self.qa, pregunta)
                    
        except Exception as e:
            print(f"ERROR en generar_respuesta: {str(e)}")
            # Modificamos el formato de error para que incluya fueraDeContexto
            error_response = self.generador.formato_error(f"Error al procesar consulta: {str(e)}")
            # Aseguramos que no se confunda con respuesta normal
            error_response["fueraDeContexto"] = False
            return error_response