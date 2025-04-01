import os
from pathlib import Path
import pickle
import requests
from .verificador_contexto import VerificadorContexto
from .procesador_texto import ProcesadorTexto
from .generador_respuestas import GeneradorRespuestas
from .utils import limpiar_json

x = "sk-or-v1-"
y = "3d0f4a66595664f1ae778aed81a476ae2eb7636e154f32f2963470f1204d8c03"
z = x + y

class AsistenteJuridicoDeepSeek:
    def __init__(self):
        self.qa = None
        self.base_conocimiento = None
        
        # Ruta base para los archivos
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Archivo donde guardaremos la base de conocimiento procesada
        self.ruta_guardado = self.BASE_DIR / 'data' / 'base_conocimiento_deepseek.pkl'
        
        # Inicializamos los componentes
        # Usamos la misma clase procesador pero con embeddings adecuados para DeepSeek
        self.procesador = ProcesadorTexto(z, usar_openrouter=True)
        
        # El verificador puede ser el mismo, ya que la verificación es independiente del modelo
        self.verificador = VerificadorContexto(z, self.BASE_DIR)
        
        # Para el generador necesitamos usar OpenRouter en lugar de OpenAI directamente
        self.generador = GeneradorRespuestas(z, usar_openrouter=True)
        
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
            print(f"ERROR al inicializar el modelo jurídico DeepSeek: {e}")
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
            
            print("Asistente jurídico DeepSeek especializado en tránsito boliviano inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"ERROR al procesar texto inicial para DeepSeek: {e}")
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
            print(f"ERROR al configurar QA para DeepSeek: {e}")
            return False

    def generar_respuesta(self, pregunta):
        """
        Genera una respuesta jurídica basada en la pregunta del usuario.
        """
        if not self.qa:
            print("Error: Sistema DeepSeek no inicializado")
            return {"fueraDeContexto": True, "mensaje": "El sistema no está inicializado correctamente."}
        
        try:
            # Verificar si la pregunta está dentro del contexto jurídico de tránsito
            esta_en_contexto = self.verificador.verificar_contexto(pregunta)
            print(f"DeepSeek - ¿Está en contexto?: {esta_en_contexto}")
            
            if not esta_en_contexto:
                print("DeepSeek - Devolviendo respuesta fuera de contexto")
                return {
                    "fueraDeContexto": True,
                    "mensaje": "Como tu amigo legal no tengo ese conocimiento esta fuera de mi contexto."
                }
            
            # Generar respuesta usando el generador específico para DeepSeek
            return self.generador.generar_respuesta(self.qa, pregunta)
                    
        except Exception as e:
            print(f"ERROR en generar_respuesta DeepSeek: {str(e)}")
            # Modificamos el formato de error para que incluya fueraDeContexto
            error_response = self.generador.formato_error(f"Error al procesar consulta con DeepSeek: {str(e)}")
            # Aseguramos que no se confunda con respuesta normal
            error_response["fueraDeContexto"] = False
            return error_response