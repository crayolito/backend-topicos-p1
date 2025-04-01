import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class ProcesadorTexto:
    def __init__(self, api_key, usar_openrouter=False):
        self.api_key = api_key
        self.usar_openrouter = usar_openrouter
    
    def _obtener_embeddings(self):
        """
        Obtiene el modelo de embeddings adecuado según la configuración.
        """
        if self.usar_openrouter:
            # Para OpenRouter/DeepSeek usamos embeddings locales de HuggingFace
            # ya que OpenRouter no tiene un servicio de embeddings confiable
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            # Para OpenAI usamos sus embeddings nativos
            return OpenAIEmbeddings(
                api_key=self.api_key,
                model="text-embedding-ada-002"
            )
    
    def procesar_texto(self, ruta_txt, ruta_guardado):
        """
        Procesa un archivo de texto y crea una base de conocimiento vectorial.
        """
        print(f"Cargando conocimiento jurídico desde: {ruta_txt}")
        
        # Verificar que el archivo existe
        if not os.path.exists(ruta_txt):
            print(f"ERROR: El archivo {ruta_txt} no existe.")
            return None
            
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
        print(f"Generando embeddings especializados usando {'HuggingFace' if self.usar_openrouter else 'OpenAI'}...")
        vectores = self._obtener_embeddings()
        
        base_conocimiento = FAISS.from_documents(fragmentos, vectores)
        
        # Guardar la base de conocimiento procesada
        self._guardar_base_conocimiento(base_conocimiento, ruta_guardado)
        
        return base_conocimiento
    
    def _guardar_base_conocimiento(self, base_conocimiento, ruta_guardado):
        """
        Guarda la base de conocimiento en disco.
        """
        print(f"Guardando base de conocimiento procesada en: {ruta_guardado}")
        
        # Creamos un directorio para los archivos de la base de conocimiento
        base_dir = os.path.dirname(ruta_guardado)
        os.makedirs(base_dir, exist_ok=True)
        
        # Guardar el índice FAISS y los metadatos de documentos por separado
        index_path = str(ruta_guardado) + ".faiss"
        
        # Guardar el índice FAISS
        base_conocimiento.save_local(
            folder_path=base_dir, 
            index_name=os.path.basename(index_path)
        )
    
    def cargar_base_conocimiento(self, index_path, ruta_guardado):
        """
        Carga una base de conocimiento existente desde disco.
        """
        vectores = self._obtener_embeddings()
        
        base_dir = os.path.dirname(ruta_guardado)
        base_conocimiento = FAISS.load_local(
            folder_path=base_dir,
            index_name=os.path.basename(index_path),
            embeddings=vectores
        )
        
        return base_conocimiento