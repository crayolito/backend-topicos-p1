import re
import unicodedata
import numpy as np
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class VerificadorContexto:
    def __init__(self, api_key, base_dir):
        self.api_key = api_key
        self.BASE_DIR = base_dir
        self.clasificador = None
        self.ruta_clasificador = self.BASE_DIR / 'data' / 'clasificador_contexto.pkl'
        
        # Verificar si scikit-learn está disponible
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            self.sklearn_disponible = True
            self._inicializar_clasificador()
        except ImportError:
            print("scikit-learn no está instalado. Se usarán métodos alternativos de verificación de contexto.")
            self.sklearn_disponible = False
    
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
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            
            print("Creando nuevo clasificador de contexto...")
            # Datos de entrenamiento: ejemplos de consultas de tránsito y no tránsito
            X_train = [
                # Consultas de tránsito (etiqueta: 1)
                "me pararon por exceso de velocidad",
                "el poli me quiere quitar la licencia",
                # ... (resto de ejemplos de entrenamiento)
            ]
            
            # Etiquetas: 1 para tránsito, 0 para no tránsito
            y_train = [1] * 24 + [0] * 10
            
            # Crear pipeline con TF-IDF y clasificador Naive Bayes
            self.clasificador = Pipeline([
                ('vectorizador', TfidfVectorizer(
                    lowercase=True,
                    analyzer='word',
                    ngram_range=(1, 2),
                    stop_words=['de', 'la', 'el', 'en', 'por', 'con', 'a', 'y'],
                    max_features=1000
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
            # ... (resto de modismos)
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
            # ... (resto de palabras clave)
        ]
        
        for palabra in palabras_clave:
            if palabra in pregunta_normalizada:
                return True
        return False
    
    def _verificar_contexto_clasificador(self, pregunta_normalizada):
        """
        Verifica si la pregunta está en contexto usando el clasificador ML.
        """
        if not self.sklearn_disponible or not self.clasificador:
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
                # ... (resto de consultas de referencia)
            ]
            
            # Usar el mismo motor de embeddings que usamos para la base de conocimiento
            vectores = OpenAIEmbeddings(
                api_key=self.api_key,
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
                temperature=0,
                model_name="gpt-3.5-turbo",
                api_key=self.api_key,
                max_tokens=50
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
                # ... (resto de palabras clave negativas)
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