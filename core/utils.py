import os
from pathlib import Path
import re
import unicodedata

def normalizar_texto(texto):
    """
    Normaliza el texto: elimina tildes y convierte a minúsculas.
    Función de utilidad disponible para todos los módulos.
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar tildes
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                  if unicodedata.category(c) != 'Mn')
    
    return texto

def reemplazar_modismos(texto, modismos=None):
    """
    Reemplaza modismos regionales por sus equivalentes estándar.
    """
    if modismos is None:
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

def asegurar_directorio(ruta):
    """
    Asegura que un directorio exista, creándolo si no existe.
    """
    directorio = os.path.dirname(ruta)
    if not os.path.exists(directorio):
        os.makedirs(directorio, exist_ok=True)
    return directorio

def limpiar_json(texto_json):
    """
    Limpia un texto JSON para asegurar que es válido.
    """
    # Si no comienza con '{'
    if not texto_json.strip().startswith('{'):
        inicio_json = texto_json.find('{')
        if inicio_json != -1:
            texto_json = texto_json[inicio_json:]
    
    # Si no termina con '}'
    if not texto_json.strip().endswith('}'):
        fin_json = texto_json.rfind('}')
        if fin_json != -1:
            texto_json = texto_json[:fin_json+1]
    
    return texto_json