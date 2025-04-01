# Exportar las clases públicas
from .asistente_juridico_openIA import AsistenteJuridicoOpenAI
from .asistente_juridico_deepseek import AsistenteJuridicoDeepSeek

# Definir qué se debe importar al hacer "from core import *"
__all__ = ['AsistenteJuridicoOpenAI', 
           'AsistenteJuridicoDeepSeek'
           ]