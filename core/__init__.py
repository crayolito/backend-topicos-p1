# Exportar las clases públicas
from .modelo_ia import AsistenteJuridicoOpenAI, AsistenteJuridicoDeepSek

# Definir qué se debe importar al hacer "from core import *"
__all__ = ['AsistenteJuridicoOpenAI', 'AsistenteJuridicoDeepSek']