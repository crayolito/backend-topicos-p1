# Exportar las clases públicas
from .modelo_ia import  AsistenteJuridico;
from .speech_to_text import GoogleSpeechToText;


# Definir qué se debe importar al hacer "from core import *"
__all__ = ['AsistenteJuridico','GoogleSpeechToText']