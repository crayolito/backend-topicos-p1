# Asistente Jurídico API

## Pasos básicos para ejecutar

1. **Crear entorno virtual**
   python -m venv venv

2. **Activar entorno virtual**
   Windows: venv\Scripts\activate

3. **Instalar dependencias**
   pip install -r requirements.txt

4. **Iniciar el servidor**
   python app.py

Si necesitas defender tu código desde cero, aquí tienes una explicación completa:

"En este proyecto, desarrollé un sistema inteligente para clasificar consultas legales de tránsito usando técnicas de procesamiento de lenguaje natural. El componente central es el `VerificadorContexto`, que determina si una consulta está relacionada con el derecho de tránsito boliviano.

El sistema implementa un enfoque de clasificación textual basado en reglas que sigue estos pasos:

1. **Normalización lingüística**: Primero normalizo el texto eliminando acentos, convirtiendo a minúsculas y traduciendo modismos locales bolivianos a términos estándar. Esto es crucial porque los usuarios escriben con variaciones ortográficas y modismos regionales.

2. **Análisis multifase**:

   - Fase 1: Detección de palabras clave relevantes al dominio de tránsito
   - Fase 2: Filtrado de falsos positivos mediante palabras clave negativas
   - Fase 3: Verificación de frases específicas del contexto de tránsito

3. **Detección de ambigüedades**: Para casos donde hay términos que podrían pertenecer a múltiples dominios (como "código" que puede referirse a código de tránsito o código de programación), implementé una función `confirmar_contexto_transito()` que analiza el contexto más amplio para desambiguar.

Este enfoque tiene ventajas significativas sobre modelos de machine learning tradicionales:

- No requiere grandes conjuntos de datos de entrenamiento
- Funciona con alta precisión desde el primer momento
- Es transparente y auditable (importante en contextos legales)
- Se puede adaptar y mejorar fácilmente

El verificador se integra con el sistema más amplio `AsistenteJuridico` que utiliza una base de conocimiento vectorial para proporcionar respuestas legales precisas sobre tránsito, pero solo cuando el verificador determina que la consulta está dentro del dominio correcto.

Esta solución híbrida combina técnicas clásicas de NLP con tecnologías modernas de embeddings vectoriales, resultando en un sistema eficiente y efectivo para consultas legales específicas."

Te prepararé respuestas para preguntas técnicas que podrían hacerte sobre tu sistema:

### Base de conocimiento - ¿Qué es y cómo funciona?

"La base de conocimiento es el cerebro de nuestro asistente jurídico. Es una colección estructurada de información legal sobre tránsito que se indexa y almacena de forma optimizada para búsquedas semánticas. Procesamos el archivo completo.txt que contiene toda la normativa de tránsito, lo dividimos en fragmentos manejables, y lo convertimos en vectores numéricos que representan el significado de cada texto."

### Formato PKL y FAISS - ¿Por qué los usamos?

"Utilizamos dos formatos para guardar la base de conocimiento:

- El formato PKL (pickle) es un formato de serialización de Python que guarda la estructura de los objetos.
- FAISS (Facebook AI Similarity Search) es una biblioteca especializada para búsquedas de similitud en vectores de alta dimensión. La elegimos porque permite búsquedas extremadamente rápidas incluso con miles de documentos, lo que garantiza que nuestro asistente responda en tiempo real."

### Embeddings - ¿Qué son y para qué sirven?

"Los embeddings son representaciones numéricas de texto en forma de vectores. Utilizamos el modelo text-embedding-ada-002 de OpenAI que convierte frases y párrafos en vectores de 1,536 dimensiones. Estos vectores capturan el significado semántico del texto, permitiendo que cuando alguien pregunte '¿Qué pasa si me detienen por exceso de velocidad?', el sistema pueda encontrar información relevante aunque no contenga exactamente esas palabras. Es como traducir el lenguaje humano a un formato matemático que las computadoras pueden comparar eficientemente."

### Divisor de texto - ¿Por qué es necesario?

"El divisor de texto (RecursiveCharacterTextSplitter) es esencial porque los modelos de embeddings tienen límites en la cantidad de texto que pueden procesar de una vez. Dividimos el documento completo en fragmentos de aproximadamente 1,000 caracteres con superposición de 200 caracteres para mantener el contexto. Esto permite:

1. Procesar documentos extensos como códigos legales completos
2. Recuperar solo las partes relevantes a una consulta específica
3. Mantener la coherencia contextual entre fragmentos gracias a la superposición

Sin esta división, perderíamos información valiosa o tendríamos respuestas imprecisas."

### Modelo OpenAI - ¿Qué hace y por qué lo elegimos?

"Utilizamos dos modelos de OpenAI para tareas diferentes:

1. text-embedding-ada-002: Genera los vectores que representan el significado del texto legal
2. gpt-4-turbo: Genera respuestas claras y con formato legal específico

Elegimos estos modelos porque:

- Tienen excelente comprensión del español y contextos legales
- Pueden seguir instrucciones precisas para generar respuestas estructuradas en formato JSON
- Son capaces de interpretar y explicar textos legales complejos en lenguaje cercano y accesible

El parámetro temperature=0.3 asegura que las respuestas sean consistentes y precisas sin demasiada creatividad, lo cual es crucial en contextos legales."

### Arquitectura general del sistema

"Nuestro sistema opera en tres etapas principales:

1. Verificación de contexto: Determina si la consulta es sobre tránsito
2. Recuperación de información: Busca los 5 fragmentos más relevantes de la base de conocimiento
3. Generación de respuesta: Formula una respuesta estructurada y personalizada como 'amigo abogado'

Esta arquitectura permite que el sistema sea preciso, relevante y eficiente en recursos computacionales."

Estas explicaciones te ayudarán a responder preguntas técnicas sobre tu implementación de manera clara y profesional.

# Verificador de Contexto para Tránsito en Bolivia

## Descripción General

El Verificador de Contexto es un sistema especializado para determinar si una consulta está relacionada con temas de tránsito en Bolivia. Utiliza programación basada en reglas con ponderación en lugar de depender de librerías de machine learning, ofreciendo las siguientes ventajas:

- **No requiere librerías externas complejas** - Implementado con Python puro
- **Especializado en el dominio boliviano** - Incorpora terminología local específica
- **Explicable y transparente** - Proporciona justificación detallada para cada clasificación
- **Ligero y eficiente** - Optimizado para entornos con recursos limitados

## Componentes Principales

### 1. Sistema de Diccionarios Ponderados

En lugar de simples listas de palabras clave, utiliza diccionarios donde:

- **Las claves** son términos o frases
- **Los valores** son pesos numéricos que indican la relevancia

```python
self.modismos_bolivianos = {
    "verde": 3,        # Término común para policía, alto peso
    "caminero": 3,     # Término específico, alto peso
    "trufi": 3,        # Tipo de transporte boliviano, alto peso
    # ...más términos
}
```

La ventaja es que no todas las palabras tienen la misma importancia. Términos especializados como "Policía Caminera" o "DIPROVE" tienen mayor peso que términos generales como "auto" o "calle".

### 2. Categorización Jerárquica

Los términos están organizados en categorías jerárquicas:

```python
# Nivel 1: Términos altamente específicos (peso 4)
self.terminos_bolivia_alta = {
    "transito bolivia": 4,
    "policia caminera": 4,
    # ...
}

# Nivel 2: Modismos locales (peso 3)
self.modismos_bolivianos = {...}

# Nivel 3: Referencias geográficas (peso 2)
self.geografia_bolivia = {...}
```

Esta estructura permite que el sistema priorice términos más específicos sobre términos generales.

### 3. Análisis de Patrones Sintácticos

El sistema detecta patrones lingüísticos usando expresiones regulares para capturar estructuras comunes:

```python
self.patrones_bolivianos = [
    (r'me (pararon|detuvieron|chaparon) (los|el|la) (transito|policia|verde)', 5),
    # ...
]
```

Esto permite identificar frases como "me pararon los verdes" o "me detuvieron los de tránsito" que son altamente indicativas del contexto buscado.

### 4. Normalización de Texto

Antes del análisis, el texto se normaliza para manejar variaciones:

```python
def normalizar_texto(self, texto):
    # Minúsculas
    texto = texto.lower()

    # Eliminar tildes
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                  if unicodedata.category(c) != 'Mn')

    # Sustituir modismos
    for modismo, estandar in self.modismos.items():
        texto = re.sub(r'\b' + modismo + r'\b', estandar, texto)
```

Esto permite que variaciones como "Brevete" y "brevete" se reconozcan igual, y que modismos como "tombo" se normalicen a "policía".

### 5. Sistema de Puntuación Ponderada

El núcleo del sistema es el cálculo de puntaje:

```python
def calcular_puntaje_texto(self, texto):
    puntaje = 0
    detalles = {...}  # Para almacenar qué términos se encontraron

    # Suma puntos por términos positivos encontrados
    for termino, peso in self.terminos_bolivia_alta.items():
        if termino in texto:
            puntaje += peso
            detalles["terminos_bolivia"].append((termino, peso))

    # Resta puntos por términos negativos
    for palabra, peso in self.palabras_clave_negativas.items():
        if re.search(r'\b' + palabra + r'\b', texto):
            puntaje -= peso
            detalles["palabras_negativas"].append((palabra, peso))
```

Si el puntaje supera un umbral predefinido (por defecto 3 puntos), la consulta se clasifica como "en contexto de tránsito".

### 6. Memoria Conversacional

El sistema aprende del contexto de la conversación:

```python
def analizar_historial(self):
    # Dar más peso a preguntas recientes
    puntajes_ponderados = [(puntajes_previos[i] * (i + 1))
                           for i in range(len(puntajes_previos))]

    # Calcular promedio ponderado
    promedio_ponderado = sum(puntajes_ponderados) / sum(range(1, len(puntajes_previos) + 2))
```

Si las preguntas anteriores eran sobre tránsito, el sistema considera más probable que la siguiente también lo sea.

### 7. Detección de Falsos Positivos

El verificador implementa reglas específicas para evitar clasificaciones incorrectas:

```python
def detectar_falsos_positivos(self, texto, puntaje, detalles):
    # Detectar si es sobre crear un sistema de tránsito
    if (any(re.search(r'\b' + palabra + r'\b', texto) for palabra in palabras_creacion) and
        any(re.search(r'\b' + palabra + r'\b', texto) for palabra in palabras_tecnologia)):

        return True, -puntaje * 0.8  # Reducir el puntaje
```

Esto previene que consultas como "quiero programar un sistema de multas de tránsito" se clasifiquen incorrectamente.

## Algoritmo Principal

El algoritmo sigue estos pasos:

1. **Normalizar** el texto de entrada
2. **Buscar términos** en diferentes categorías y asignar puntos
3. **Buscar patrones** sintácticos específicos
4. **Detectar falsos positivos** y aplicar ajustes
5. **Considerar el historial** de la conversación
6. **Calcular puntuación final** y compararla con un umbral
7. **Generar diagnóstico detallado** de la clasificación

## Uso Básico

```python
# Crear instancia del verificador
verificador = VerificadorContexto()

# Verificar una consulta
esta_en_contexto, puntaje, confianza, diagnostico = verificador.verificar_contexto("Me pararon los verdes en la tranca")

# Para depuración y análisis detallado
verificador.procesar_pregunta_para_desarrollo("Me pararon los verdes en la tranca")
```

## Ventajas del Enfoque

- **Conocimiento de dominio incorporado**: Incluye terminología específica boliviana que modelos genéricos no tendrían
- **Eficiencia de recursos**: No requiere librerías pesadas de ML
- **Explicabilidad**: Cada decisión puede ser explicada exactamente
- **Flexibilidad**: Fácil de actualizar con nuevos términos o patrones
- **Robustez ante ambigüedades**: Maneja bien casos límite con el sistema de puntuación

Este enfoque es técnicamente sofisticado y bien fundamentado en principios de procesamiento de lenguaje natural, pero implementado de forma práctica y eficiente para el caso específico de tránsito en Bolivia.
