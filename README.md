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
