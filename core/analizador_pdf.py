import os
import urllib.parse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# API Key predefinida
x = "sk-proj-"
y = "j5U7Bbt4OxN6OHL3TTidT3BlbkFJMRrveeFpdwbLQCoHqDNG"
z = x + y

def main():
    while True:
        # Pedir la ruta del PDF
        ruta = input("Ruta del PDF: ")
        
        # Verificar si existe el archivo y es un PDF
        if not os.path.exists(ruta) or not ruta.lower().endswith('.pdf'):
            print("Error: Archivo no válido")
            continue
        print("Procesando...")
        
        try:
            # Cargar PDF usando la ruta decodificada
            cargador = PyPDFLoader(ruta)
            documento = cargador.load()
            
            # Dividir en fragmentos
            divisor = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            fragmentos = divisor.split_documents(documento)
            
            # Crear vectores y base de datos
            vectores = OpenAIEmbeddings(api_key=z)
            informacionPDF = FAISS.from_documents(fragmentos, vectores)
            
            # Modelo de chat
            modelo = ChatOpenAI(temperature=0.8, 
                               model_name="gpt-3.5-turbo", 
                               api_key=z)
            
            # Cadena de preguntas y respuestas
            qa = RetrievalQA.from_chain_type(
                llm=modelo,
                chain_type="stuff",
                retriever=informacionPDF.as_retriever()
            )
            
            print("Listo para responder preguntas")
            
            # Bucle de preguntas
            while True:
                pregunta = input("\nPregunta: ")
                
                if pregunta.lower() == "salir":
                    return
                elif pregunta.lower() == "reset":
                    break
                    
                try:
                    print("Procesando...")
                    respuesta = qa.run(pregunta)
                    print(f"\nRespuesta: {respuesta}")
                except:
                    print("Error al procesar la pregunta")
                    
        except Exception as e:
            print(f"Error al procesar el PDF: {e}")

        # En esta parte se limpia la ruta por si tiene caracteres especiales 
if __name__ == "__main__":
    print("=== ANALIZADOR DE PDF ===")
    main()
    print("Programa finalizado")