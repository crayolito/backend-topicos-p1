import psycopg2
from pathlib import Path

class BaseConocimientoMobil:
    def __init__(self, db_config=None):
        # Si no se proporciona configuración, usar la del AsistenteJuridico
        if db_config is None:
            self.db_config = {
                'dbname': 'BDRodalex',
                'user': 'postgres',
                'password': 'clave123',
                'host': 'localhost',
                'port': '5432'
            }
        else:
            self.db_config = db_config
            
    def obtener_todos_fragmentos(self):
        """
        Obtiene todos los registros de la tabla fragmentos_texto
        Retorna una lista de tuplas (id, contenido)
        """
        try:
            # Conectar a la base de datos
            conn = psycopg2.connect(**self.db_config)
            
            # Crear un cursor
            cursor = conn.cursor()
            
            # Ejecutar la consulta
            cursor.execute("SELECT id, contenido FROM fragmentos_texto")
            
            # Obtener todos los resultados
            resultados = cursor.fetchall()
            
            # Cerrar cursor y conexión
            cursor.close()
            conn.close()
            
            return resultados
            
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return []
    
    def obtener_fragmento_por_id(self, id):
        """
        Obtiene un fragmento específico por su ID
        Retorna una tupla (id, contenido) o None si no existe
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, contenido FROM fragmentos_texto WHERE id = %s", (id,))
            resultado = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return resultado
            
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return None
    
    def buscar_fragmentos(self, texto_busqueda):
        """
        Busca fragmentos que contengan el texto de búsqueda
        Retorna una lista de tuplas (id, contenido)
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Búsqueda usando ILIKE para que sea insensible a mayúsculas/minúsculas
            cursor.execute("SELECT id, contenido FROM fragmentos_texto WHERE contenido ILIKE %s", 
                          (f'%{texto_busqueda}%',))
            
            resultados = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return resultados
            
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return []