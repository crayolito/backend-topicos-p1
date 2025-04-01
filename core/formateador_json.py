class FormateadorJSON:
    def __init__(self):
        # Valores predeterminados para cada campo
        self.valores_por_defecto = {
            "siEresInocente": [
                "Solicita ver la evidencia de la infracción. Según el Art. 149 del Código de Tránsito, tienen que mostrarte la prueba fehaciente.",
                "Pregunta específicamente qué artículo estás infringiendo. Si no pueden decirte, anótalo como evidencia.",
                "Pide el nombre completo del oficial y su número de placa. El Art. 13 del Reglamento Policial exige identificación visible."
            ],
            "siEresCulpable": [
                "Solicita una boleta oficial en vez de pagar en efectivo. Según el Art. 225, las multas se pagan en entidades bancarias.",
                "Si aceptas la infracción, pide la categoría correcta según el Reglamento para evitar sobrecargos.",
                "No discutas ni niegues lo evidente. Mantén la calma y busca la solución más proporcional según Art. 73 de la Ley 2341."
            ],
            "frasesClave": [
                "'Oficial, entiendo la situación. Por favor, emítame la boleta oficial para pagarla en el banco como establece el Art. 225 del Código de Tránsito.'",
                "'Con todo respeto oficial, necesito conocer el artículo específico que estoy infringiendo para entender la situación.'"
            ],
            "trucosLegales": [
                "Si el equipo de medición (radar) no está calibrado en las últimas 24 horas, puedes impugnar la multa (Art. 149).",
                "Si no especifican el artículo infringido en la boleta, tienes base legal para impugnar (Art. 73 Ley 2341)."
            ],
            "derechosEsenciales": [
                "Derecho a conocer la infracción específica (Art. 16 de la Ley 2341).",
                "Derecho a no ser detenido por infracciones que no constituyan delito penal (Art. 231)."
            ]
        }
        
        # Requisitos mínimos para cada lista
        self.min_elementos = {
            "siEresInocente": 3,
            "siEresCulpable": 3,
            "frasesClave": 2,
            "trucosLegales": 2,
            "derechosEsenciales": 2
        }
    
    def validar_completar_json(self, respuesta_json):
        """
        Valida que el JSON tenga todos los campos requeridos y los completa si faltan.
        """
        # Campos requeridos en el formato simplificado
        campos_requeridos = [
            "situacionLegal", 
            "siEresInocente", 
            "siEresCulpable", 
            "frasesClave", 
            "trucosLegales", 
            "derechosEsenciales",
            "infoImportante"
        ]
        
        # Verificar y completar campos faltantes
        for campo in campos_requeridos:
            if campo not in respuesta_json:
                if campo in ["siEresInocente", "siEresCulpable", "frasesClave", "trucosLegales", "derechosEsenciales"]:
                    respuesta_json[campo] = ["Información no disponible"]
                else:
                    respuesta_json[campo] = "Información no disponible"
        
        # Asegurar que los campos de listas sean listas
        campos_lista = ["siEresInocente", "siEresCulpable", "frasesClave", "trucosLegales", "derechosEsenciales"]
        for campo in campos_lista:
            if not isinstance(respuesta_json[campo], list):
                respuesta_json[campo] = [respuesta_json[campo]]
        
        # Completar con valores predeterminados si no hay suficientes elementos
        self._completar_valores_predeterminados(respuesta_json)
        
        return respuesta_json
    
    def _completar_valores_predeterminados(self, respuesta_json):
        """
        Completa con valores predeterminados las listas que no tienen suficientes elementos.
        """
        # Para cada tipo de lista, completar hasta el mínimo requerido
        for campo, min_valor in self.min_elementos.items():
            while len(respuesta_json[campo]) < min_valor:
                # Obtener valores predeterminados para este campo
                valores = self.valores_por_defecto.get(campo, ["Información no disponible"])
                # Agregar el siguiente valor predeterminado que no esté ya en la lista
                for valor in valores:
                    if valor not in respuesta_json[campo]:
                        respuesta_json[campo].append(valor)
                        break