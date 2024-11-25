from src.interface.options import direccion_carpetas_datos,options_metricas_desempenio

class Interface:
    def __init__(self):
        self.welcome_message()
        
        self.modelo_seleccionado = None
        self.algorithms_seleccionado = None
        self.tecnica_regularizacion = None
        self.forma_aprendizaje = None
        self.metrica_desempenio = None
        self.carperta_datos = None
        self.url_datos = None
        self.tasa_entrenamiento = 0.1

        self.configuration_menu = [
            {
                "id":"MODELO",
                "title":"Selecciona el modelo a usar",
                "options":[
                    {
                        "id":"REGRESION",
                        "text":"Regresion"
                    },
                    {
                        "id":"CLASIFICACION",
                        "text":"Clasificación"
                    }
                ],
                "get_option":self.getOption,
                "set_value":self.setValueModelo,
                "get_value":self.getValueModelo
            },
            {
                "id":"ALGORITMO",
                "title":"Selecciona el Algoritmo a usar",
                "options":[
                    {
                        "id":"ADAGRAD",
                        "text":"Adagrad",
                    },
                    {
                        "id":"SGD_U_CLIP",
                        "text":"SGD U-CLIP",
                    },
                ],
                "get_option":self.getOption,
                "set_value":self.setValueAlgoritmo,
                "get_value":self.getValueAlgoritmo
            },
            {
                "id":"TECNICA_REGULARIZACION",
                "title":"Selecciona la tecnica de regularización",
                "options":[
                    {
                        "id":"L1",
                        "text":"L1",
                    },
                    {
                        "id":"L2",
                        "text":"L2",
                    },
                ],
                "get_option":self.getOption,
                "set_value":self.setValueTecnicaRegularizacion,
                "get_value":self.getValueTecnicaRegularizacion
            },
            {
                "id":"FORMA_DE_APRENDIZAJE",
                "title":"Selecciona la formas de aprendizaje",
                "options":[
                    {
                        "id":"ONLINE",
                        "text":"online",
                    },
                    {
                        "id":"BATCH",
                        "text":"batch",
                    },
                    {
                        "id":"MINI_BATCH",
                        "text":"minibatch",
                    }
                ],
                "get_option":self.getOption,
                "set_value":self.setValueFormaAprendizaje,
                "get_value":self.getValueFormaAprendizaje
            },
            {
                "id":"METRICA_DESEMPENIO",
                "title":"Selecciona la métrica de desempeño",
                "options":self.getOptionsMetricasDesempenio,
                "get_option":self.getOption,
                "set_value":self.setValueMetricaDesempenio,
                "get_value":self.getValueMetricaDesempenio
            },
            {
                "id":"CARPETA_DATOS",
                "title":"Selecciona el tipo de dato",
                "options":self.getOptionsCarpetaDatos,
                "get_option":self.getOption,
                "set_value":self.setValueCarpetaDatos,
                "get_value":self.getValueCarpetaDatos
            },
            {
                "id":"URL_DATOS",
                "title":"Selecciona el tipo de archivo",
                "options":self.getOptionsUrlDatos,
                "get_option":self.getOption,
                "set_value":self.setValueUrlDatos,
                "get_value":self.getValueUrlDatos
            }
        ]

    def setValueModelo(self,value):
        self.modelo_seleccionado = value

    def setValueAlgoritmo(self,value):
        self.algorithms_seleccionado = value

    def setValueTecnicaRegularizacion(self,value):
        self.tecnica_regularizacion = value
    
    def setValueFormaAprendizaje(self,value):
        self.forma_aprendizaje = value

    def setValueMetricaDesempenio(self,value):
        self.metrica_desempenio = value
    
    def setValueCarpetaDatos(self,value):
        self.carperta_datos = value
    
    def setValueUrlDatos(self,value):
        self.url_datos = value
    
    def getValueModelo(self):
        return self.modelo_seleccionado
    
    def getValueAlgoritmo(self):
        return self.algorithms_seleccionado

    def getValueTecnicaRegularizacion(self):
        return self.tecnica_regularizacion
    
    def getValueFormaAprendizaje(self):
        return self.forma_aprendizaje

    def getValueMetricaDesempenio(self):
        return self.metrica_desempenio

    def getValueCarpetaDatos(self):
        return self.carperta_datos

    def getValueUrlDatos(self):
        return self.url_datos

    def getOptionsMetricasDesempenio(self):
        options = options_metricas_desempenio

        return options[self.modelo_seleccionado]
    
    def getOptionsCarpetaDatos(self):
        options = direccion_carpetas_datos
        return list(options[self.modelo_seleccionado].values())

    def getOptionsUrlDatos(self):
        options = direccion_carpetas_datos
        archivos = options[self.modelo_seleccionado][self.carperta_datos]["options"]
        return archivos
    
    def getOptionsSection(self,value_section):
        options = value_section["options"]

        if not isinstance(options, list):
            options = options()
        
        options = options[:] #Copia del arreglo
        
        if value_section["id"] == "MODELO":
            options.append({
                "id":"SALIR",
                "text":"Salir"
            })
        else:
            options.append({
                "id":"REGRESAR",
                "text":"Regresar"
            })
        
        return options
    
    def getVariablesSeleccionadas(self):
        print("\n")
        datos = {
            "MODELO":self.modelo_seleccionado,
            "ALGORITMO":self.algorithms_seleccionado,
            "TECNICA_DE_REGULARIZACION":self.tecnica_regularizacion,
            "FORMA_DE_APRENDIZAJE":self.forma_aprendizaje,
            "METRICA_DE_DESEMPENIO":self.metrica_desempenio,
            "CAPERTA_DE_DATOS":self.carperta_datos,
            "URL_DE_DATOS":self.url_datos,
            "TASA_DE_ENTRENAMIENTO":self.tasa_entrenamiento
        }
        print("************************************")
        print("\n")
        print("DATOS SELECCIONADOS: ")
        print("\n")
        print("************************************")
        print("\n")

        for index,value in datos.items():
            print(index+" : "+str(value))
        print("\n")
        return datos
    
    def showOptions(self,options):
        for index_option,value_option in enumerate(options):
            text = str(index_option+1) + ":" + value_option["text"]
            print(text)
        return True

    def init_interface(self):
        status_repeat = False
        print("\n")

        for index_section,value_section in enumerate(self.configuration_menu):
            
            if value_section["get_value"]() is not None:
                continue

            print(value_section["title"].upper())

            options = self.getOptionsSection(value_section)

            self.showOptions(options)
            
            id_seleccionado = value_section["get_option"](options)

            if id_seleccionado is None:
                status_repeat = True
                break
            
            if id_seleccionado == "REGRESAR":
                self.configuration_menu[index_section-1]["set_value"](None)
                value_section["set_value"](None)
                status_repeat = True
                break

            if id_seleccionado == "SALIR":
                status_repeat = False
                self.ending_message()
                break

            value_section["set_value"](id_seleccionado)

        if status_repeat:
            self.init_interface()
        
        
        return True

    def getOption(self,options):
        option = int(input("Seleccione una opción: "))
        print("\n")
      
        longitud_options = len(options)

        if option <= 0 or option > longitud_options:
            return None
        
        id_option = options[int(option)-1]["id"]

        return id_option
    
    def welcome_message(self):
        message = """**********************************\n*\n*\n*    ¡PROYECTO DE IA!\n*\n*\n**********************************"""
        print(message)
    
    def ending_message(self):
        message = """**********************************\n*\n*\n*    ¡PROCESO TERMINADO!\n*\n*\n**********************************"""

        print(message)

