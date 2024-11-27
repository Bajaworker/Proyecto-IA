from src.reading.index import ReadingDataSets
from src.interface.index import Interface
import pandas as pd





class proyectoIA:
    def __init__(self):
        self.class_interface = Interface()
        self.class_interface.init_interface()
        self.variablesSeleccionadas = self.class_interface.getVariablesSeleccionadas()

        self.class_reading = ReadingDataSets()

        #TRAE TODOS LOS DATOS QUE EL USUARIO SELECCIONA EN CONSOLA
        variables_datos = self.class_reading.reading(self.variablesSeleccionadas["URL_DE_DATOS"])

        self.numero_filas = variables_datos["rows"]
        self.numero_columns = variables_datos["colums"]

        #DATOS DEL DOCUMENTO SELECCIONADO
        self.datos = variables_datos["data"]

        #NORMALIZACION DE DATOS
        # self.class_reading.desNormalizarDatosX()
        # self.class_reading.normalizarDatosX()

        self.MODELO = self.variablesSeleccionadas["MODELO"]
        self.ALGORITMO = self.variablesSeleccionadas["ALGORITMO"]
        self.TECNICA_DE_REGULARIZACION = self.variablesSeleccionadas["TECNICA_DE_REGULARIZACION"]

        self.FORMA_DE_APRENDIZAJE = self.variablesSeleccionadas["FORMA_DE_APRENDIZAJE"]
        self.METRICA_DE_DESEMPENIO = self.variablesSeleccionadas["METRICA_DE_DESEMPENIO"]
        self.TASA_DE_ENTRENAMIENTO = self.variablesSeleccionadas["TASA_DE_ENTRENAMIENTO"]
        
        self.CAPERTA_DE_DATOS = self.variablesSeleccionadas["CAPERTA_DE_DATOS"]
        self.URL_DE_DATOS = self.variablesSeleccionadas["URL_DE_DATOS"]

        #AGREGAR ALGORITMO
        if self.MODELO == "ADAGRAD":
            #llamar algoritmo
            pass

        print(self.datos)


initClass = proyectoIA()