import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from src.reading.index import ReadingDataSets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#falta agragar
class EstructuraDatos(ReadingDataSets):

    def __init__(self, ruta, estructura_datos, porcentaje, inversar, delimiter):
        super().__init__(delimiter)

        self.ruta = ruta
        self.estructura_datos = estructura_datos
        self.porcentaje = porcentaje
        self.inversar = inversar

        self.X = None
        self.Y = None

        self.Columns_X=None
        self.Columns_Y=None

        self.Columns_X=None
        self.Columns_Y=None

    def definirXY(self):
        try:
            dato = self.reading(self.ruta)

            type = self.estructura_datos["type"]

            match type:
                case "TABLE_DEFAULT":
                    self.getDataTableDefault(dato=dato["data"], columns_x=self.estructura_datos["columns_x"],
                                             columns_y=self.estructura_datos["columns_y"])
                case "TABLE_SPLIT":
                    self.getDataTableSplit(data_x=dato["x"], data_y=dato["y"])

                case "TABLE_DEFAULT_BINARIO":
                    self.definirXYBinario(
                        colIniciaX=self.estructura_datos["columns_x"][0],
                        colFinalX=self.estructura_datos["columns_x"][1],
                        colIniciaY=self.estructura_datos["columns_y"][0],
                        colFinalY=self.estructura_datos["columns_y"][1],
                        tipoSeparacion=",",
                        funcion_matrix=self.to_binary_matrix_binario
                    )
                case "TABLE_DEFAULT_NUMERACION":
                    self.definirXYBinario(
                        colIniciaX=self.estructura_datos["columns_x"][0],
                        colFinalX=self.estructura_datos["columns_x"][1],
                        colIniciaY=self.estructura_datos["columns_y"][0],
                        colFinalY=self.estructura_datos["columns_y"][1],
                        tipoSeparacion=" ",
                        funcion_matrix=self.to_binary_matrix
                    )
                case _:
                    print("LA ESTRUCTURA NO ESTA DEFINIDA EN EL CASE")
                    sys.error()
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")

    def definirXYBinario(self,colIniciaX,colFinalX,colIniciaY,colFinalY,tipoSeparacion,funcion_matrix):
        try:
            dato = np.loadtxt(self.ruta, delimiter=tipoSeparacion)
            X=dato[:,colIniciaX:colFinalX]
            Y=dato[:,colIniciaY:colFinalY]
            inputs_train, inputs_test, targets_train, targets_test = train_test_split(X,Y,random_state=1, test_size=1-self.porcentaje)

            if self.inversar == 0:
                # Datos de entrenamiento
                self.X = inputs_train
                self.Y = funcion_matrix(targets_train)
            elif self.inversar == 1:
                # Datos de prueba
                self.X = inputs_test
                self.Y = funcion_matrix(targets_test)
            else:
                raise ValueError("El parámetro 'inversar' debe ser 0 (entrenamiento) o 1 (prueba).")
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")

    def getDataTableDefaultBinario(self, dato, columns_x, columns_y):
        datosTomado = int(dato.shape[0] * self.porcentaje)

        if self.inversar == 0:
            # Datos de entrenamiento
            self.X = dato[:datosTomado, columns_x[0]:columns_x[1]]
            self.Y = self.to_binary_matrix_binario(dato[:datosTomado, columns_y[0]:columns_y[1]])
        else:
            # Datos de prueba
            self.X = dato[datosTomado:, columns_x[0]:columns_x[1]]
            self.Y = self.to_binary_matrix_binario(dato[datosTomado:, columns_y[0]:columns_y[1]])
        
        self.Columns_X=dato[:, columns_x[0]:columns_x[1]]
        self.Columns_Y=dato[:, columns_y[0]:columns_y[1]]

    def getDataTableDefault(self, dato, columns_x, columns_y):
        datosTomado = int(dato.shape[0] * self.porcentaje)

        if self.inversar == 0:
            # Datos de entrenamiento
            self.X = dato[:datosTomado, columns_x[0]:columns_x[1]]
            self.Y = dato[:datosTomado, columns_y[0]:columns_y[1]]
        else:
            # Datos de prueba
            self.X = dato[datosTomado:, columns_x[0]:columns_x[1]]
            self.Y = dato[datosTomado:, columns_y[0]:columns_y[1]]
        
        self.Columns_X=dato[:, columns_x[0]:columns_x[1]]
        self.Columns_Y=dato[:, columns_y[0]:columns_y[1]]

    def getDataTableSplit(self, data_x, data_y):
        datosTomado = int(data_x.shape[0] * self.porcentaje)

        if self.inversar == 0:
            # Datos de entrenamiento
            self.X = data_x[:datosTomado]
            self.Y = data_y[:datosTomado]
        else:
            # Datos de prueba
            self.X = data_x[datosTomado:]
            self.Y = data_y[datosTomado:]

        self.Columns_X=data_x
        self.Columns_Y=data_y


    def to_binary_matrix(self,z):
        # Aseguramos que los elementos de z sean de tipo float
        z = z.astype(float)
        
        # Convertimos los elementos de z a enteros para usar como índices
        z = z.astype(int)
        
        # Encuentra el valor máximo en z para determinar el número de columnas
        max_value = np.max(z)
        
        # Crea la matriz binaria con el tamaño adecuado
        binary_matrix = np.zeros((z.shape[0], max_value), dtype=int)
        
        # Llena la matriz binaria
        for i, value in enumerate(z):
            binary_matrix[i, value - 1] = 1  # Ajusta el índice para usar en el array
        
        return binary_matrix
    
    def to_binary_matrix_binario(self,z):
        # Aseguramos que los elementos de z sean de tipo float
        z = z.astype(float)
        
        # Convertimos los elementos de z a enteros para usar como índices
        z = z.astype(int)
        
        # Encuentra el valor máximo en z para determinar el número de columnas
        max_value = np.max(z)
        
        # Crea la matriz binaria con el tamaño adecuado
        binary_matrix = np.zeros((z.shape[0], max_value + 1), dtype=int)
        
        # Llena la matriz binaria
        for i, value in enumerate(z):
            if value >= 0:  # Asegura que solo los valores positivos y cero se consideren
                binary_matrix[i, value] = 1  # Ajusta el índice para usar en el array
        
        return binary_matrix

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def setX(self,X):
        self.X=X

    def setY(self,Y):
        self.Y=Y
    
    def getAllColumnsX(self):
        return self.Columns_X

    def getAllColumnsY(self):
        return self.Columns_Y

    def normalizarDatosX(self, ymin, ymax):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")

        X_min = self.X.min(axis=0)
        X_max = self.X.max(axis=0)
        self.X = ((ymax - ymin) * (self.X - X_min) / (X_max - X_min)) + ymin
        return self.X

    def desNormalizarDatosX(self, X_min, X_max, ymin, ymax):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")

        self.X = ((X_max - X_min) * (self.X - ymin) / (ymax - ymin)) + X_min
        return self.X

    def tamañoDeDatos(self):
        try:
            if self.X is None and self.Y is None:
                dato = np.loadtxt(self.ruta, delimiter=self.delimiter)
                return dato.shape
            else:
                return (self.X.shape[0], self.X.shape[1] + self.Y.shape[1])
        except Exception as e:
            print(f"Error al calcular el tamaño de los datos: {e}")
            return None

    def renglonColumnaDeY(self):
        return self.Y.shape

    def obtenerMiniLote(self, indices):
        if self.X is None or self.Y is None:
            raise ValueError("Los datos no están definidos. Usa definirXY() primero.")
        return self.X[indices, :], self.Y[indices, :]

    def setPorcentajeDedatos(self, nuevo_porcentaje):

        if nuevo_porcentaje <= 0 or nuevo_porcentaje > 1:
            raise ValueError("El porcentaje debe ser un valor entre 0 y 1.")
        self.porcentaje = nuevo_porcentaje