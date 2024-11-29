import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from src.reading.index  import ReadingDataSets

class EstructuraDatos(ReadingDataSets):

    def __init__(self,ruta,porcentaje,inversar,delimiter):
        super().__init__(delimiter)

        self.ruta = ruta
        self.porcentaje=porcentaje
        self.inversar = inversar

        self.X=None
        self.Y=None

    def definirXY(self):
        #DATOS DE PRUEBA, FALTA AGREGAR COLUMNS POR ARCHIVO
        colIniciaX = 0
        colFinalX = 8
        colIniciaY = 8
        colFinalY = None

        try:
            dato = self.reading(self.ruta)

            dato = dato["data"]

            datosTomado = int(dato.shape[0] * self.porcentaje)

            if self.inversar == 0:
                # Datos de entrenamiento
                self.X = dato[:datosTomado, colIniciaX:colFinalX]
                self.Y = dato[:datosTomado, colIniciaY:colFinalY]
            elif self.inversar == 1:
                # Datos de prueba
                self.X = dato[datosTomado:, colIniciaX:colFinalX]
                self.Y = dato[datosTomado:, colIniciaY:colFinalY]
            else:
                raise ValueError("El parámetro 'inversar' debe ser 0 (entrenamiento) o 1 (prueba).")
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def normalizarDatosX(self,ymin,ymax):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")

        X_min=self.X.min(axis=0)
        X_max=self.X.max(axis=0)
        self.X=((ymax - ymin) * (self.X - X_min) / (X_max - X_min)) + ymin
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