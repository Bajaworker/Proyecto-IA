from abc import ABC, abstractmethod
import numpy as np

class Optimizador(ABC):
    def __init__(self, theta, funcion, tasaDeAprendizaje,Datos):
        self.theta = np.array(theta, dtype=np.float64)
        self.funcion = funcion
        self.tasaDeAprendizaje = tasaDeAprendizaje
        self.Datos=Datos

    def getValuesInitStateSum(self):
        dimensiones = self.theta.shape
        if len(dimensiones) == 1:
            return np.zeros(dimensiones[0], dtype=np.float64)
        return np.zeros((dimensiones[0], dimensiones[1]), dtype=np.float64)

