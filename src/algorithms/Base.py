from abc import ABC, abstractmethod
import numpy as np

class Optimizador(ABC):
    def __init__(self, theta, funcion, tasaDeAprendizaje):
        self.theta = np.array(theta, dtype=np.float64)
        self.funcion = funcion
        self.tasaDeAprendizaje = tasaDeAprendizaje

    @abstractmethod
    def optimizar(self):
        pass