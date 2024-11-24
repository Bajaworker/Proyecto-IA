from abc import ABC, abstractmethod
import numpy as np

class Modelo(ABC):
    def __init__(self, Datos, FuncionObjetivo, Optimizador,MatrizDiseño,theta,FuncionError):
        self.Datos = Datos
        self.FuncionObjetivo = FuncionObjetivo
        self.Optimizador = Optimizador
        self.MatrizDiseño = MatrizDiseño
        self.theta = theta
        self.FuncionError = FuncionError

    @abstractmethod
    def predecir(self, theta):
        pass

    @abstractmethod
    def entrenar(self,theta):
        pass

