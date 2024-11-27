from abc import ABC, abstractmethod
import numpy as np

class Modelo(ABC):
    def __init__(self, DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta,FuncionError):
        self.DatosE = DatosE
        self.DatosT = DatosT
        self.Metrica = Metrica
        self.Optimizador = Optimizador
        self.MatrizDiseñoE = MatrizDiseñoE
        self.MatrizDiseñoT=MatrizDiseñoT
        self.theta = theta
        self.FuncionError = FuncionError

    @abstractmethod
    def predecir(self, theta):
        pass

    @abstractmethod
    def entrenar(self,theta):
        pass

