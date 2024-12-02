from abc import ABC, abstractmethod



class Modelo(ABC):
    def __init__(self, DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta):
        self.DatosE = DatosE
        self.DatosT = DatosT
        self.Metrica = Metrica
        self.Optimizador = Optimizador
        self.MatrizDiseñoE = MatrizDiseñoE
        self.MatrizDiseñoT=MatrizDiseñoT
        self.theta = theta

    @abstractmethod
    def predecir(self, theta):
        pass

    @abstractmethod
    def entrenar(self,theta):
        pass

