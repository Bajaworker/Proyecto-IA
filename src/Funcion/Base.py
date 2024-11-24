from abc import ABC, abstractmethod

class Funcion(ABC):
    def __init__(self,MatrizDiseño,Datos):
        self.Funcion = None
        self.MatrizDiseño=MatrizDiseño
        self.Datos=Datos

    @abstractmethod
    def ejecutarFuncion(self):
        pass

    @abstractmethod
    def gradiente(self):
        pass