from abc import ABC, abstractmethod

class Funcion(ABC):
    def __init__(self,MatrizDiseño,Datos):
        self.Funcion = None
        self.MatrizDiseño=MatrizDiseño
        self.Datos=Datos

    def setDatos(self,Datos):
        self.Datos=Datos


    @abstractmethod
    def ejecutarFuncion(self,theta,X_Batch,Y_Batch):
        pass

    @abstractmethod
    def gradiente(self,theta,X_Batch,Y_Batch):
        pass

    def setMatrizDiseño(self,MatrizDiseño):
        self.MatrizDiseño=MatrizDiseño






