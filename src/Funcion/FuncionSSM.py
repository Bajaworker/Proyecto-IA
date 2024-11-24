import numpy as np

from src.Funcion.Base import Funcion

class FuncionSSM(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion):
        super().__init__(MatrizDiseño,Datos)
        self.Funcion=Funcion

    def ejecutarFuncion(self,theta):
        error=self.Funcion.ejecutarFuncion(theta)
        SSE=np.sum(error**2)
        return SSE

    def gradiente(self,theta):
        error = self.Funcion.ejecutarFuncion(theta)
        gt = 2 * (self.MatrizDiseño.matrixDiseño.T @ error)
        return gt