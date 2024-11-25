import numpy as np

from src.Funcion.Base import Funcion

class FuncionSSM(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion):
        super().__init__(MatrizDiseño,Datos)
        self.Funcion=Funcion

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        error=self.Funcion.ejecutarFuncion(theta,X_Batch,Y_Batch)
        SSE=np.sum(error**2)
        return SSE

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        if X_Batch is not None:
            self.MatrizDiseño.setMatrizDiseño(X_Batch)
            matrizDiseño = self.MatrizDiseño.getMatrizDiseño()
        else:
            matrizDiseño = self.MatrizDiseño.getMatrizDiseño()

        error = self.Funcion.ejecutarFuncion(theta, X_Batch, Y_Batch)
        gt = -2 * (matrizDiseño.T @ error)
        return gt