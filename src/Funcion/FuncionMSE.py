import numpy as np

from src.Funcion.Base import Funcion

class FuncionMSE(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion):
        super().__init__(MatrizDiseño,Datos)
        self.Funcion=Funcion

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        error=self.Funcion.ejecutarFuncion(theta,X_Batch,Y_Batch)
        SSE=np.sum(error**2)
        if X_Batch is not None and Y_Batch is not None:
            r, c = Y_Batch.shape
        else:
            r, c = self.Datos.renglonColumnaDeY()
        MSE=SSE/(r*c)
        return MSE

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        error = self.Funcion.ejecutarFuncion(theta,X_Batch,Y_Batch)
        if X_Batch is not None and Y_Batch is not None:
            r, c = Y_Batch.shape
            matrizMinilote=self.MatrizDiseño.getMatrizDiseñoMiniLote(X_Batch)
            gt = (-2 * (matrizMinilote.T @ error)) / (r * c)
        else:
            r, c = self.Datos.renglonColumnaDeY()
            gt = (-2 * (self.MatrizDiseño.matrixDiseño.T @ error))/(r*c)
        return gt