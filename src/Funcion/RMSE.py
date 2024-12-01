import numpy as np

from src.Funcion.Base import Funcion

class FuncionRMSE(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion):
        super().__init__(MatrizDiseño,Datos)
        self.Funcion=Funcion

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        MSE=self.Funcion.ejecutarFuncion(theta,X_Batch,Y_Batch)
        RMSE=np.sqrt(MSE)
        return RMSE

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        gtMSE =self.Funcion.gradiente(theta,X_Batch,Y_Batch)
        RMSE=self.ejecutarFuncion(theta,X_Batch,Y_Batch)
        gt=gtMSE*(1/(2*RMSE))
        return gt