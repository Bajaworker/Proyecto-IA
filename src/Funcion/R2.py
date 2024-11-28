import numpy as np

from src.Funcion.Base import Funcion

class R2(Funcion):
    def __init__(self,MatrizDiseño,Datos):
        super().__init__(MatrizDiseño, Datos)

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        Y = self.Datos.getY()
        Yp=self.MatrizDiseño.getMatrizDiseño()@theta
        Ym = np.mean(Y)
        SSE=np.sum((Y-Yp)**2)
        SST = np.sum((Y-Ym)**2)
        R2=1-(SSE/SST)
        return R2

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        return "Todavia no esta definido"