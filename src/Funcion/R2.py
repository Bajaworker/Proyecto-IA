import numpy as np

from src.Funcion.Base import Funcion

class R2(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion):
        super().__init__(MatrizDiseño, Datos)
        self.Funcion = Funcion

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        SSE=self.ejecutarFuncion(theta,X_Batch,Y_Batch)
        Y=self.Datos.getY()
        Ym=np.mean(Y)
        SST = np.sum((Y-Ym)**2)
        R2=1-(SSE/SST)
        return R2

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        return "Todavia no esta definido"