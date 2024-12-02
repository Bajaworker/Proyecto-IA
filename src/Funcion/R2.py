import numpy as np
from sklearn.metrics import r2_score
from src.Funcion.Base import Funcion

#Falta mejorar
class R2(Funcion):
    def __init__(self,MatrizDiseño,Datos):
        super().__init__(MatrizDiseño, Datos)

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        Y = self.Datos.getY()
        Yp=self.MatrizDiseño.getMatrizDiseño()@theta
        R2=r2_score(Y.reshape(-1,1),Yp.reshape(-1,1))
        return R2

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        return "Todavia no esta definido"



#def __init__(self,MatrizDiseño,Datos,yp):

#    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        #Y = self.Datos.getY()
        #Yp=self.yp
        #R2=r2_score(Y.reshape(-1,1),Yp.reshape(-1,1))
        #return R2