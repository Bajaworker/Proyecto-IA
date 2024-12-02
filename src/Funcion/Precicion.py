from scipy.special import softmax
from sklearn.metrics import r2_score
from src.Funcion.Base import Funcion
import numpy as np


class Precicion(Funcion):
    def __init__(self,MatrizDiseño,Datos):
        super().__init__(MatrizDiseño, Datos)

    def ejecutarFuncion(self,theta,X=None,Y=None):
        y = self.Datos.getY()
        Y=self.to_classlabel(y)
        matriz=self.MatrizDiseño.getMatrizDiseño()
        Z=-matriz@theta
        P=softmax(Z,axis=1)
        Yp=self.to_classlabel(P)
        precicion = np.sum(Y == Yp) / len(Y)
        return precicion

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        return "Todavia no esta definido"

    def to_classlabel(self,z):
        return z.argmax(axis=1)