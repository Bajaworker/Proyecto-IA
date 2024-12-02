from src.Funcion.Base import Funcion
import numpy as np
from scipy.special import softmax

class EntropiaCruzadaBinaria(Funcion):
    def __init__(self,MatrizDiseño,Datos,mu):
        super().__init__(MatrizDiseño,Datos)
        self.mu = mu

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        if X_Batch is not None or Y_Batch is not None:
            matrizBatch = self.MatrizDiseño.getMatrizDiseñoMiniLote(X_Batch)
            Z=-matrizBatch@theta
            N=X_Batch.shape[0]
            y_batch=self.to_classlabel(Y_Batch)
            Y=self.one_hot_encode(y_batch)
            entropia = 1 / N * (np.trace(matrizBatch @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
            return entropia

        matrizDiseño=self.MatrizDiseño.getMatrizDiseño()
        Z=-matrizDiseño@theta
        N=self.Datos.getX().shape[0]
        y1=self.Datos.getY()
        y2=self.to_classlabel(y1)
        Y=self.one_hot_encode(y2)
        entropia = 1 / N * (np.trace(matrizDiseño @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return entropia

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        if X_Batch is not None or Y_Batch is not None:
            matrizBatch = self.MatrizDiseño.getMatrizDiseñoMiniLote(X_Batch)
            Z = -matrizBatch @ theta
            P=softmax(Z,axis=1)
            N = X_Batch.shape[0]
            y_batch=self.to_classlabel(Y_Batch)
            Y = self.one_hot_encode(y_batch)
            gd = 1/N * (matrizBatch.T @ (Y - P)) + 2 * self.mu * theta
            return gd
        matrizDiseño=self.MatrizDiseño.getMatrizDiseño()
        Z=-matrizDiseño@theta
        P=softmax(Z,axis=1)
        N=self.Datos.getX().shape[0]
        y1=self.Datos.getY()
        y2=self.to_classlabel(y1)
        Y=self.one_hot_encode(y2)
        gd = 1 / N * (matrizDiseño.T @ (Y - P)) + 2 * self.mu * theta
        return gd

    def to_classlabel(self,z):
        return z.argmax(axis=1)


    def one_hot_encode(self,y):
        n_class = np.unique(y).shape[0]
        y_encode = np.zeros((y.shape[0], n_class))
        for idx, val in enumerate(y):
            y_encode[idx, val] = 1.0
        return y_encode

