import numpy as np

from src.Funcion.Base import Funcion

class FuncionRidge(Funcion):
    def __init__(self,MatrizDiseño,Datos,Funcion,landa):
        super().__init__(MatrizDiseño,Datos)
        self.Funcion=Funcion
        self.landa=landa

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        MSE = self.Funcion.ejecutarFuncion(theta, X_Batch, Y_Batch)
        if X_Batch is not None:
            r, c = Y_Batch.shape
        else:
            r, c = self.Datos.renglonColumnaDeY()

        theta2 = theta[:,1:]
        ridge_penalizacion = (self.landa / (r * c)) * (theta2.T @ theta2)
        ridge = MSE + ridge_penalizacion
        return ridge


    def gradiente(self, theta, X_Batch=None, Y_Batch=None):
        gradienteMSE = self.Funcion.gradiente(theta, X_Batch, Y_Batch)
        if X_Batch is not None:
            r, c = Y_Batch.shape
        else:
            r, c = self.Datos.renglonColumnaDeY()
        theta2 = theta[:, 1:]

        gradiente_penalizacion = (2 * self.landa / (r * c)) * theta2

        # Crear gradiente completo
        zeros = np.zeros((theta.shape[0], 1))
        penalizacion_completa = np.concatenate((zeros, gradiente_penalizacion), axis=1)
        gt = gradienteMSE + penalizacion_completa
        return gt

