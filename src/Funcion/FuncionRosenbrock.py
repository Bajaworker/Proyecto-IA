import numpy as np
from src.Funcion.Base import Funcion

class FuncionRosenbrock(Funcion):
    def __init__(self, MatrizDiseño=None,Datos=None):
        super().__init__(MatrizDiseño, Datos)

    def rosenbrock_n_z_numerico(self,X):
        n = len(X)

        if n % 2 != 0:
            raise ValueError("El tamaño de X debe ser par para la función de Rosenbrock.")

        z = np.zeros(n)

        for i in range(n // 2):
            xi = X[2 * i]
            xi1 = X[2 * i + 1]
            z[2 * i] = 10 * (xi1 - xi ** 2)
            z[2 * i + 1] = 1 - xi
        return z

    def matrizJacobienaConPatron(self,X):
        n = len(X)
        if n % 2 != 0:
            raise ValueError("El tamaño de X debe ser par para la función de Rosenbrock.")
        matriz = np.zeros((n, n))
        for i in range(n // 2):
            matriz[2 * i, 2 * i] = -20 * X[2 * i]
            matriz[2 * i, 2 * i + 1] = 10
            matriz[2 * i + 1, 2 * i] = -1
        return matriz

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        z = self.rosenbrock_n_z_numerico(theta)
        funcion_objetivoVector = z * z.T
        suma = 0
        for i in range(len(funcion_objetivoVector)):
            suma = suma + funcion_objetivoVector[i]
        return suma


    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        z = self.rosenbrock_n_z_numerico(theta)
        y = self.matrizJacobienaConPatron(theta).T
        gradiente_P = 2 * (y @ z)
        return gradiente_P