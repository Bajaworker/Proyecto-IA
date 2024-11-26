import numpy as np
from src.Funcion.Base import Funcion

class FuncionRosenbrock(Funcion):
    def __init__(self, MatrizDiseño=None,Datos=None):
        super().__init__(MatrizDiseño, Datos)

    def ejecutarFuncion(self, theta, X_Batch=None, Y_Batch=None):
        z = self.rosenbrock_n_z_numerico(theta.flatten())  # Asegúrate de usar un vector plano
        return np.sum(z**2)  # Suma de cuadrados de los valores de z

    def gradiente(self, theta, X_Batch=None, Y_Batch=None):
        return self.gradientePatron(theta.flatten()).reshape(theta.shape)

    def rosenbrock_n_z_numerico(self, X):
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

    def gradientePatron(self, X):
        z = self._rosenbrock_n_z_numerico(X)
        y = self._matrizJacobienaConPatron(X).T
        gradiente_P = 2 * (y @ z)
        return gradiente_P

    def matrizJacobienaConPatron(self, X):
        n = len(X)
        if n % 2 != 0:
            raise ValueError("El tamaño de X debe ser par para la función de Rosenbrock.")
        matriz = np.zeros((n, n))
        for i in range(n // 2):
            matriz[2 * i, 2 * i] = -20 * X[2 * i]
            matriz[2 * i, 2 * i + 1] = 10
            matriz[2 * i + 1, 2 * i] = -1
        return matriz
