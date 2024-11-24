import numpy as np

from src.algorithms.Base import Optimizador


class AlgorithmAdagrad(Optimizador):
    def __init__(self, theta, funcion, tasaDeAprendizaje, lr_decay=0, peso_decay=0, epsilon=1e-8, epoca=1000, steps=100):
        super().__init__(theta, funcion, tasaDeAprendizaje)
        self.lr_decay = lr_decay
        self.peso_decay = peso_decay
        self.epsilon = epsilon
        self.epoca = epoca
        self.steps = steps

    def getValuesInitStateSum(self):
        dimensiones = self.theta.shape
        if len(dimensiones) == 1:
            return np.zeros(dimensiones[0], dtype=np.float64)
        return np.zeros((dimensiones[0], dimensiones[1]), dtype=np.float64)

    def optimizar(self):
        state_sum = self.getValuesInitStateSum()

        for t in range(1, self.epoca + 1):
            # Calcula el gradiente
            g_t = self.funcion.gradiente(self.theta)

            # Ajusta la tasa de aprendizaje con decaimiento
            lr_t = self.tasaDeAprendizaje / (1 + (t - 1) * self.lr_decay)

            # Aplica el peso de regularización
            if self.peso_decay != 0:
                g_t += self.peso_decay * self.theta

            # Actualiza la suma acumulativa de gradientes al cuadrado
            state_sum += g_t ** 2

            # Actualiza los parámetros con Adagrad
            self.theta -= (lr_t / (np.sqrt(state_sum) + self.epsilon)) * g_t

            # Imprime resultados cada `self.steps` iteraciones
            if t % self.steps == 0:
                print({
                    "iteracion": t,
                    "theta": self.theta,
                    "gradiente": g_t,
                    "state_sum": state_sum,
                })

        return self.theta
