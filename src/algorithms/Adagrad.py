import numpy as np

from src.algorithms.Base import Optimizador


class AlgorithmAdagrad(Optimizador):
    def __init__(self, theta, funcion, tasaDeAprendizaje,Datos, lr_decay=0, peso_decay=0, epsilon=1e-8, epoca=1000, steps=100):
        super().__init__(theta, funcion, tasaDeAprendizaje,Datos)
        self.lr_decay = lr_decay
        self.peso_decay = peso_decay
        self.epsilon = epsilon
        self.epoca = epoca
        self.steps = steps

    def optimizarLote(self):
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

    def optimizarConMiniLote(self,tamañoDeLote):
        state_sum = self.getValuesInitStateSum()
        X=self.Datos.getX()
        q=X.shape[0]
        n_minilote=q//tamañoDeLote

        for i in range(1,self.epoca + 1):
            indices=np.random.permutation(q)
            X,Y=self.Datos.obtenerMiniLote(indices)


            for j in range(n_minilote):
                X_Batch = X[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                Y_Batch = Y[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                g_t=self.funcion.gradiente(self.theta,X_Batch,Y_Batch)

                lr_t = self.tasaDeAprendizaje / (1 + (i - 1) * self.lr_decay)

                if self.peso_decay != 0:
                    g_t += self.peso_decay * self.theta

                state_sum += g_t ** 2
                self.theta -= (lr_t / (np.sqrt(state_sum) + self.epsilon)) * g_t

            if q % tamañoDeLote != 0:
                X_Batch = X[n_minilote * tamañoDeLote:, :]
                Y_Batch = Y[n_minilote * tamañoDeLote:, :]
                g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)

                lr_t = self.tasaDeAprendizaje / (1 + (i - 1) * self.lr_decay)

                if self.peso_decay != 0:
                    g_t += self.peso_decay * self.theta

                state_sum += g_t ** 2
                self.theta -= (lr_t / (np.sqrt(state_sum) + self.epsilon)) * g_t

        return self.theta


    def optimizarOnline(self):
        state_sum = self.getValuesInitStateSum()
        X = self.Datos.getX()
        Y = self.Datos.getY()
        q = X.shape[0]
        for i in range(1,self.epoca + 1):
            for j in range(q):
                X_Batch = X[j:j+1,:]
                Y_Batch = Y[j:j+1,:]
                g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)

                lr_t = self.tasaDeAprendizaje / (1 + (i - 1) * self.lr_decay)

                if self.peso_decay != 0:
                    g_t += self.peso_decay * self.theta

                state_sum += g_t ** 2
                self.theta -= (lr_t / (np.sqrt(state_sum) + self.epsilon)) * g_t

            return self.theta










