import numpy as np

from src.algorithms.Base import Optimizador


class AlgorithmAdagrad(Optimizador):
    def __init__(self, theta, funcion, tasaDeAprendizaje, Datos, lr_decay=0, peso_decay=0, epsilon=1e-8, epoca=1000,
                 steps=100):
        super().__init__(theta, funcion, tasaDeAprendizaje, Datos)
        self.lr_decay = lr_decay
        self.peso_decay = peso_decay
        self.epsilon = epsilon
        self.epoca = epoca
        self.steps = steps

    def _actualizar_parametros(self, g_t, state_sum, lr_t):
        state_sum += g_t ** 2
        self.theta -= (lr_t / (np.sqrt(state_sum) + self.epsilon)) * g_t
        return state_sum

    def _calcular_tasa_aprendizaje(self, iteracion):
        return self.tasaDeAprendizaje / (1 + (iteracion - 1) * self.lr_decay)

    def optimizar(self, modo="lote", tamañoDeLote=None):
        state_sum = self.getValuesInitStateSum()
        X = self.Datos.getX()
        Y = self.Datos.getY()
        q = X.shape[0]

        for t in range(1, self.epoca + 1):
            if modo == "lote":
                g_t = self.funcion.gradiente(self.theta)
                lr_t = self._calcular_tasa_aprendizaje(t)
                if self.peso_decay != 0:
                    g_t += self.peso_decay * self.theta
                state_sum = self._actualizar_parametros(g_t, state_sum, lr_t)

            elif modo == "mini-lote":
                if tamañoDeLote is None:
                    raise ValueError("Se requiere tamañoDeLote para el modo 'mini-lote'")
                indices = np.random.permutation(q)
                X, Y = self.Datos.obtenerMiniLote(indices)
                n_minilote = q // tamañoDeLote

                for j in range(n_minilote):
                    X_Batch = X[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                    Y_Batch = Y[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                    g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)
                    lr_t = self._calcular_tasa_aprendizaje(t)
                    if self.peso_decay != 0:
                        g_t += self.peso_decay * self.theta
                    state_sum = self._actualizar_parametros(g_t, state_sum, lr_t)

                if q % tamañoDeLote != 0:
                    X_Batch = X[n_minilote * tamañoDeLote:, :]
                    Y_Batch = Y[n_minilote * tamañoDeLote:, :]
                    g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)
                    lr_t = self._calcular_tasa_aprendizaje(t)
                    if self.peso_decay != 0:
                        g_t += self.peso_decay * self.theta
                    state_sum = self._actualizar_parametros(g_t, state_sum, lr_t)

            elif modo == "online":
                for j in range(q):
                    X_Batch = X[j:j + 1, :]
                    Y_Batch = Y[j:j + 1, :]
                    g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)
                    lr_t = self._calcular_tasa_aprendizaje(t)
                    if self.peso_decay != 0:
                        g_t += self.peso_decay * self.theta
                    state_sum = self._actualizar_parametros(g_t, state_sum, lr_t)

            else:
                raise ValueError("Modo desconocido: debe ser 'lote', 'mini-lote' o 'online'")

            if t % self.steps == 0:
                print({
                    "iteracion": t,
                    "theta": self.theta,
                    "state_sum": state_sum,
                })

        return self.theta
