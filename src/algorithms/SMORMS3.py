import numpy as np

from src.algorithms.Base import Optimizador

class SMORMS3(Optimizador):
    def __init__(self,theta,funcion,tasaDeAprendizaje,Datos,epsilon=1e-8,epoca=1000,steps=100,tolerancia=1e-16):
        super().__init__(theta,funcion,tasaDeAprendizaje, Datos)
        self.epsilon = epsilon
        self.epoca = epoca
        self.steps = steps
        self.tolerancia = tolerancia

    def optimizar(self, modo="lote", tamañoDeLote=None):
        m_t = self.getValuesInitStateSum()
        v_t=self.getValuesInitStateSum()
        s_t=np.ones_like(self.theta)
        x_t=self.getValuesInitStateSum()
        p_t=self.getValuesInitStateSum()
        X = self.Datos.getX()
        Y = self.Datos.getY()
        q = X.shape[0]

        for t in range(1, self.epoca + 1):
            if modo == "lote":
                g_t = self.funcion.gradiente(self.theta)
                s_t=1+(1-x_t)*s_t
                p_t=1/(s_t+1)
                m_t=(1-p_t)*m_t+(p_t*g_t)
                v_t =(1-p_t)*v_t+(p_t*g_t ** 2)
                x_t=m_t**2/(v_t+self.epsilon)
                paso=(np.minimum(self.tasaDeAprendizaje, x_t)/(np.sqrt(v_t)+self.epsilon))*g_t
                self.theta-=paso
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
                    s_t = 1 + (1 - x_t) * s_t
                    p_t = 1 / (s_t + 1)
                    m_t = (1 - p_t) * m_t + (p_t * g_t)
                    v_t = (1 - p_t) * v_t + (p_t * g_t ** 2)
                    x_t = m_t ** 2 / (v_t + self.epsilon)
                    paso = (np.minimum(self.tasaDeAprendizaje, x_t) / (np.sqrt(v_t) + self.epsilon)) * g_t
                    self.theta -= paso
                if q % tamañoDeLote != 0:
                    X_Batch = X[n_minilote * tamañoDeLote:, :]
                    Y_Batch = Y[n_minilote * tamañoDeLote:, :]
                    g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)
                    s_t = 1 + (1 - x_t) * s_t
                    p_t = 1 / (s_t + 1)
                    m_t = (1 - p_t) * m_t + (p_t * g_t)
                    v_t = (1 - p_t) * v_t + (p_t * g_t ** 2)
                    x_t = m_t ** 2 / (v_t + self.epsilon)
                    paso = (np.minimum(self.tasaDeAprendizaje, x_t) / (np.sqrt(v_t) + self.epsilon)) * g_t
                    self.theta -= paso

            elif modo == "online":
                for j in range(q):
                    X_Batch = X[j:j + 1, :]
                    Y_Batch = Y[j:j + 1, :]
                    g_t = self.funcion.gradiente(self.theta, X_Batch, Y_Batch)
                    s_t = 1 + (1 - x_t) * s_t
                    p_t = 1 / (s_t + 1)
                    m_t = (1 - p_t) * m_t + (p_t * g_t)
                    v_t = (1 - p_t) * v_t + (p_t * g_t ** 2)
                    x_t = m_t ** 2 / (v_t + self.epsilon)
                    paso = (np.minimum(self.tasaDeAprendizaje, x_t) / (np.sqrt(v_t) + self.epsilon)) * g_t
                    self.theta -= paso

            else:
                raise ValueError("Modo desconocido: debe ser 'lote', 'mini-lote' o 'online'")

            funcionObejtivo = self.funcion.ejecutarFuncion(self.theta)
            if funcionObejtivo < self.tolerancia:
                print(f"Optimización detenida en iteración {t}: Funcion objetivo = {funcionObejtivo}")
                break

            if t % self.steps == 0:
                print({
                    "iteracion": t,
                    "theta": self.theta,
                    "Funcion Objetivo": funcionObejtivo,
                })

        return self.theta





