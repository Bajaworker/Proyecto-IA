import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from src.algorithms.Base import Optimizador


class AlgoritmoSDGWithClic(Optimizador):
    def __init__(self,Datos, theta,funcion, tasaDeAprendizaje, epoca,steps, gamma,tolerancia_MSE=1e-6,tolerancia_theta=1e-4):
        super().__init__(theta, funcion, tasaDeAprendizaje, Datos)
        self.epoca = epoca
        self.gamma = gamma
        self.tolerancia_MSE = tolerancia_MSE
        self.tolerancia_theta = tolerancia_theta
        self.steps = steps

    def CorteMaximo(self,elemento, y):
        return np.clip(elemento, -y, y)

    def optimizar(self,modo="lote", tamañoDeLote=None):

        state_sum = self.getValuesInitStateSum()
        X = self.Datos.getX()
        Y = self.Datos.getY()
        q = X.shape[0]

        for t in range(1, self.epoca + 1):


            if modo == "lote":
                g_t = self.funcion.gradiente(self.theta)
                # Actualización de parámetros
                elemento = g_t + state_sum
                corte = self.CorteMaximo(elemento, self.gamma)
                state_sum = (g_t + state_sum) - corte
                theta_new = self.theta - self.tasaDeAprendizaje * corte

                # Condiciones de convergencia combinadas
                if t > 1:
                    cambio_theta = np.linalg.norm(theta_new - self.theta)

                    if cambio_theta < self.tolerancia_theta:
                        print(f"Convergencia alcanzada en la iteración {t}")
                        break
                # Actualizar theta
                self.theta = theta_new



            elif modo == "mini-lote":
                if tamañoDeLote is None:
                    raise ValueError("Se requiere tamañoDeLote para el modo 'mini-lote'")
                indices = np.random.permutation(q)
                X, Y = self.Datos.obtenerMiniLote(indices)
                n_minilote = q // tamañoDeLote

                for j in range(n_minilote):
                    X_Batch = X[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                    Y_Batch = Y[j * tamañoDeLote: (j + 1) * tamañoDeLote, :]
                    
                    
                    g_t = self.funcion.gradiente(self.theta,X_Batch, Y_Batch)
                    # Actualización de parámetros
                    elemento = g_t + state_sum
                    corte = self.CorteMaximo(elemento, self.gamma)
                    state_sum = (g_t + state_sum) - corte
                    theta_new = self.theta - self.tasaDeAprendizaje * corte

                    # Condiciones de convergencia combinadas
                    if t > 1:
                        cambio_theta = np.linalg.norm(theta_new - self.theta)

                        if cambio_theta < self.tolerancia_theta:
                            print(f"Convergencia alcanzada en la iteración {t}")
                            break
                    # Actualizar theta
                    self.theta = theta_new



                if q % tamañoDeLote != 0:
                    X_Batch = X[n_minilote * tamañoDeLote:, :]
                    Y_Batch = Y[n_minilote * tamañoDeLote:, :]


                    g_t = self.funcion.gradiente(self.theta,X_Batch, Y_Batch)
                    # Actualización de parámetros
                    elemento = g_t + state_sum
                    corte = self.CorteMaximo(elemento, self.gamma)
                    state_sum = (g_t + state_sum) - corte
                    theta_new = self.theta - self.tasaDeAprendizaje * corte

                    # Condiciones de convergencia combinadas
                    if t > 1:
                        cambio_theta = np.linalg.norm(theta_new - self.theta)

                        if cambio_theta < self.tolerancia_theta:
                            print(f"Convergencia alcanzada en la iteración {t}")
                            break
                    # Actualizar theta
                    self.theta = theta_new

            elif modo == "online":
                for j in range(q):
                    X_Batch = X[j:j + 1, :]
                    Y_Batch = Y[j:j + 1, :]

                    g_t = self.funcion.gradiente(self.theta,X_Batch, Y_Batch)
                    # Actualización de parámetros
                    elemento = g_t + state_sum
                    corte = self.CorteMaximo(elemento, self.gamma)
                    state_sum = (g_t + state_sum) - corte
                    theta_new = self.theta - self.tasaDeAprendizaje * corte

                    # Condiciones de convergencia combinadas
                    if t > 1:
                        cambio_theta = np.linalg.norm(theta_new - self.theta)

                        if cambio_theta < self.tolerancia_theta:
                            print(f"Convergencia alcanzada en la iteración {t}")
                            break
                    # Actualizar theta
                    self.theta = theta_new
            else:
                raise ValueError("Modo desconocido: debe ser 'lote', 'mini-lote' o 'online'")


            funcionObejtivo = self.funcion.ejecutarFuncion(self.theta)

            if t % self.steps == 0:
                print({
                    "iteracion": t,
                    "theta": self.theta,
                    "gradiente": g_t,
                    "Funcion Objetivo": funcionObejtivo,
                })

        return self.theta


# def test():
#     # Cargar datos
#     Ruta = "challenge0_dataset22.txt"
#     data = np.loadtxt(Ruta)

#     X = data[:, :2]
#     Y = data[:, 2:]

#     # Numero de fila y columna de theta
#     FilaDeTheta = X.shape[1] + 1
#     ColumnaDeTheta = Y.shape[1]

#     # Media y desviación estándar
#     media_X = np.mean(X, axis=0)
#     std_X = np.std(X, axis=0)

#     # Inicializa theta en un intervalo aleatorio
#     intervalo_theta = 0.1
#     theta = np.random.uniform(
#         low=media_X - intervalo_theta * std_X,
#         high=media_X + intervalo_theta * std_X,
#         size=(FilaDeTheta, ColumnaDeTheta)
#     )

#     tasaDeAprendizaje = 0.01
#     Iteraciones = 100000
#     gamma = 2

#     test = AlgoritmoSDGWithClic(Datos={"X":X,"Y":Y}, theta=theta,funcion=calculoDeGradiente, tasaDeAprendizaje=tasaDeAprendizaje, epoca=Iteraciones, gamma=gamma)
#     delta, msd=test.optimizar() 

# test()