from src.Modelo.Base import Modelo


class ModeloRegresion(Modelo):
    def __init__(self,Datos, FuncionObjetivo, Optimizador,MatrizDiseño,theta,FuncionError):
        super().__init__(Datos, FuncionObjetivo, Optimizador, MatrizDiseño, theta, FuncionError)

    def predecir(self, theta):
        return self.MatrizDiseño.matrixDiseño@theta

    def entrenar(self,theta):
        theta=self.Optimizador.optimizar()
        return theta



