from src.Modelo.Base import Modelo
from src.algorithms.PruebaAdagrad import matriz_diseño


class ModeloRegresion(Modelo):
    def __init__(self, DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta):
        super().__init__(DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta)

    def predecir(self):
        matriz_diseñoE=self.MatrizDiseñoE.getMatrizDiseño()
        matriz_diseñoT=self.MatrizDiseñoT.getMatrizDiseño()
        YpE=matriz_diseñoE@self.theta
        YpT=matriz_diseñoT@self.theta
        return YpE,YpT

    def entrenar(self):
        theta=self.Optimizador.optimizar()
        self.theta=theta
        return self.theta

    def calcularMetrica(self):
        R2E=self.Metrica.ejecutarFuncion(self.theta)
        self.Metrica.setMatrizDiseño(self.MatrizDiseñoT)
        self.Metrica.setDatos(self.DatosT)
        R2T=self.Metrica.ejecutarFuncion(self.theta)
        self.Metrica.setDatos(self.DatosE)
        self.Metrica.setMatrizDiseño(self.MatrizDiseñoE)
        return R2E,R2T





