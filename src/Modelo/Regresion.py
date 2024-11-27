from src.Modelo.Base import Modelo


class ModeloRegresion(Modelo):
    def __init__(self, DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta,FuncionError):
        super().__init__(self, DatosE,DatosT, Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta,FuncionError)

    def predecir(self):
        YpE=self.MatrizDiseñoE.getMatrizDiseño@self.theta
        YpT=self.MatrizDiseñoT.getMatrizDiseño@self.theta
        return YpE,YpT

    def entrenar(self):
        theta=self.Optimizador.optimizar()
        self.theta=theta
        return self.theta

    def calcularMetrica(self):
        R2E=self.Metrica.ejecutarFuncion(self.theta)
        self.Metrica.setDatos(self.DatosT)
        R2T=self.Metrica.ejecutarFuncion(self.theta)
        self.Metrica.setDatos(self.DatosE)
        return R2E,R2T




