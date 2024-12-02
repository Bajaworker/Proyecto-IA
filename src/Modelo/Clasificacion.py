from src.Modelo.Base import Modelo
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report


class Clasificacion(Modelo):
    def __init__(self, DatosE,DatosT,Metrica,Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta):
        super().__init__(DatosE,DatosT,Metrica, Optimizador,MatrizDiseñoE,MatrizDiseñoT,theta)

    def predecir(self):
        matrizE=self.MatrizDiseñoE.getMatrizDiseño()
        ZE=-matrizE@self.theta
        PE=softmax(ZE,axis=1)
        YpE=self.to_classlabel(PE)
        matrizT=self.MatrizDiseñoT.getMatrizDiseño()
        ZT = -matrizT @ self.theta
        PT = softmax(ZT, axis=1)
        YpT = self.to_classlabel(PT)
        return YpE, YpT


    def entrenar(self):
        theta=self.Optimizador.optimizar()
        self.theta=theta
        return self.theta

    def calcularPrecicion(self):
        theta=self.theta
        precicionE=self.Metrica.ejecutarFuncion(theta)
        self.Metrica.setMatrizDiseño(self.MatrizDiseñoT)
        self.Metrica.setDatos(self.DatosT)
        precicionT=self.Metrica.ejecutarFuncion(theta)
        self.Metrica.setMatrizDiseño(self.MatrizDiseñoE)
        self.Metrica.setDatos(self.DatosE)
        return precicionE,precicionT

    def getMatrizConfunsionE(self):
        yE=self.DatosE.getY()
        YE=self.to_classlabel(yE)
        YpE,YpT=self.predecir()
        matriz_confunsion_E=confusion_matrix(YE,YpE)
        return matriz_confunsion_E

    def getMatrizConfunsionT(self):
        yT=self.DatosT.getY()
        YT=self.to_classlabel(yT)
        _,YpT=self.predecir()
        matriz_confunsion_E=confusion_matrix(YT,YpT)
        return matriz_confunsion_E

    def getReporteE(self):
        yE=self.DatosE.getY()
        YE=self.to_classlabel(yE)
        YpE,YpT=self.predecir()
        reporteE=classification_report(YE,YpE)
        return reporteE

    def getReporteT(self):
        yT=self.DatosT.getY()
        YT=self.to_classlabel(yT)
        YpE,YpT=self.predecir()
        reporteT=classification_report(YT,YpT)
        return reporteT

    def to_classlabel(self,z):
        return z.argmax(axis=1)


