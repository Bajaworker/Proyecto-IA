from src.interface.index import Interface
from src.reading.estructura_datos import EstructuraDatos
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.Funcion.FuncionError import FuncionError
from src.Funcion.FuncionMSE import FuncionMSE
from src.Funcion.FuncionRidge import FuncionRidge
from src.Funcion.FuncionSSM import FuncionSSM
from src.Funcion.R2 import R2
from src.algorithms.Adagrad import AlgorithmAdagrad
from src.Modelo.Regresion import ModeloRegresion
import numpy as np
import sys

class proyectoIA(Interface):
    def __init__(self):
        super().__init__()
        
        self.init_interface()

        self.variablesSeleccionadas = self.getVariablesSeleccionadas()

        self.MODELO = self.variablesSeleccionadas["MODELO"]
        self.ALGORITMO = self.variablesSeleccionadas["ALGORITMO"]
        self.TECNICA_DE_REGULARIZACION = self.variablesSeleccionadas["TECNICA_DE_REGULARIZACION"]

        self.FORMA_DE_APRENDIZAJE = self.variablesSeleccionadas["FORMA_DE_APRENDIZAJE"]
        self.METRICA_DE_DESEMPENIO = self.variablesSeleccionadas["METRICA_DE_DESEMPENIO"]
        self.TASA_DE_ENTRENAMIENTO = self.variablesSeleccionadas["TASA_DE_ENTRENAMIENTO"]
        
        self.CAPERTA_DE_DATOS = self.variablesSeleccionadas["CAPERTA_DE_DATOS"]

        self.URL_DE_DATOS = self.variablesSeleccionadas["URL_DE_DATOS"]
        self.ESTRUCTURA_DATOS = self.variablesSeleccionadas["ESTRUCTURA_DATOS"]
        self.PORCENTAJE = self.variablesSeleccionadas["PORCENTAJE_DATOS"]
        
        self.init_model()
    

    def init_model(self):
        match self.MODELO:
            case "REGRESION":
                self.model_regresion()
            case "CLASIFICACION":
                print("MODELO SELECCIONADO NO DISPONIBLE")
                sys.error()

        return

    def getAlgoritmo(self,theta,funtion):
        match self.ALGORITMO:
            case "ADAGRAD":
                adagrad = AlgorithmAdagrad(
                    theta=theta,
                    funcion=funtion,
                    tasaDeAprendizaje=0.1,
                    Datos=self.DATOS_ENTRENAR,
                    lr_decay=0.001,
                    peso_decay=0.1,
                    epsilon=1e-8,
                    epoca=1000,
                    steps=100,
                    tolerancia=1e-6
                )

                return adagrad
            case "SGD_U_CLIP":
                print("ALGORITMO SELECCIONADO NO DISPONIBLE")
                sys.error()
        return

    def getFormaDeAprendizajeRegresion(self,algoritmo):
        match self.FORMA_DE_APRENDIZAJE:
            case "ONLINE":
                algoritmo.optimizar(modo="online")
            case "BATCH":
                algoritmo.optimizar(modo="lote")
            case "MINI_BATCH":
                algoritmo.optimizar(modo="mini-lote",tamañoDeLote=100)
            case _:
                print("FORMA DE APRENDIZAJE NO DEFINIDA")
                sys.error()


#Hay la nombre y la orden de la funcion esta algo mal
    def getMetricasDesempenioRegresion(self,matrix_disenio_entrenar):
        landa = 0.5
        # Crear funciones de evaluación
        funcion_error = FuncionError(matrix_disenio_entrenar, self.DATOS_ENTRENAR)
        funcion_mse = FuncionMSE(matrix_disenio_entrenar, self.DATOS_ENTRENAR, funcion_error)
        funcionObjetivo = None

        match self.METRICA_DE_DESEMPENIO:
            case "SSE":
                funcion_SSE = FuncionSSM(matrix_disenio_entrenar,self.DATOS_ENTRENAR,funcion_error)                
                funcionObjetivo = funcion_SSE
            case "RMSE":
                funcion_ridge = FuncionRidge(matrix_disenio_entrenar, self.DATOS_ENTRENAR, funcion_mse,landa)
                funcionObjetivo = funcion_ridge
            case _:
                print("METRICA DESEMPENIO NO DEFINIDA")
                sys.error()
        
        return funcionObjetivo

#Ejecutar Correcto
    def model_regresion(self):

        self.DATOS_ENTRENAR = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=0,delimiter=None)
        self.DATOS_PRUEBA = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=1,delimiter=None)
        
        self.DATOS_ENTRENAR.definirXY()
        self.DATOS_PRUEBA.definirXY()

        #Nomalizar
        #self.DATOS_PRUEBA.normalizarDatosX(), ha cambiado la funcion de normalizar, ante tomar parametro pero ahora no
        # self.DATOS_ENTRENAR.normalizarDatosX()

        X_ENTRENAR=self.DATOS_ENTRENAR.getX()
        X_PRUEBA=self.DATOS_PRUEBA.getX()
        print(X_ENTRENAR.shape, X_PRUEBA.shape)

        
        matrix_disenio_entrenar=MatrizDiseño(X_ENTRENAR,1)
        print(matrix_disenio_entrenar.getMatrizDiseño().shape)
        matrix_disenio_prueba=MatrizDiseño(X_PRUEBA,1)
        print(matrix_disenio_prueba.getMatrizDiseño().shape)

        # Inicialización de theta y parámetros
        r, c = self.DATOS_ENTRENAR.renglonColumnaDeY()
        theta = np.random.rand(matrix_disenio_entrenar.getTamañoParametro(), c)

        r2=R2(matrix_disenio_entrenar,self.DATOS_ENTRENAR)

        Yp=matrix_disenio_entrenar.getMatrizDiseño()@theta
        YpT=matrix_disenio_prueba.getMatrizDiseño()@theta

        funcionObjectivo = self.getMetricasDesempenioRegresion(matrix_disenio_entrenar=matrix_disenio_entrenar)

        algoritmo = self.getAlgoritmo(theta=theta,funtion=funcionObjectivo)

        self.getFormaDeAprendizajeRegresion(algoritmo=algoritmo)

        regresion = ModeloRegresion(DatosE=self.DATOS_ENTRENAR,
                                  DatosT=self.DATOS_PRUEBA,
                                  Metrica=r2,
                                  Optimizador=algoritmo,
                                  MatrizDiseñoE=matrix_disenio_entrenar,
                                  MatrizDiseñoT=matrix_disenio_prueba,
                                  theta=theta)

        print("theta ante de entrenamiento")
        print(theta)
        Yp,YpT=regresion.predecir()
        print("El predicion ante de entrenamiento")
        print("El predicion sobre datos entrenamiento")
        print(Yp)
        print("El predicion sobre datos test")
        print(YpT)

        theta=regresion.entrenar()
        print("theta despues de entrenar")
        print(theta)
        #vuelve a predecir
        #convieterlanormalizarendesnomalizar (Falta modificar este parte)
        #ejecuatarR2

        R2E,R2T=regresion.calcularMetrica()
        print("R2 despues de entrenar")
        print(R2E)
        print("R2 despues de test")
        print(R2T)






initClass = proyectoIA()