import numpy as np
import sys
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
from src.algorithms.SDGConClic import AlgoritmoSDGWithClic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from src.Funcion.FuncionEntropiaCruzadaBinaria import EntropiaCruzadaBinaria
from src.Funcion.Precicion import Precicion
from src.Modelo.Clasificacion import Clasificacion

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
                self.model_clasificacion()
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
                SDGWithClic = AlgoritmoSDGWithClic(
                    Datos=self.DATOS_ENTRENAR, 
                    theta=theta,
                    funcion=funtion,
                    tasaDeAprendizaje=0.01,
                    epoca=100000,
                    steps=10000,
                    gamma=2
                )
                return SDGWithClic
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

    def model_clasificacion(self):
        self.DATOS_ENTRENAR = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=0,delimiter=self.ESTRUCTURA_DATOS["delimiter"])
        self.DATOS_PRUEBA = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=1,delimiter=self.ESTRUCTURA_DATOS["delimiter"])
        
        self.DATOS_ENTRENAR.definirXY()
        self.DATOS_PRUEBA.definirXY()

        X_ENTRENAR=self.DATOS_ENTRENAR.getX()
        X_PRUEBA=self.DATOS_PRUEBA.getX()
        print(X_ENTRENAR.shape)
        print(X_PRUEBA.shape)

        YE=self.DATOS_ENTRENAR.getY()
        YT=self.DATOS_PRUEBA.getY()
        print(YE.shape)
        print(YT.shape)

        matrix_disenio_entrenar=MatrizDiseño(X_ENTRENAR,1)
        matrix_disenio_prueba=MatrizDiseño(X_PRUEBA,1)

        # Inicialización de theta y parámetros
        r, c = self.DATOS_ENTRENAR.renglonColumnaDeY()
        theta = np.random.rand(matrix_disenio_entrenar.getTamañoParametro(), c)
        print(theta.shape)
        mu = 0.1

        # Crear funciones de evaluación
        funcionObjectivo=EntropiaCruzadaBinaria(matrix_disenio_entrenar,self.DATOS_ENTRENAR,mu)
        precicion=Precicion(matrix_disenio_entrenar,self.DATOS_ENTRENAR)

        algoritmo = self.getAlgoritmo(theta=theta,funtion=funcionObjectivo)

        self.getFormaDeAprendizajeRegresion(algoritmo=algoritmo)

        clasificacion=Clasificacion(self.DATOS_ENTRENAR,self.DATOS_PRUEBA,precicion,algoritmo,matrix_disenio_entrenar,matrix_disenio_prueba,theta)

        clasificacion.entrenar()

        # Predicción y evaluación
        y_pred_train, y_pred_test = clasificacion.predecir()

        acc_train,acc_test  =clasificacion.calcularPrecicion()


        print("Accuracy en entrenamiento:", acc_train)
        print("Accuracy en prueba:", acc_test)

        # Mostrar matriz de confusión y reporte
        print("Matriz de confusión (entrenamiento):\n", clasificacion.getMatrizConfunsionE())
        print("Matriz de confusión (prueba):\n", clasificacion.getMatrizConfunsionT())
        print("Reporte de clasificación (entrenamiento):\n", clasificacion.getReporteE())
        print("Reporte de clasificación (prueba):\n", clasificacion.getReporteT())


    def model_regresion(self):

        self.DATOS_ENTRENAR = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=0,delimiter=self.ESTRUCTURA_DATOS["delimiter"])
        self.DATOS_PRUEBA = EstructuraDatos(ruta=self.URL_DE_DATOS,estructura_datos=self.ESTRUCTURA_DATOS,porcentaje=self.PORCENTAJE,inversar=1,delimiter=self.ESTRUCTURA_DATOS["delimiter"])
        
        self.DATOS_ENTRENAR.definirXY()
        self.DATOS_PRUEBA.definirXY()

        # Entradas y objetivos de normalización de datos
        # Inicializar RobustScaler
        scalerInputs  = RobustScaler()
        scalerTargets = RobustScaler()
        # Transformar los datos
        robust_scaled_Inputs  = scalerInputs.fit_transform(self.DATOS_ENTRENAR.getAllColumnsX())
        robust_scaled_Targets = scalerTargets.fit_transform(self.DATOS_ENTRENAR.getAllColumnsY())

        # Datos divididos de entrenamiento y prueba
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(robust_scaled_Inputs, robust_scaled_Targets, random_state = 1, test_size = self.PORCENTAJE)


        self.DATOS_ENTRENAR.setX(inputs_train)
        self.DATOS_ENTRENAR.setY(targets_train)

        self.DATOS_PRUEBA.setX(inputs_test)
        self.DATOS_PRUEBA.setY(targets_test)


        # Entradas y objetivos de normalización de datos
        # Inicializar RobustScaler
        scalerInputs  = RobustScaler()
        scalerTargets = RobustScaler()
        # Transformar los datos
        robust_scaled_Inputs  = scalerInputs.fit_transform(self.DATOS_ENTRENAR.getAllColumnsX())
        robust_scaled_Targets = scalerTargets.fit_transform(self.DATOS_ENTRENAR.getAllColumnsY())

        # Datos divididos de entrenamiento y prueba
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(robust_scaled_Inputs, robust_scaled_Targets, random_state = 1, test_size = self.PORCENTAJE)


        self.DATOS_ENTRENAR.setX(inputs_train)
        self.DATOS_ENTRENAR.setY(targets_train)

        self.DATOS_PRUEBA.setX(inputs_test)
        self.DATOS_PRUEBA.setY(targets_test)

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

        Yp,YpT=regresion.predecir()

        # # Transformación inversa en RobustScaler
        # # Transformación inversa de los datos de entrenamiento para las salidas
        yTrain = scalerTargets.inverse_transform(targets_train)
        yhTrain = scalerTargets.inverse_transform(Yp)

        # # Transformación inversa de los datos de prueba para las salidas
        yTest  = scalerTargets.inverse_transform(targets_test)
        yhTest = scalerTargets.inverse_transform(YpT)

        # # R2 for raw train data
        R2_train = r2_score(yTrain.reshape(-1, 1),yhTrain.reshape(-1, 1))
        print("R2 despues de entrenar")
        print(R2_train)

        # # R2 for raw test data
        R2_test = r2_score(yTest.reshape(-1, 1),yhTest.reshape(-1, 1))
        print("R2 despues de test")
        print(R2_test)

        # R2E,R2T=regresion.calcularMetrica()
        # print("R2 despues de entrenar")
        # print(R2E)
        # print("R2 despues de test")
        # print(R2T)






initClass = proyectoIA()