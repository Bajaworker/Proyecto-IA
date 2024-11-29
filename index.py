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
        self.URL_DE_DATOS = "C:/Users/DesarrolladorJR/Documents/Proyecto-IA/energyefficiency_dataset.txt"

        self.PORCENTAJE = self.variablesSeleccionadas["PORCENTAJE_DATOS"]
        
        self.init_modelo()

    def init_modelo(self):

        self.DATOS_ENTRENAR = EstructuraDatos(ruta=self.URL_DE_DATOS,porcentaje=self.PORCENTAJE,inversar=0,delimiter=None)
        self.DATOS_PRUEBA = EstructuraDatos(ruta=self.URL_DE_DATOS,porcentaje=self.PORCENTAJE,inversar=1,delimiter=None)
        
        self.DATOS_ENTRENAR.definirXY()
        self.DATOS_PRUEBA.definirXY()

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
        print(theta.shape)
        landa = 0.5

        # Crear funciones de evaluación
        funcion_error = FuncionError(matrix_disenio_entrenar, self.DATOS_ENTRENAR)
        funcion_mse = FuncionMSE(matrix_disenio_entrenar, self.DATOS_ENTRENAR, funcion_error)
        funcion_ridge = FuncionRidge(matrix_disenio_entrenar, self.DATOS_ENTRENAR, funcion_mse, landa)
        funcion_SSE=FuncionSSM(matrix_disenio_entrenar,self.DATOS_ENTRENAR,funcion_error)
        r2=R2(matrix_disenio_entrenar,self.DATOS_ENTRENAR)
        Yp=matrix_disenio_entrenar.getMatrizDiseño()@theta
        YpT=matrix_disenio_prueba.getMatrizDiseño()@theta

        adagrad = AlgorithmAdagrad(
            theta=theta,
            funcion=funcion_SSE,
            tasaDeAprendizaje=0.1,
            Datos=self.DATOS_ENTRENAR,
            lr_decay=0.001,
            peso_decay=0.1,
            epsilon=1e-8,
            epoca=1000,
            steps=100,
            tolerancia=1e-6
        )

        regresion=ModeloRegresion(self.DATOS_ENTRENAR,self.DATOS_PRUEBA,r2,adagrad,matrix_disenio_entrenar,matrix_disenio_prueba,theta)

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

        R2E,R2T=regresion.calcularMetrica()
        print("R2 despues de entrenar")
        print(R2E)
        print("R2 despues de test")
        print(R2T)






initClass = proyectoIA()