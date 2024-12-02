import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionError import FuncionError
from src.Funcion.FuncionMSE import FuncionMSE
from src.Funcion.FuncionRidge import FuncionRidge
from src.Funcion.FuncionSSM import FuncionSSM
from src.Funcion.R2 import R2
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.algorithms.Adagrad import AlgorithmAdagrad
from src.Modelo.Regresion import ModeloRegresion


ruta2="C:/Users/benit/Downloads/energyefficiency_dataset.txt"

datosE = Datos(ruta2, 0.8,0)
datosT = Datos(ruta2, 0.8,1)
# Configurar columnas de X e Y
col_inicio_X = 0
col_final_X = 8
col_inicio_Y = 8
col_final_Y = None
tipo_separacion = None

# Definir los datos Entrenamientos de X e Y
datosE.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)
#Definir los datos Test
datosT.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

XE=datosE.getX()
XT=datosT.getX()
print(XE.shape, XT.shape)


matrizDiseñoEn=MatrizDiseño(XE,1)
print(matrizDiseñoEn.getMatrizDiseño().shape)

matrizDiseñoTest=MatrizDiseño(XT,1)
print(matrizDiseñoTest.getMatrizDiseño().shape)

# Inicialización de theta y parámetros
r, c = datosE.renglonColumnaDeY()
theta = np.random.rand(matrizDiseñoEn.getTamañoParametro(), c)
print(theta.shape)
landa = 0.5

# Crear funciones de evaluación
funcion_error = FuncionError(matrizDiseñoEn, datosE)
funcion_mse = FuncionMSE(matrizDiseñoEn, datosE, funcion_error)
funcion_ridge = FuncionRidge(matrizDiseñoEn, datosE, funcion_mse, landa)
funcion_SSE=FuncionSSM(matrizDiseñoEn,datosE,funcion_error)
r2=R2(matrizDiseñoEn,datosE)

Yp=matrizDiseñoEn.getMatrizDiseño()@theta
YpT=matrizDiseñoTest.getMatrizDiseño()@theta


adagrad = AlgorithmAdagrad(
    theta=theta,
    funcion=funcion_ridge,
    tasaDeAprendizaje=0.1,
    Datos=datosE,
    lr_decay=0.001,
    peso_decay=0.1,
    epsilon=1e-8,
    epoca=1000,
    steps=100,
    tolerancia=1e-6
)

regresion=ModeloRegresion(datosE,datosT,r2,adagrad,matrizDiseñoEn,matrizDiseñoTest,theta)

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










