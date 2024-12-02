import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionEntropiaCruzadaBinaria import EntropiaCruzadaBinaria
from src.Funcion.Precicion import Precicion
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.Modelo.Clasificacion import Clasificacion
from src.algorithms.Adagrad import AlgorithmAdagrad
from sklearn.metrics import confusion_matrix, classification_report

ruta="C:/Users/benit/Downloads/cancer_dataset.txt"


def to_classlabel( z):
    return z.argmax(axis=1)


datosE = Datos(ruta, 0.6,0)
datosT = Datos(ruta, 0.6,1)
# Configurar columnas de X e Y
col_inicio_X = 0
col_final_X = 9
col_inicio_Y = 9
col_final_Y = None
tipo_separacion = ","

# Definir los datos Entrenamientos de X e Y
datosE.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)
#Definir los datos Test
datosT.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

XE=datosE.getX()
XT=datosT.getX()
print(XE.shape)
print(XT.shape)

YE=datosE.getY()
YT=datosT.getY()
print(YE.shape)
print(YT.shape)

matrizDiseñoEn=MatrizDiseño(XE,1)


matrizDiseñoTest=MatrizDiseño(XT,1)

# Inicialización de theta y parámetros
r, c = datosE.renglonColumnaDeY()
theta = np.random.rand(matrizDiseñoEn.getTamañoParametro(), c)
print(theta.shape)
mu = 0.1

# Crear funciones de evaluación
entropia=EntropiaCruzadaBinaria(matrizDiseñoEn,datosE,mu)
precicion=Precicion(matrizDiseñoEn,datosE)


adagrad = AlgorithmAdagrad(
    theta=theta,
    funcion=entropia,
    tasaDeAprendizaje=0.1,
    Datos=datosE,
    lr_decay=0.001,
    peso_decay=0.1,
    epsilon=1e-8,
    epoca=1000,
    steps=100,
    tolerancia=1e-6
)




clasificacion=Clasificacion(datosE,datosT,precicion,adagrad,matrizDiseñoEn,matrizDiseñoTest,theta)

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








