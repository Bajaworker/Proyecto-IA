import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionError import FuncionError
from src.Funcion.FuncionMSE import FuncionMSE
from src.Funcion.FuncionRidge import FuncionRidge
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.algorithms.Adagrad import AlgorithmAdagrad
from src.Funcion.FuncionSSM import FuncionSSM


# Ruta del archivo
ruta = "C:/Users/benit/Downloads/challenge1_dataset22 (1).txt"
ruta2="C:/Users/benit/Downloads/energyefficiency_dataset.txt"

# Crear instancia de la clase Datos
datos = Datos(ruta2, 0.5,0)

# Configurar columnas de X e Y
col_inicio_X = 0
col_final_X = 8
col_inicio_Y = 8
col_final_Y = None
tipo_separacion = None

# Definir los datos de X e Y
datos.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

# Preparar la matriz de diseño
X = datos.getX()
matriz_diseño = MatrizDiseño(X, grados=1)
print(matriz_diseño.getMatrizDiseño().shape)
print(X.shape)

# Inicialización de theta y parámetros
r, c = datos.renglonColumnaDeY()
theta = np.random.rand(matriz_diseño.getTamañoParametro(), c)
print(theta.shape)
landa = 0.5

# Crear funciones de evaluación
funcion_error = FuncionError(matriz_diseño, datos)
funcion_mse = FuncionMSE(matriz_diseño, datos, funcion_error)
funcion_ridge = FuncionRidge(matriz_diseño, datos, funcion_mse, landa)
funcion_SSE=FuncionSSM(matriz_diseño,datos,funcion_error)

# Configuración del optimizador Adagrad
adagrad = AlgorithmAdagrad(
    theta=theta,
    funcion=funcion_ridge,
    tasaDeAprendizaje=0.1,
    Datos=datos,
    lr_decay=0.001,
    peso_decay=0.1,
    epsilon=1e-8,
    epoca=1000,
    steps=100,
    tolerancia=1e-6
)

# Prueba en modo "lote"
print("\n--- Optimización en modo 'lote' ---")
theta_opt = adagrad.optimizar(modo="lote")
print(f"Parámetros optimizados (lote):\n{theta_opt}")

# Prueba en modo "mini-lote"
print("\n--- Optimización en modo 'mini-lote' ---")
theta_opt_mini = adagrad.optimizar(modo="mini-lote", tamañoDeLote=10)
print(f"Parámetros optimizados (mini-lote):\n{theta_opt_mini}")

print("\n--- Optimización en modo 'online' ---")
theta_opt_online = adagrad.optimizar(modo="online")
print(f"Parámetros optimizados (online):\n{theta_opt_online}")


# Calcular y comparar errores
error_inicial = funcion_error.ejecutarFuncion(theta)
error_final_lote = funcion_error.ejecutarFuncion(theta_opt)
error_final_minilote = funcion_error.ejecutarFuncion(theta_opt_mini)


print("\n--- Comparación de errores ---")
print(f"Error inicial: {error_inicial}")
print(f"Error final (lote): {error_final_lote}")
print(f"Error final (mini-lote): {error_final_minilote}")
print(theta_opt_online.shape)
print(theta_opt.shape)
print(theta_opt_mini.shape)
print(matriz_diseño.getMatrizDiseño().shape)
print(X.shape)


