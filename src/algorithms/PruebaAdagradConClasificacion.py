import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionEntropiaCruzadaBinaria import EntropiaCruzadaBinaria
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.algorithms.Adagrad import AlgorithmAdagrad



# Ruta del archivo
ruta = "C:/Users/benit/Downloads/cancer_dataset.txt"


# Crear instancia de la clase Datos
datos = Datos(ruta, 0.6,0)

# Configurar columnas de X e Y
col_inicio_X = 0
col_final_X = 9
col_inicio_Y = 9
col_final_Y = None
tipo_separacion = ","

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
mu = 0.1

entropia=EntropiaCruzadaBinaria(matriz_diseño,datos, mu)



# Configuración del optimizador Adagrad
adagrad = AlgorithmAdagrad(
    theta=theta,
    funcion=entropia,
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
