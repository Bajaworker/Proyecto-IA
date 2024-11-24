import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionError import FuncionError
from src.Funcion.FuncionSSM import FuncionSSM
from src.Modelo.Regresion import ModeloRegresion
from src.algorithms.Adagrad import AlgorithmAdagrad
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño

np.random.seed(42)  # Para reproducibilidad
X = np.random.rand(100, 1) * 10  # 100 datos entre 0 y 10
true_theta = np.array([2.5])  # Coeficiente verdadero
y = X @ true_theta + np.random.randn(100, 1)  # y = 2.5 * X + ruido

# Guardar los datos en un archivo temporal
ruta = "datos_prueba.txt"
datos = np.hstack((X, y))
np.savetxt(ruta, datos, delimiter=',')

# Paso 2: Inicializar clases
# Datos
datosClase = Datos(ruta)
datosClase.definirXY(0, 1, 1, 2, ",")

# Matriz de Diseño
grado = 1
matriz_diseño = MatrizDiseño(datosClase, grado)

print(matriz_diseño.getTamaño())
# Función de error
funcion_error = (
    FuncionError(matriz_diseño, datosClase))

# Función objetivo (Suma de errores al cuadrado)
funcion_objetivo = FuncionSSM(matriz_diseño, datosClase, funcion_error)

# Optimización con AlgorithmAdagrad
theta_inicial = np.random.rand(grado + 1, 1)
print(theta_inicial.shape)
optimizador = AlgorithmAdagrad(
    theta=theta_inicial,
    funcion=funcion_objetivo,
    tasaDeAprendizaje=0.1,
    lr_decay=0.01,
    epoca=1000,
    steps=200,
)

# Modelo de regresión lineal
modelo = ModeloRegresion(
    datosClase, funcion_objetivo, optimizador, matriz_diseño, theta_inicial, funcion_error
)

# Paso 3: Entrenar el modelo
theta_final = modelo.entrenar(theta_inicial)
print(f"Theta final: {theta_final}")

# Paso 4: Evaluar las predicciones
predicciones = modelo.predecir(theta_final)
print(f"Primeras 10 predicciones: {predicciones[:10].flatten()}")
print(f"Primeros 10 valores reales: {y[:10].flatten()}")