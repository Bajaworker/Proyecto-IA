import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionRosenbrock import FuncionRosenbrock
from src.algorithms.SMORMS3 import SMORMS3

funcion_rosenbrock=FuncionRosenbrock()

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

theta=10*np.random.rand(100,1)
theta2=theta.flatten().tolist()

smorms3=SMORMS3(
    theta=theta2,
    funcion=funcion_rosenbrock,
    tasaDeAprendizaje=0.1,
    Datos=datos,
    epsilon=1e-10,
    epoca=2000000,
    steps=10000,
    tolerancia=1e-6
    )

theta=smorms3.optimizar()
print(theta)
