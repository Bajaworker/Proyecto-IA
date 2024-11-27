import numpy as np

from src.Dato.Datos import Datos
from src.Funcion.FuncionError import FuncionError
from src.Funcion.FuncionMSE import FuncionMSE
from src.Funcion.FuncionRidge import FuncionRidge
from src.Funcion.FuncionRosenbrock import FuncionRosenbrock
from src.Funcion.FuncionSSM import FuncionSSM
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño

# Ruta del archivo
ruta = "C:/Users/benit/Downloads/challenge1_dataset22 (1).txt"

# Crear instancia de la clase Datos
datos = Datos(ruta,0.5,0)

# Configurar columnas de X e Y (adaptar si cambia el formato del archivo)
col_inicio_X = 0
col_final_X = 2
col_inicio_Y = 2
col_final_Y = None  # Esto toma todas las columnas restantes para Y
tipo_separacion = None  # Cambiar según el separador real del archivo

# Definir los datos de X e Y
datos.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

r,c=datos.renglonColumnaDeY()
X=datos.getX()
#Matriz
matriz_diseño = (MatrizDiseño(X, grados=1))

#theta y landa
theta=np.random.randint(0,1,size=(matriz_diseño.getTamañoParametro(),c))
landa=0.5

# Probar FuncionError
funcion_error = FuncionError(matriz_diseño,datos)
print("Error:")
print(funcion_error.ejecutarFuncion(theta))

# Probar FuncionMSE
funcion_mse = FuncionMSE(matriz_diseño, datos,funcion_error)
print("\nMSE:")
print(funcion_mse.ejecutarFuncion(theta))

# Probar FuncionRidge
funcion_ridge = FuncionRidge(matriz_diseño, datos,funcion_mse,landa)
print("\nRidge:")
print(funcion_ridge.ejecutarFuncion(theta))

# Probar FuncionSSM
funcion_ssm = FuncionSSM(matriz_diseño,datos, funcion_error)
print("\nSSE:")
print(funcion_ssm.ejecutarFuncion(theta))

funcion_rosenbrock=FuncionRosenbrock()
theta2=np.array([1,1,1,1])
resultado_funcion = funcion_rosenbrock.ejecutarFuncion(theta2)
print("Resultado de la función de Rosenbrock:", resultado_funcion)
gradiente = funcion_rosenbrock.gradiente(theta2)
print("Gradiente de la función de Rosenbrock:", gradiente)


# Prueba con mini-lotes
indices_minilote = np.random.choice(range(datos.getX().shape[0]), size=10, replace=False)
X_batch, Y_batch = datos.obtenerMiniLote(indices_minilote)

error_minilote = funcion_error.ejecutarFuncion(theta, X_Batch=X_batch, Y_Batch=Y_batch)
print(f"Error para mini-lote (primeras 5 filas): \n{error_minilote[:5]}")

ridge_minilote = funcion_ridge.ejecutarFuncion(theta, X_Batch=X_batch, Y_Batch=Y_batch)
print(f"Función Ridge para mini-lote: {ridge_minilote}")