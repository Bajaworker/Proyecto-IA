# Ruta del archivo
from src.Dato.Datos import Datos

ruta = "C:/Users/benit/Downloads/challenge1_dataset22 (1).txt"

# Crear instancia de la clase Datos
datos = Datos(ruta,1)

# Configurar columnas de X e Y (adaptar si cambia el formato del archivo)
col_inicio_X = 0
col_final_X = 2
col_inicio_Y = 2
col_final_Y = None  # Esto toma todas las columnas restantes para Y
tipo_separacion = None# Cambiar según el separador real del archivo

# Definir los datos de X e Y
datos.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

# Obtener X y Y para verificar
X = datos.getX()
Y = datos.getY()

# Imprimir resultados
print("Datos X:")
print(X)
print("\nDatos Y:")
print(Y)

# Verificar el tamaño de los datos
tamaño = datos.tamañoDeDatos(tipo_separacion)
print("\nTamaño de los datos:")
print(tamaño)

# Normalizar X para un rango entre 0 y 1
X_normalizado = datos.normalizarDatosX(0, 1)
print("\nX Normalizado (entre 0 y 1):")
print(X_normalizado)