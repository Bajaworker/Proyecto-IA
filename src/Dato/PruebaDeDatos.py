# Ruta del archivo
from src.Dato.Datos import Datos

ruta = "C:/Users/benit/Downloads/challenge1_dataset22 (1).txt"

# Crear instancia de la clase Datos
datos1 = Datos(ruta,0.5,0)

datos2 = Datos(ruta,0.5,1)

# Configurar columnas de X e Y (adaptar si cambia el formato del archivo)
col_inicio_X = 0
col_final_X = 2
col_inicio_Y = 2
col_final_Y = None  # Esto toma todas las columnas restantes para Y
tipo_separacion = None# Cambiar según el separador real del archivo

# Definir los datos de X e Y
datos1.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)
datos2.definirXY(col_inicio_X, col_final_X, col_inicio_Y, col_final_Y, tipo_separacion)

# Obtener X y Y para verificar
X = datos1.getX()
Y = datos1.getY()
X2 = datos2.getX()
Y2 = datos2.getY()

# Imprimir resultados
print("Datos X:")
print(X)
print("\nDatos Y:")
print(Y)

print("Datos X:")
print(X2)
print("\nDatos Y:")
print(Y2)

