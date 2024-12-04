import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
class Datos:
    def __init__(self,ruta,porcentajeDedatos,inversar):
        self.ruta = ruta
        self.X=None
        self.Y=None
        self.porcentajeDedatos=porcentajeDedatos
        self.inversar = inversar
        self.scalerInputs = None
        self.scalerTargets = None

    def to_binary_matrix(self,z):
        # Aseguramos que los elementos de z sean de tipo float
        z = z.astype(float)
        
        # Convertimos los elementos de z a enteros para usar como índices
        z = z.astype(int)
        
        # Encuentra el valor máximo en z para determinar el número de columnas
        max_value = np.max(z)
        
        # Crea la matriz binaria con el tamaño adecuado
        binary_matrix = np.zeros((z.shape[0], max_value), dtype=int)
        
        # Llena la matriz binaria
        for i, value in enumerate(z):
            binary_matrix[i, value - 1] = 1  # Ajusta el índice para usar en el array
        
        return binary_matrix
    
    def to_binary_matrix_binario(self,z):
        # Aseguramos que los elementos de z sean de tipo float
        z = z.astype(float)
        
        # Convertimos los elementos de z a enteros para usar como índices
        z = z.astype(int)
        
        # Encuentra el valor máximo en z para determinar el número de columnas
        max_value = np.max(z)
        
        # Crea la matriz binaria con el tamaño adecuado
        binary_matrix = np.zeros((z.shape[0], max_value + 1), dtype=int)
        
        # Llena la matriz binaria
        for i, value in enumerate(z):
            if value >= 0:  # Asegura que solo los valores positivos y cero se consideren
                binary_matrix[i, value] = 1  # Ajusta el índice para usar en el array
        
        return binary_matrix

    def definirXY(self,colIniciaX,colFinalX,colIniciaY,colFinalY,tipoSeparacion):
        try:
            dato = np.loadtxt(self.ruta, delimiter=tipoSeparacion)
            X=dato[:,colIniciaX:colFinalX]
            Y=dato[:,colIniciaY:colFinalY]
            inputs_train, inputs_test, targets_train, targets_test = train_test_split(X,Y,random_state=1, test_size=1-self.porcentajeDedatos)

            if self.inversar == 0:
                # Datos de entrenamiento
                self.X = inputs_train
                self.Y = self.to_binary_matrix_binario(targets_train)
            elif self.inversar == 1:
                # Datos de prueba
                self.X = inputs_test
                self.Y = self.to_binary_matrix_binario(targets_test)
            else:
                raise ValueError("El parámetro 'inversar' debe ser 0 (entrenamiento) o 1 (prueba).")
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def normalizarDatos(self):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")
        self.scalerInputs = RobustScaler()
        self.scalerTargets = RobustScaler()
        self.X= self.scalerInputs.fit_transform(self.X)
        self.Y= self.scalerTargets.fit_transform(self.Y)



    def desNormalizarDatos(self):
        if self.X is None or self.Y is None:
            raise ValueError("Los datos no están definidos. Usa definirXY() primero.")
        if self.scalerInputs is None or self.scalerTargets is None:
            raise ValueError("Los datos no están normalizados. Usa normalizarDatos() primero.")
        self.X = self.scalerInputs.inverse_transform(self.X)
        self.Y = self.scalerTargets.inverse_transform(self.Y)


    def normalizarDatosX(self):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")
        self.scalerInputs = RobustScaler()
        self.X= self.scalerInputs.fit_transform(self.X)


    def desNormalizarDatos(self):
        if self.X is None or self.Y is None:
            raise ValueError("Los datos no están definidos. Usa definirXY() primero.")
        if self.scalerInputs is None:
            raise ValueError("Los datos no están normalizados. Usa normalizarDatos() primero.")
        self.X = self.scalerInputs.inverse_transform(self.X)


    def tamañoDeDatos(self,tipoSeparacion):
        try:
            if self.X is None and self.Y is None:
                dato = np.loadtxt(self.ruta, delimiter=tipoSeparacion)
                return dato.shape
            else:
                return (self.X.shape[0], self.X.shape[1] + self.Y.shape[1])
        except Exception as e:
            print(f"Error al calcular el tamaño de los datos: {e}")
            return None

    def renglonColumnaDeY(self):
        return self.Y.shape

    def obtenerMiniLote(self, indices):
        if self.X is None or self.Y is None:
            raise ValueError("Los datos no están definidos. Usa definirXY() primero.")
        return self.X[indices, :], self.Y[indices, :]


    def setPorcentajeDedatos(self, nuevo_porcentaje):

        if nuevo_porcentaje <= 0 or nuevo_porcentaje > 1:
            raise ValueError("El porcentaje debe ser un valor entre 0 y 1.")
        self.porcentajeDedatos = nuevo_porcentaje

    def getScalaSalida(self):
        return self.scalerTargets

    def getScalaEntrada(self):
        return self.scalerInputs




