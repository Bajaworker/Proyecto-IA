import numpy as np
class Datos:
    def __init__(self,ruta,porcentajeDedatos):
        self.ruta = ruta
        self.X=None
        self.Y=None
        self.porcentajeDedatos=porcentajeDedatos
        ## para implementar la otras forma, es necesario infresar un parametro de que si es invesor o no, si es inverso, entoces podemos aplicar la obtenercion de inverso para que aplicar obtener datos XyY de test
        ###Agregar un nuevo atributo que podemos permite que obtener la datos de test si la porcentajeDeDatos es diferentes a la 1
###self.Ytest y self.Xtest o otros forma es podemos separar crear un datos llamado datos de entrenamiento y datos de test
    def definirXY(self,colIniciaX,colFinalX,colIniciaY,colFinalY,tipoSeparacion):
        try:
            dato = np.loadtxt(self.ruta, delimiter=tipoSeparacion)
            datosTomado = int(dato.shape[0] * self.porcentajeDedatos)
            self.X = dato[:datosTomado, colIniciaX:colFinalX]
            self.Y = dato[:datosTomado, colIniciaY:colFinalY]
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def normalizarDatosX(self,ymin,ymax):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")

        X_min=self.X.min(axis=0)
        X_max=self.X.max(axis=0)
        self.X=((ymax - ymin) * (self.X - X_min) / (X_max - X_min)) + ymin
        return self.X

    def desNormalizarDatosX(self, X_min, X_max, ymin, ymax):
        if self.X is None:
            raise ValueError("Los datos de X no están definidos. Usa definirXY() primero.")

        self.X = ((X_max - X_min) * (self.X - ymin) / (ymax - ymin)) + X_min
        return self.X

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




