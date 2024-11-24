import numpy as np
class Datos:
    def __init__(self,ruta):
        self.ruta = ruta
        self.X=None
        self.Y=None

    def definirXY(self,colIniciaX,colFinalX,colIniciaY,colFinalY,tipoSeparacion):
        try:
            dato = np.loadtxt(self.ruta, delimiter=tipoSeparacion)
            self.X = dato[:, colIniciaX:colFinalX]
            self.Y = dato[:, colIniciaY:colFinalY]
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

    def ColumnaDeY(self):
        c,z=self.Y.shape
        return z


