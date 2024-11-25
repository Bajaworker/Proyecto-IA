import numpy as np

class MatrizDiseño:
    def __init__(self, DatosX,grados):
        self.X = DatosX
        self.grados=grados
        self.matrixDiseño=self.designMatrix(self.grados,self.X)

    def designMatrix(self, degree, dataInputs):
        A = []
        for p in range(len(dataInputs)):
            M = self.polyPowerMatrix(degree, dataInputs[p, :])
            A.append(M)
        return np.array(A)

    def polyPowerMatrix(self, degree, V):
        if len(V) == 0 or degree == 0:
            return np.array([1])
        else:
            M = []
            X = V[:-1]
            W = V[-1]
            for k in range(degree + 1):
                M.append(self.polyPowerMatrix(degree - k, X) * W ** k)
            return np.concatenate(M)

    def setMatrizDiseño(self, X_Batch=None):
        if X_Batch is not None:
            # Calcula la matriz de diseño para el mini-lote
            self.matrixDiseño = self.designMatrix(self.grados, X_Batch)
        else:
            # Calcula la matriz de diseño para todos los datos
            self.matrixDiseño = self.designMatrix(self.grados, self.X)

    def setGrado(self,grados):
        self.grados = grados

    def getMatrizDiseño(self):
        return self.matrixDiseño

    def getX(self):
        return self.X

    def getGrado(self):
        return self.grados

    def getTamañoParametro(self):
        r,c=self.matrixDiseño.shape
        return c