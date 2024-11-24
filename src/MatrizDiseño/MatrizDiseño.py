import numpy as np

class MatrizDiseño:
    def __init__(self, Datos,grados):
        self.Datos = Datos
        self.grados=grados
        self.matrixDiseño=self.designMatrix(self.grados,self.Datos.getX())

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

    def setMatrizDiseño(self,grado):
        self.grado=grado
        self.matrixDiseño=self.designMatrix(self.grados,self.Datos.getX())

    def getMatrizDiseño(self):
        return self.matrixDiseño

    def getTamaño(self):
        return self.matrixDiseño.shape