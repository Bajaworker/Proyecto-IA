from src.Funcion.Base import Funcion
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño


class FuncionError(Funcion):
    def __init__(self,MatrizDiseño,Datos):
        super().__init__(MatrizDiseño,Datos)

    def ejecutarFuncion(self,theta,X_Batch=None,Y_Batch=None):
        if X_Batch is not None or Y_Batch is not None:
            self.MatrizDiseño.setMatrizDiseño(X_Batch)
            matrizBatch=self.MatrizDiseño.getMatrizDiseño()
            error=Y_Batch-(matrizBatch@theta)
            return error

        Y=self.Datos.getY()
        e=Y-(self.MatrizDiseño.getMatrizDiseño()@theta)
        return e

    def gradiente(self,theta,X_Batch=None,Y_Batch=None):
        return "No esta definida"

