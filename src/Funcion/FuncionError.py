from src.Funcion.Base import Funcion

class FuncionError(Funcion):
    def __init__(self,MatrizDiseño,Datos):
        super().__init__(MatrizDiseño, Datos)

    def ejecutarFuncion(self,theta):
        Y=self.Datos.getY()
        e=Y-(self.MatrizDiseño.matrixDiseño@theta)
        return e

    def gradiente(self):
        return "No esta definida"

