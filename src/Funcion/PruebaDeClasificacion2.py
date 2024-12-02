import numpy as np
from src.Dato.Datos import Datos
from src.MatrizDiseño.MatrizDiseño import MatrizDiseño
from src.Funcion.FuncionEntropiaCruzadaBinaria import EntropiaCruzadaBinaria
from src.Modelo.PruebaDeClasificacion import precicion

ruta="C:/Users/benit/Downloads/microchips_dataset (1).txt"
ruta2="C:/Users/benit/Downloads/cancer_dataset.txt"

datos=Datos(ruta,0.6,0)
dato2=Datos(ruta2,0.6,0)



datos.definirXY(0,2,2,None,",")
dato2.definirXY(0,9,9,None,",")

X=datos.getX()
X2=dato2.getX()
r,c=datos.renglonColumnaDeY()
y,z=dato2.renglonColumnaDeY()


matriz=MatrizDiseño(X,1)
matriz2=MatrizDiseño(X2,1)
theta=np.random.randint(0,1,size=(matriz.getTamañoParametro(),c))
theta2=np.random.randint(0,1,size=(matriz2.getTamañoParametro(),z))
mu=0.60

entropia=EntropiaCruzadaBinaria(matriz,datos,mu)
preci=precicion(matriz2,dato2)

resultado=entropia.ejecutarFuncion(theta)
gradiente=entropia.gradiente(theta)

entropia2=EntropiaCruzadaBinaria(matriz2,dato2,mu)

resultado2=entropia2.ejecutarFuncion(theta2)
gradiente2=entropia2.gradiente(theta2)


