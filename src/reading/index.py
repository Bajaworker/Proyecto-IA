import pandas as pd
import numpy as np
from scipy.io import loadmat


class ReadingDataSets:

    def __init__(self):
        self.delimiter = ","
    
    def setFormatDefault(self,data):
        
        data = np.array(data)

        rows, columns = data.shape

        return{"colums":columns,"rows":rows,"data":data}
    
    def readingTxt(self,url):
        doc = pd.read_csv(url, delimiter=self.delimiter)
        return self.setFormatDefault(doc)

    def readingDat(self,url):
        doc = pd.read_csv(url, delimiter=self.delimiter)
        return self.setFormatDefault(doc)
    
    
    def readingMat(self,url):
        data = loadmat(url)#Verificar Formato
        return data

        pass

    def readingIdX1Ubyte(self,url):
        pass


    def readingXls(self,url):
        doc = pd.read_excel(url, sheet_name="sheet1",header=None)

        data = np.array(doc)

        matrix_sin_header = data[1:, :]  # Seleccionar desde la fila 1 hasta el final

        return self.setFormatDefault(matrix_sin_header)
    
    def readingCsv(self,url):
        doc = pd.read_csv(url, delimiter=self.delimiter)
        return self.setFormatDefault(doc)
    
    def reading(self,url):
        direction,extension = url.split(".")
        
        options={
            "txt":self.readingTxt,
            "dat":self.readingDat,
            "mat":self.readingMat,
            # "idx1-ubyte",
            "xls":self.readingXls,
            "csv":self.readingCsv
        }

        return options[extension](url)
    
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



# url="src/datasets/classification/Admisiones/ex2data1.txt"
# url="src/datasets/classification/Breast cancer/cancer_dataset.dat"
# url="src/datasets/classification/HandWrittenDigit/handWrittenDigit_dataset.mat"
# url="src/datasets/classification/MNIST/t10k-labels.idx1-ubyte"
# url="src/datasets/regression/Concrete Compressive Strength/Concrete_Data.xls"
# url="src/datasets/regression/Engine Behavior/engine_dataset.mat"
# url="src/datasets/regression/Gas Turbine Emission/gt_2011.csv"
# url = "src/datasets/regression/Synthetic Data/challenge03_syntheticdataset22.mat"

# test = ReadingDataSets()
# print(test.reading(url))


# Cargar archivo .mat
# url = "src/datasets/classification/HandWrittenDigit/handWrittenDigit_dataset.mat"
# file_path = url
# data = loadmat(file_path)





# url = "src/datasets/classification/HandWrittenDigit/handWrittenDigit_dataset.mat"
# df = pd.read_csv(url, ",")

# print(np.array(df))






# def load_idx1_ubyte(file_path):
#     with open(file_path, 'rb') as f:
#         # Leer encabezado (primero 8 bytes)
#         magic = int.from_bytes(f.read(4), 'big')  # Código mágico
#         num_items = int.from_bytes(f.read(4), 'big')  # Número de etiquetas

#         # Leer etiquetas
#         labels = np.frombuffer(f.read(), dtype=np.uint8)  # Cargar como enteros de 8 bits
        
#     return labels

# # Ruta al archivo
# file_path = url

# # Cargar etiquetas
# labels = load_idx1_ubyte(file_path)

# print(labels)

