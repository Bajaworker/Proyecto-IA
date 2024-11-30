import pandas as pd
import numpy as np
from scipy.io import loadmat


class ReadingDataSets:

    def __init__(self,delimiter):
        self.delimiter = delimiter
    
    def setFormatDefault(self,data):
        data = np.array(data)
        return{"data":data}
    
    def readingTxt(self,url):
        doc = np.loadtxt(url, delimiter=None)

        return self.setFormatDefault(doc)

    def readingDat(self,url):
        doc = pd.read_csv(url, delimiter=self.delimiter)
        return self.setFormatDefault(doc)
    
    
    def readingMat(self,url):
        data = loadmat(url)
        return {"x":np.array(data["engineInputs"]).T,"y":np.array(data["engineTargets"]).T}

    def readingIdX1Ubyte(self,url):
        pass


    def readingXls(self,url):
        # doc = pd.read_excel(url, sheet_name="sheet1",header=None)
        doc=pd.read_excel(url,engine="xlrd")
        data = np.array(doc)
        matrix_sin_header=data
        # matrix_sin_header = data[1:, :]  # Seleccionar desde la fila 1 hasta el final

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

