from src.reading.index import ReadingDataSets

url="src/datasets/classification/Admisiones/ex2data1.txt"
# url="src/datasets/classification/Breast cancer/cancer_dataset.dat"
# url="src/datasets/regression/Concrete Compressive Strength/Concrete_Data.xls"
# url="src/datasets/regression/Gas Turbine Emission/gt_2011.csv"
# url = "src/datasets/regression/Synthetic Data/challenge03_syntheticdataset22.mat"
# url="src/datasets/regression/Engine Behavior/engine_dataset.mat"
# url="src/datasets/classification/HandWrittenDigit/handWrittenDigit_dataset.mat"
# url="src/datasets/classification/MNIST/t10k-labels.idx1-ubyte"


test = ReadingDataSets()

# resultado={
#     columns:10,
#     rows:5,
#     data:[1,2,2,....]
# }

print(test.reading(url))