direccion_carpetas_datos = {
    "REGRESION":{
        "ENGINE_BEHAVIOR":{
            "id":"ENGINE_BEHAVIOR",
            "text":"Engine behavior",
            "options":[
                {
                    "id":"src/datasets/regression/Engine Behavior/engine_dataset.mat",
                    "text":"engine_dataset.mat"
                },
            ],
        },
        "CONCRETE_COMPRESSIVE_STRENGTH":{
            "id":"CONCRETE_COMPRESSIVE_STRENGTH",
            "text":"Concrete Compressive Strength",
            "options":[
                {
                    "id":"src/datasets/regression/Concrete Compressive Strength/Concrete_Data.xls",
                    "text":"Concrete_Data.xls"
                }
            ],
        },
        "GAS_TURBINE_EMISSION":{
            "id":"GAS_TURBINE_EMISSION",
            "text":"Gas Turbine Emission",
            "options":[
                {
                    "id":"src/datasets/regression/Gas Turbine Emission/gt_2011.csv",
                    "text":"gt_2011.csv"
                },
                                {
                    "id":"src/datasets/regression/Gas Turbine Emission/gt_2012.csv",
                    "text":"gt_2012.csv"
                },
                                {
                    "id":"src/datasets/regression/Gas Turbine Emission/gt_2013.csv",
                    "text":"gt_2013.csv"
                },
                                {
                    "id":"src/datasets/regression/Gas Turbine Emission/gt_2014.csv",
                    "text":"gt_2014.csv"
                },
                                {
                    "id":"src/datasets/regression/Gas Turbine Emission/gt_2015.csv",
                    "text":"gt_2015.csv"
                },

            ],
        },
    },
    "CLASIFICACION":{
        "MICROCHIPS":{
            "id":"MICROCHIPS",
            "text":"Microchips",
            "options":[
                {
                    "id":"C:/Users/benit/PycharmProjects/Proyecto-IA2/src/datasets/classification/Microchips/microchips_dataset.txt",
                    "text":"microchips_dataset.txt"
                }
            ],
        },
        "BREAST_CANCER":{
            "id":"BREAST_CANCER",
            "text":"Breast cancer",
            "options":[
                {
                    "id":"src/datasets/classification/Breast cancer/cancer_dataset.dat",
                    "text":"cancer_dataset.dat"
                }
            ],
        },
        "DERMATOLOGY":{
            "id":"DERMATOLOGY",
            "text":"Dermatology",
            "options":[
                {
                    "id":"src/datasets/classification/Dermatology/dermatology.dat",
                    "text":"dermatology.dat"
                }
            ],
        },
    },
}

options_metricas_desempenio = {
    "REGRESION":        [
            {
                "id":"SSE",
                "text":"SSE",
            },
            {
                "id":"MSE",
                "text":"MSE",
            },
            {
                "id":"RMSE",
                "text":"RMSE",
            },
            {
                "id":"R2",
                "text":"R2",
            }
    ],
    "CLASIFICACION":[
            {
                "id":"MATRIZ_DE_CONFUSION",
                "text":"matriz de confusion",
            },
            {
                "id":"ROC",
                "text":"ROC",
            },
            {
                "id":"ENTROPIA_CRUZADA_BINARIA",
                "text":"Entropia cruzada binaria",
            },
            {
                "id":"ENTROPIA_CRUZADA_CATEGORICA",
                "text":"Entropia cruzada categorica",
            }
    ]
}