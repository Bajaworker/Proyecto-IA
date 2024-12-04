url = "C:/Users/benit/PycharmProjects/Proyecto-IA2/"
url = ""
direccion_carpetas_datos = {
    "REGRESION":{
        "ENGINE_BEHAVIOR":{
            "id":"ENGINE_BEHAVIOR",
            "text":"Engine behavior",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/regression/Engine Behavior/engine_dataset.mat",
                        "structure":{
                            "type":"TABLE_SPLIT",
                            "delimiter":None
                        }
                    },
                    "text":"engine_dataset.mat"
                },
                {
                    "id":{
                        "url":url+"src/datasets/regression/Synthetic Data/challenge01_syntheticdataset22.txt",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,2],
                            "columns_y":[2,None],
                            "delimiter":","
                        }
                    },
                    "text":"Prueba datos normalizacion.txt"
                    
                }
            ],
        },
        "CONCRETE_COMPRESSIVE_STRENGTH":{
            "id":"CONCRETE_COMPRESSIVE_STRENGTH",
            "text":"Concrete Compressive Strength",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/regression/Concrete Compressive Strength/Concrete_Data.xls",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,8],
                            "columns_y":[8,None],
                            "delimiter":None

                        }
                    },
                    "text":"Concrete_Data.xls"
                }
            ],
        },
        "GAS_TURBINE_EMISSION":{
            "id":"GAS_TURBINE_EMISSION",
            "text":"Gas Turbine Emission",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/regression/Gas Turbine Emission/gt_2011.csv",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":None
                        }
                    },
                    "text":"gt_2011.csv",
                },
                {
                    "id":{
                        "url":url+"src/datasets/regression/Gas Turbine Emission/gt_2012.csv",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":None
                        }
                    },
                    "text":"gt_2012.csv",
                },
                {
                    "id":{
                        "url":url+"src/datasets/regression/Gas Turbine Emission/gt_2013.csv",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":None
                        }
                    },
                    "text":"gt_2013.csv",
                },
                {
                    "id":{
                        "url":url+"src/datasets/regression/Gas Turbine Emission/gt_2014.csv",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":None

                        }
                    },
                    "text":"gt_2014.csv",
                },
                {
                    "id":{
                        "url":url+"src/datasets/regression/Gas Turbine Emission/gt_2015.csv",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":None
                        }
                    },
                    "text":"gt_2015.csv",
                },

            ],
        },
    },
    "CLASIFICACION":{
        "BREAST_CANCER":{
            "id":"BREAST_CANCER",
            "text":"Breast cancer",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/classification/Breast cancer/cancer_dataset.txt",
                        "structure":{
                            "type":"TABLE_DEFAULT",
                            "columns_x":[0,9],
                            "columns_y":[9,None],
                            "delimiter":","
                        }
                    },
                    "text":"cancer_dataset.txt"
                }
            ],
        },
        "MICROCHIPS":{
            "id":"MICROCHIPS",
            "text":"Microchips",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/classification/Microchips/microchips_dataset.txt",
                        "structure":{
                            "type":"TABLE_DEFAULT_BINARIO",
                            "columns_x":[0,2],
                            "columns_y":[2,None],
                            "delimiter":",",

                        }
                    },
                    "text":"microchips_dataset.txt"
                }
            ],
        },

        "DERMATOLOGY":{
            "id":"DERMATOLOGY",
            "text":"Dermatology",
            "options":[
                {
                    "id":{
                        "url":url+"src/datasets/classification/Dermatology/dermatology.dat",
                        "structure":{
                            "type":"TABLE_DEFAULT_NUMERACION",
                            "columns_x":[0,34],
                            "columns_y":[34,None],
                            "delimiter":" "
                        }
                    },
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
            # {
            #     "id":"MSE",
            #     "text":"MSE",
            # },
            {
                "id":"RMSE",
                "text":"RMSE",
            },
            # {
            #     "id":"R2",
            #     "text":"R2",
            # }
    ],
    "CLASIFICACION":[
            # {
            #     "id":"MATRIZ_DE_CONFUSION",
            #     "text":"matriz de confusion",
            # },
            # {
            #     "id":"ROC",
            #     "text":"ROC",
            # },
            {
                "id":"ENTROPIA_CRUZADA_BINARIA",
                "text":"Entropia cruzada binaria",
            },
            # {
            #     "id":"ENTROPIA_CRUZADA_CATEGORICA",
            #     "text":"Entropia cruzada categorica",
            # }
    ]
}