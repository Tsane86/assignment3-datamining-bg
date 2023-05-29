from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


# Preprocessing as per Workshop 4



def preprocessing(loationOfDataset):
    fulldataset = pd.read_csv(loationOfDataset)
    pd.concat([fulldataset.head(), fulldataset.tail()])

    def convertCategoricaltoNumerical(input_target):
        targets = fulldataset[input_target].unique()
        target2code = dict(zip(targets, range(len(targets))))
        return fulldataset[input_target].replace(target2code)

    # Check the shape the abalone data
    fulldataset.shape

    #drop row ID
    fulldataset = fulldataset.drop(['row ID'], axis=1)

    #convert categorical data to numerical data
    categoriesToConvert = ['HEAT_D','AC', 'STRUCT_D', 'GRADE_D', 'CNDTN_D', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D']
    for category in categoriesToConvert:
        fulldataset[category] = convertCategoricaltoNumerical(category)


    # Performing a Standard scaler transform of the Abalone dataset
        features_to_drop = ['QUALIFIED', 'CNDTN_D', 'AC', 'STYLE_D', 'SALEDATE',
                            'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'GIS_LAST_MOD_DTTM']
    fulldataset = fulldataset.drop(columns=features_to_drop, axis=1)
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(fulldataset)

    # convert the array back to a dataframe
    dataset = pd.DataFrame(data, columns=[
        'row ID', 'QUALIFIED', 'BATHRM', 'HF_BATHRM', 'HEAT', 'HEAT_D', 'AC', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB',
        'STORIES', 'SALEDATE', 'PRICE', 'SALE_NUM', 'GBA', 'BLDG_NUM', 'STYLE', 'STYLE_D', 'STRUCT', 'STRUCT_D',
        'GRADE', 'GRADE_D', 'CNDTN', 'CNDTN_D', 'EXTWALL', 'EXTWALL_D', 'ROOF', 'ROOF_D', 'INTWALL', 'INTWALL_D',
        'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM'
    ])

    # summarize
    dataset.describe()

    # Check mean & Standard dev


preprocessing('Training Dataset\Training Data.csv')


