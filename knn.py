from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import sklearn as sklearn
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn():
    # Read CSV file
    full_dataset = pd.read_csv('Training Dataset\\Training Data.csv')

    def convertCategoricaltoNumerical(input_target):
        targets = full_dataset[input_target].unique()
        target2code = dict(zip(targets, range(len(targets))))
        return full_dataset[input_target].replace(target2code)

    categoriesToConvert = ['HEAT_D', 'AC', 'STRUCT_D',
                           'GRADE_D', 'CNDTN_D', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D']
    for category in categoriesToConvert:
        full_dataset[category] = convertCategoricaltoNumerical(category)
    print(full_dataset.head())

    # clean
    #drop rows with missing values
    #full_dataset = full_dataset.dropna()

    #replace any missing values with the mean of the column
    full_dataset = full_dataset.fillna(full_dataset.mean())

    # drop the target from the training set
    features_to_drop = ['QUALIFIED', 'row ID', 'CNDTN_D', 'AC', 'STYLE_D', 'SALEDATE',
                        'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'GIS_LAST_MOD_DTTM']
    X = full_dataset.drop(columns=features_to_drop, axis=1)
    y = full_dataset['QUALIFIED']

    #split the dataset into training and testing datasets (from workshop 9)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train the KNN classifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate and print the accuracy of the training data
    trained_model = clf.predict(X_train)
    accuracy = accuracy_score(y_train, trained_model)
    print(
        f'The accuracy of the model for the test set is {math.floor(accuracy * 100)}%')
    
    # output the prediction results to a csv with the original data
    X_test['QUALIFIED'] = y_test
    X_test['PREDICTION'] = y_pred
    X_test.to_csv(
        'Training Predictions\\prediction_results_knn.csv', index=False)

    # run an unknow set through the model
    # Read the unknown dataset
    unknown_data = pd.read_csv('Production dataset\\Unknown data.csv')

    # Preprocess the unknown dataset same as before
    # convert categorical data to numerical data again
    for category in categoriesToConvert:
        unknown_data[category] = convertCategoricaltoNumerical(category)
    print(unknown_data.head())

    #drop missing values
    unknown_data = unknown_data.dropna()

    # Drop the same columns as the training dataset (except for qualified, which dosnt exist in the unknown dataset. This is why there is a [1:] in the features_to_drop list)])
    unknown_data_input = unknown_data.drop(
        columns=features_to_drop[1:], axis=1)

    # Make predictions on the unknown dataset
    unknown_predictions = clf.predict(unknown_data_input)

    # Add the predicted values to the unknown dataset as a new column
    unknown_data['Predict-Qualified'] = unknown_predictions

    # Save the unknown dataset with predictions to a CSV file
    unknown_data.drop(columns=[
        'BATHRM', 'HF_BATHRM', 'HEAT', 'HEAT_D', 'AC', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB',
        'STORIES', 'SALEDATE', 'PRICE', 'SALE_NUM', 'GBA', 'BLDG_NUM', 'STYLE', 'STYLE_D', 'STRUCT', 'STRUCT_D',
        'GRADE', 'GRADE_D', 'CNDTN', 'CNDTN_D', 'EXTWALL', 'EXTWALL_D', 'ROOF', 'ROOF_D', 'INTWALL', 'INTWALL_D',
        'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM'
    ]).to_csv(
        'Unknow Predictions\\unknown_dataset_with_predictions_knn.csv', index=False)
    


