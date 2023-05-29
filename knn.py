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
    full_dataset = pd.read_csv('Training data.csv')

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
    full_dataset = full_dataset.dropna()

    # drop the target from the training set
    features_to_drop = ['QUALIFIED', 'row ID', 'CNDTN_D', 'AC', 'STYLE_D', 'SALEDATE',
                        'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'GIS_LAST_MOD_DTTM']

    X_train = full_dataset.drop(columns=features_to_drop, axis=1)
    y_train = full_dataset['QUALIFIED']

    # Train the KNN classifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    # Calculate and print the accuracy of the training data
    trained_model = clf.predict(X_train)
    accuracy = accuracy_score(y_train, trained_model)
    print(
        f'The accuracy of the model for the test set is {math.floor(accuracy * 100)}%')

    # run an unknow set through the model
    # Read the unknown dataset
    unknown_data = pd.read_csv('Unknown data.csv')

    # Preprocess the unknown dataset same as before
    # convert categorical data to numerical data again
    for category in categoriesToConvert:
        unknown_data[category] = convertCategoricaltoNumerical(category)
    print(unknown_data.head())
    # Drop the same columns as the training dataset (except for qualified, which dosnt exist in the unknown dataset. This is why there is a [1:] in the features_to_drop list)])
    unknown_data = unknown_data.drop(columns=features_to_drop[1:], axis=1)
    #drop missing values
    unknown_data = unknown_data.dropna()

    # Make predictions on the unknown dataset
    unknown_predictions = clf.predict(unknown_data)

    # Add the predicted values to the unknown dataset as a new column
    unknown_data['QUALIFIED'] = unknown_predictions

    # Save the unknown dataset with predictions to a CSV file
    unknown_data.to_csv('unknown_dataset_with_predictions_KNN.csv', index=False)
    


