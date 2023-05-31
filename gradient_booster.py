import numpy as np
import pandas as pd
import math
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

def gradient_boosting():

    # Read CSV file
    full_dataset = pd.read_csv('Training Dataset\\Training Data.csv')

    # Print the first 5 rows of the full dataframe
    print(full_dataset.head())

    #print length of dataframe
    length = len(full_dataset)
    print((f'Before dropping, the length is: {length}'))

    # output basic statistics using the describe() method from pandas
    print(full_dataset.describe())
    basic_stats = full_dataset.describe().reset_index()
    #round basic stats to 2 decimal places
    basic_stats = basic_stats.round(2)
    #output to csv
    basic_stats.to_csv('basic_stats.csv', index=False)

    def convertCategoricaltoNumerical(input_target):
        targets = full_dataset[input_target].unique()
        target2code = dict(zip(targets, range(len(targets))))
        return full_dataset[input_target].replace(target2code)
    
    
    #clean data (missing values, outliers, etc.)
    #if column has more than 50% missing values, drop the column
    full_dataset = full_dataset.dropna(thresh=0.5*len(full_dataset), axis=1)


    #convert any categorical data to numerical data (from Workshop 7)
    #convert categorical data to numerical data
    categoriesToConvert = ['HEAT_D', 'AYB', 'AC', 'CNDTN_D', 'STYLE_D',
                           'STRUCT_D', 'GRADE_D', 'BEDRM', 'GBA', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D']
    for category in categoriesToConvert:
        full_dataset[category] = convertCategoricaltoNumerical(category)
    print(full_dataset.head())
    

    #TODO Work out better features, see feature selection workshop

    print('Feature Selection Wizard')

    print('Please enter the column name you wish to use as the Target')
    print('For example: QUALIFIED')
    target = input('Enter the target you wish to use: ')
    data_set_without_target = full_dataset.drop(target, axis=1)
    data_set_without_target = data_set_without_target.drop('row ID', axis=1)

    print('Please choose the desired features for the model')
    print('Column names are: ')
    #print the column names and seperate with a comma
    for col in data_set_without_target.columns:
        print(col, end=', ')
    
    #print on new line
    print('\n')

    print('Please enter the column names you wish to drop as Features, separated by a comma')
    print('For example: HEAT_D, AYB, CNDTN_D, BEDRM, GBA, SALEDATE')
    
    features_to_drop = input('Enter the features you wish to drop: ')
    print('You have chosen the following features to drop: ', features_to_drop)
    features_to_drop = features_to_drop.split(', ')

    # convert the SALEDATE to just the year
    for index, row in data_set_without_target.iterrows():
        data_set_without_target.at[index,'SALEDATE'] = row['SALEDATE'][:4]

    #drop row
    print('Dropping the following features: ', features_to_drop)

    # drop the columns from the training set
    X = data_set_without_target.drop(columns=features_to_drop, axis=1)
    y = full_dataset['QUALIFIED']

    # imputer to handle missing values as per this article https://machinelearningmastery.com/handle-missing-data-python/
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns)

    #print length of dataframe
    length = len(X_imputed)
    print((f'After dropping, the length is: {length}'))

    # These are for testing purposes only
    #print(X.head())
    # output a csv of X
    #X.to_csv('X.csv', index=False)
    #print(y.head())

    #split the dataset into training and testing datasets (from workshop 9)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3)

    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    # Construct a random forest classifier.
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # We can print the classifier like this.
    print(clf)

    # We can make predictions using this classifier like this.
    y_pred = clf.predict(X_test)

    # F1 score. As per this documentation https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # calculate the Recall score
    recall = recall_score(y_test, y_pred)
    print(f'The Recall score is {recall}')

    #print f1 scores
    print("F1 Weighted score: {:.4f}".format(f1_weighted))
    print("F1 Micro score: {:.4f}".format(f1_micro))
    print("F1 Macro score: {:.4f}".format(f1_macro))

    # get information about the importance of the features used in the model, as per this article https://machinelearningmastery.com/calculate-feature-importance-with-python/
    # get the feature importances
    importances = clf.feature_importances_

    # make a DataFrame to store feature importance and sort it
    importance_data = pd.DataFrame(
        {'Feature': X.columns, 'Importance': importances})
    importance_data = importance_data.sort_values(by='Importance', ascending=False)

    #output the improtance data to a csv
    importance_data.to_csv('Training Predictions\\feature_importance_gb.csv', index=False)

    #making Prediction
    clf.score(X_test, y_test)

    # Confusion matrix as per Workshop 7
    mat = confusion_matrix(y_test, y_pred)
    print(mat)

    # output the prediction results to a csv with the original data
    X_test['QUALIFIED'] = y_test
    X_test['PREDICTION'] = y_pred
    X_test.to_csv(
        'Training Predictions\\prediction_results_bg.csv', index=False)

    # calculate the accuracy of the model by comparing the predicted values with the actual values
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(
        f'The accuracy of the model for the test set is {math.floor(accuracy * 100)}%')

    # run an unknow set through the model
    # Read the unknown dataset
    unknown_data = pd.read_csv('Production dataset\\Unknown data.csv')

    # Preprocess the unknown dataset same as before
    # convert categorical data to numerical data again
    for category in categoriesToConvert:
        unknown_data[category] = convertCategoricaltoNumerical(category)
    print(unknown_data.head())

    # convert the SALEDATE to just the year
    for index, row in unknown_data.iterrows():
        unknown_data.at[index, 'SALEDATE'] = row['SALEDATE'][:4]

    #if column has more than 50% missing values, drop the column
    unknown_data = unknown_data.dropna(thresh=0.5*len(unknown_data), axis=1)

    # Drop the same columns as the training dataset (except for qualified, which dosnt exist in the unknown dataset.)
    unknown_data_input = unknown_data.drop(
        columns=features_to_drop, axis=1)
    unknown_data_input = unknown_data_input.drop(columns=['row ID'], axis=1)

    # imputer to handle missing values as per this article https://machinelearningmastery.com/handle-missing-data-python/
    imputer = SimpleImputer(strategy='mean')
    Unknow_imputed = pd.DataFrame(
        imputer.fit_transform(unknown_data_input), columns=unknown_data_input.columns)

    # Make predictions on the unknown dataset
    unknown_predictions = clf.predict(Unknow_imputed)

    # Add the predicted values to the unknown dataset as a new column
    unknown_data['Predict-Qualified'] = unknown_predictions

    # Save the unknown dataset with predictions to a CSV file
    unknown_data.drop(columns=[
        'BATHRM', 'HF_BATHRM', 'HEAT', 'HEAT_D', 'AC', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'EYB',
        'STORIES', 'SALEDATE', 'PRICE', 'SALE_NUM', 'GBA', 'BLDG_NUM', 'STYLE', 'STYLE_D', 'STRUCT', 'STRUCT_D',
        'GRADE', 'GRADE_D', 'CNDTN', 'CNDTN_D', 'EXTWALL', 'EXTWALL_D', 'ROOF', 'ROOF_D', 'INTWALL', 'INTWALL_D',
        'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA', 'GIS_LAST_MOD_DTTM'
    ]).to_csv(
        'Unknown Predictions\\unknown_dataset_with_predictions_gb.csv', index=False)
