import numpy as np
import pandas as pd
import math
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer


def randomForestOptimised():

    # from Workshops slides 9
    # Set random seed for reproducing results
    np.random.seed(422)

    # Read CSV file
    full_dataset = pd.read_csv('Training Dataset\\Training Data.csv')

    # Print the first 5 rows of the full dataframe
    print(full_dataset.head())

    #print type of dataframe
    print(type(full_dataset))

    #print length of dataframe
    length = len(full_dataset)
    print((f'Before dropping, the length is: {length}'))

    # Print Data types in a table
    data_types = full_dataset.dtypes.reset_index()
    data_types.columns = ['Column', 'Data Type']
    print(data_types.to_string(index=False))

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
    

    #TODO Work out better features, see feature selection workshop

    #convert any categorical data to numerical data (from Workshop 7)
    #clean data (missing values, outliers, etc.)
    #if column has more than 50% missing values, drop the column
    full_dataset = full_dataset.dropna(thresh=0.5*len(full_dataset), axis=1)
    #convert categorical data to numerical data
    categoriesToConvert = ['HEAT_D', 'AYB', 'CNDTN_D', 'BEDRM', 'GBA']
    for category in categoriesToConvert:
        full_dataset[category] = convertCategoricaltoNumerical(category)
    print(full_dataset.head())

    # convert the SALEDATE to just the year
    for index, row in full_dataset.iterrows():
        full_dataset.at[index,'SALEDATE'] = row['SALEDATE'][:4]

    # drop the target from the training set
    features_to_drop = ['QUALIFIED', 'row ID', 'HEAT', 'HEAT_D', 'FIREPLACES', 'INTWALL','HF_BATHRM','GRADE_D', 'STRUCT', 'STRUCT_D', 'STORIES', 
                        'USECODE', 'STYLE', 'KITCHENS', 'NUM_UNITS', 'BLDG_NUM', 'CNDTN_D', 'AC', 'STYLE_D', 'GIS_LAST_MOD_DTTM', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'EXTWALL']
    X = full_dataset.drop(columns=features_to_drop, axis=1)
    y = full_dataset['QUALIFIED']

    # imputer to handle missing values as per this article https://machinelearningmastery.com/handle-missing-data-python/
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

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
    clf = RandomForestClassifier(n_estimators=50, oob_score=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # We can print the classifier like this.
    print(clf)

    # We can make predictions using this classifier like this.
    y_pred = clf.predict(X_test)

    # get information about the features used in the model, as per this article https://machinelearningmastery.com/calculate-feature-importance-with-python/
    # get the feature importances
    importances = clf.feature_importances_

    # make a DataFrame to store feature importance and sort it
    importance_data = pd.DataFrame(
        {'Feature': X.columns, 'Importance': importances})
    importance_data = importance_data.sort_values(by='Importance', ascending=False)

    #output the improtance data to a csv
    importance_data.to_csv('Training Predictions\\feature_importance.csv', index=False)

    #making Prediction
    clf.score(X_test, y_test)

    # Confusion matrix as per Workshop 7
    mat = confusion_matrix(y_test, y_pred)
    print(mat)

    # output the prediction results to a csv with the original data
    X_test['QUALIFIED'] = y_test
    X_test['PREDICTION'] = y_pred
    X_test.to_csv(
        'Training Predictions\\prediction_results_RF.csv', index=False)

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

    # Drop the same columns as the training dataset (except for qualified, which dosnt exist in the unknown dataset. This is why there is a [1:] in the features_to_drop list)])
    unknown_data_input = unknown_data.drop(
        columns=features_to_drop[1:], axis=1)

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
        'Unknown Predictions\\unknown_dataset_with_predictions_RF.csv', index=False)
