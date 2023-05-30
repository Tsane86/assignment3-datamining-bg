from sklearn.model_selection import train_test_split
import pandas as pd
import math
import sklearn as sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def Neural_Network():
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
    #if column has more than 50% missing values, drop the column
    full_dataset = full_dataset.dropna(thresh=0.5*len(full_dataset), axis=1)

    # convert the SALEDATE to just the year
    for index, row in full_dataset.iterrows():
        full_dataset.at[index, 'SALEDATE'] = row['SALEDATE'][:4]

    # drop the target from the training set
    features_to_drop = ['QUALIFIED', 'row ID', 'HEAT', 'HEAT_D', 'FIREPLACES', 'INTWALL', 'HF_BATHRM', 'GRADE_D', 'STRUCT', 'STRUCT_D', 'STORIES',
                        'USECODE', 'STYLE', 'KITCHENS', 'NUM_UNITS', 'BLDG_NUM', 'CNDTN_D', 'AC', 'STYLE_D', 'GIS_LAST_MOD_DTTM', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'EXTWALL']
    X = full_dataset.drop(columns=features_to_drop, axis=1)
    y = full_dataset['QUALIFIED']

    # Standard Scaler function from Workshop 4
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)

    # imputer to handle missing values as per this article https://machinelearningmastery.com/handle-missing-data-python/
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_scaled), columns=X.columns)


    #split the dataset into training and testing datasets (from workshop 9)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3)

    # Train the KNN classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 100))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # calculate the Recall score
    recall = recall_score(y_test, y_pred)
    print(f'The Recall score is {recall}')

    # Calculate the F1 score. As per this documentation https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    #print f1 scores
    print("F1 Weighted score: {:.4f}".format(f1_weighted))
    print("F1 Micro score: {:.4f}".format(f1_micro))
    print("F1 Macro score: {:.4f}".format(f1_macro))
    
    # Calculate and print the accuracy of the training data
    trained_model = clf.predict(X_train)
    accuracy = accuracy_score(y_train, trained_model)
    print(
        f'The accuracy of the model for the test set is {math.floor(accuracy * 100)}%')
    
    # output the prediction results to a csv with the original data
    X_test['QUALIFIED'] = y_test
    X_test['PREDICTION'] = y_pred
    X_test.to_csv(
        'Training Predictions\\prediction_results_nn.csv', index=False)

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

    # Standard Scaler function from Workshop 4
    std_scaler = StandardScaler()
    unknown_scaled = std_scaler.fit_transform(unknown_data_input)

    # imputer to handle missing values as per this article https://machinelearningmastery.com/handle-missing-data-python/
    imputer = SimpleImputer(strategy='mean')
    Unknow_imputed = pd.DataFrame(
        imputer.fit_transform(unknown_scaled), columns=unknown_data_input.columns)

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
        'Unknown Predictions\\unknown_dataset_with_predictions_nn.csv', index=False)
    


