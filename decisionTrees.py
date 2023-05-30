import pandas as pd
import math
import sklearn as sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer



def DecisionTree():
    # Read CSV file
    full_dataset = pd.read_csv('Training Dataset\Training Data.csv')

    # Print the first 5 rows of the full dataframe
    print(full_dataset.head())

    #print type of dataframe
    print(type(full_dataset))

    #print length of dataframe
    length = len(full_dataset)
    print((f'Before dropping, the length is: {length}'))

    # Print Data types in a table
    #data_types = full_dataset.dtypes.reset_index()
    #data_types.columns = ['Column', 'Data Type']
    #print(data_types.to_string(index=False))


    # output basic statistics using the describe() method from pandas
    #print(full_dataset.describe())
    #basic_stats = full_dataset.describe().reset_index()
    #round basic stats to 2 decimal places
    #basic_stats = basic_stats.round(2)
    #output to csv
    #basic_stats.to_csv('basic_stats.csv', index=False)

    def convertCategoricaltoNumerical(input_target):
        targets = full_dataset[input_target].unique()
        target2code = dict(zip(targets, range(len(targets))))
        return full_dataset[input_target].replace(target2code)

    #convert any categorical data to numerical data (from Workshop 7)
    #clean data (missing values, outliers, etc.)

    #if column has more than 50% missing values, drop the column
    full_dataset = full_dataset.dropna(thresh=0.5*len(full_dataset), axis=1)


    #convert categorical data to numerical data
    categoriesToConvert = ['HEAT_D','AC', 'STRUCT_D', 'GRADE_D', 'CNDTN_D', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D']
    for category in categoriesToConvert:
        full_dataset[category] = convertCategoricaltoNumerical(category)
    print(full_dataset.head())

    #TODO have the app print out the columsn and let the user select which features they want. Add the differences to features to drop variable
    # drop the target from the training set and drop features not needed
    features_to_drop = ['QUALIFIED', 'row ID', 'CNDTN_D', 'AC', 'STYLE_D', 'SALEDATE',
                        'EXTWALL_D', 'ROOF_D', 'INTWALL_D', 'GIS_LAST_MOD_DTTM']
    X = full_dataset.drop(columns=features_to_drop, axis=1)
    y = full_dataset['QUALIFIED']

    #TODO add imputer to all models
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

    #split the dataset into training and testing datasets (from workshop 7)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=1234)

    # create a decision tree classifier (Workshop 7 material)
    clf = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=3)
    clf.fit(X_imputed, y)
    tree.plot_tree(clf)

    # do DT with the training set
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Making Prediction
    clf.score(X_test, y_test)

    # Confusion matrix as per Workshop 7
    mat = confusion_matrix(y_test, y_pred)
    print(mat)

    # output the prediction results to a csv with the original data
    X_test['QUALIFIED'] = y_test
    X_test['PREDICTION'] = y_pred
    X_test.to_csv('Training Predictions\prediction_results_DT.csv', index=False)

    # calculate the accuracy of the model by comparing the predicted values with the actual values
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f'The accuracy of the model for the test set is {math.floor(accuracy * 100)}%')


    # output the confusion matrix as a heatmap
    # Using the methods from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    #disp = ConfusionMatrixDisplay(confusion_matrix=mat)
    #disp.plot()
    #plt.pyplot.show()

    # run an unknow set through the model
    # Read the unknown dataset
    unknown_data = pd.read_csv('Production dataset\\Unknown data.csv')

    # Preprocess the unknown dataset same as before
    # convert categorical data to numerical data again
    for category in categoriesToConvert:
        unknown_data[category] = convertCategoricaltoNumerical(category)
    print(unknown_data.head())

    #print length of dataframe
    length = len(unknown_data)
    print((f'Before dropping, the Unknow Data length is: {length}'))

    #if column has more than 50% missing values, drop the column
    unknown_data = unknown_data.dropna(thresh=0.5*len(unknown_data), axis=1)

    #print length of dataframe
    length = len(unknown_data)
    print((f'After dropping, the Unknown Data length is: {length}'))

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
        'Unknow Predictions\\unknown_dataset_with_predictions_DT.csv', index=False)