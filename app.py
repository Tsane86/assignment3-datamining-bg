import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Menu


# Read CSV file
testingData = pd.read_csv('Training data.csv')

# Print the first 5 rows of the dataframe.
print(testingData.head())

#print type of dataframe
print(type(testingData))

#print length of dataframe
print(len(testingData))

#print Column Qualified
print(testingData['QUALIFIED'])

# Print Data types in a table
data_types = testingData.dtypes.reset_index()
data_types.columns = ['Column', 'Data Type']
print(data_types.to_string(index=False))

# output basic statistics using the describe() method from pandas
print(testingData.describe())
basic_stats = testingData.describe().reset_index()
#round basic stats to 2 decimal places
basic_stats = basic_stats.round(2)
#output to csv
basic_stats.to_csv('basic_stats.csv', index=False)

# output value counts of Column QUALIFIED using Pandas
print(testingData['QUALIFIED'].value_counts())

#check for Missing values. If yes, display a percentage of missing values
total_length = len(testingData['QUALIFIED'])
missing_values = testingData['QUALIFIED'].isnull().sum()
percentage_missing = (missing_values / total_length) * 100
print(percentage_missing)
if percentage_missing > 20: #if more than 20% of the data is missing, drop the rows of missing values
    rows_without_missing_data = testingData.dropna()
    rows_without_missing_data.shape
    print('More thanb 20% missing, removing affected rows')
elif 1 <= percentage_missing <= 20: #if less than 20%, replace missing with mean
    testingData['QUALIFIED'].fillna(testingData['QUALIFIED'].mean(), inplace=True)
    print('Less than 20% missing values, replacing with mean')
elif percentage_missing == 0:
    print('No missing values')

min_value = testingData['QUALIFIED'].min()
max_value = testingData['QUALIFIED'].max()

invalid_entries = testingData[(testingData['QUALIFIED'] < min_value) | (
    testingData['QUALIFIED'] > max_value)]

if not invalid_entries.empty:
    print("Invalid entries found in the QUALIFIED column:")
    print(invalid_entries)
else:
    unique_data_types = testingData['QUALIFIED'].apply(type).unique()
    if len(unique_data_types) == 1:
        print("All entries in the QUALIFIED column have the same data type:",
              unique_data_types[0])
    else:
        print("Entries in the QUALIFIED column have different data types.")

min_value = testingData['QUALIFIED'].min()
max_value = testingData['QUALIFIED'].max()

# test for invalid entries in QUALIFIED column
invalid_entries = testingData[(testingData['QUALIFIED'] < min_value) | (
    testingData['QUALIFIED'] > max_value)]

if not invalid_entries.empty:
    print("Invalid entries found in the QUALIFIED column:")
    print(invalid_entries)

    # Remove invalid entries from DataFrame
    testingData = testingData.drop(invalid_entries.index)

#test that all data types are the same
unique_data_types = testingData['QUALIFIED'].apply(type).unique()
if len(unique_data_types) == 1:
    print("All entries in the QUALIFIED column have the same data type:",
          unique_data_types[0])
else:
    print("Entries in the QUALIFIED column have different data types:")
    for data_type in unique_data_types:
        rows = testingData.loc[testingData['QUALIFIED'].apply(
            type) == data_type]
        print(f"Data type {data_type}:")
        print(rows)

#test for outliers
def detect_outliers(df, column_name):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find the outliers
    outliers = df.loc[(df[column_name] < lower_bound) | (
        df[column_name] > upper_bound), column_name]

    if not outliers.empty:
        print("Outliers found in the {} column:".format(column_name))
        print(outliers)
    else:
        print("No outliers found in the {} column.".format(column_name))


# Use function
detect_outliers(testingData, 'QUALIFIED')

# Check for Normalisation
def check_normalization(dataFrame, data_column):
    column = dataFrame[data_column]
    min_val = column.min()
    max_val = column.max()
    range_val = max_val - min_val
    std_val = column.std()

    return range_val > 1 or std_val > 1

# if check normalisation is true, Normalise the data
if check_normalization(testingData, 'QUALIFIED'):
    print('QUALIFIED is not normalized')
    # Normalize the data
    testingData['QUALIFIED'] = (testingData['QUALIFIED'] - testingData['QUALIFIED'].min()) / (testingData['QUALIFIED'].max() - testingData['QUALIFIED'].min())
    print(testingData['QUALIFIED'])
    print('Data is now Normalised')
else:
    print('QUALIFIED is normalised')
    

def perform_clustering(df, column_name, num_clusters):
    column = df[column_name].values.reshape(-1, 1)
    scaler = StandardScaler()
    column_scaled = scaler.fit_transform(column)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(column_scaled)
    labels = kmeans.labels_

    return labels


# Example usage

cluster_labels = perform_clustering(testingData, 'QUALIFIED', 2)
print(cluster_labels)
print(len(cluster_labels))


testingData['cluster_label'] = cluster_labels
testingData.to_csv('clustered_data.csv', index=False)

#test the data against the training set


def testData(testing_data, column1, column2):
    counter = 0

    for val1, val2 in zip(testing_data[column1], testing_data[column2]):
        if val1 == val2:
            counter += 1

    matching_percentage = (counter / len(testing_data[column1])) * 100

    return matching_percentage


