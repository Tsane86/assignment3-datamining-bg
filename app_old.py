import pandas as pd

# Read CSV file
testingData = pd.read_csv('Training data.csv')

# Print the first 5 rows of the dataframe.
print(testingData.head())

#print type of dataframe
print(type(testingData))

