
from decisionTrees import DecisionTree
from knn import knn
from randomForest import randomForest

# Menu
print("Welcome to the Model building app")
print("Please ensure that your Training Data is in the Training Dataset folder and named Training Data.csv")
print("Please ensure that your Unknow Data set is in the Production Dataset folder and named Unknown Data.csv")      
print("Please select an option from the menu below")
print("1. Decision Tree")
print("2. Random Forest")
print("3. KNN")

# Get user input
user_input = input("Please enter your selection: ")

if user_input == "1":
    DecisionTree()
elif user_input == "2":
    randomForest()
elif user_input == "3":
    knn()