
from decisionTrees import DecisionTree
from knn import knn
from randomForest import randomForest
from randomForestOptimised import randomForestOptimised
from svm import Support_Vector_Machine
from nn import Neural_Network

# Menu
print("Welcome to the Model building app")
print("Please ensure that your Training Data is in the Training Dataset folder and named Training Data.csv")
print("Please ensure that your Unknow Data set is in the Production Dataset folder and named Unknown Data.csv")      
print("Please select an option from the menu below")
print("1. Decision Tree")
print("2. Random Forest")
print("3. KNN")
print("4. Support Vector Machine")
print("5. Neural Network")
print("6. Random Forest Optimised")

# Get user input
user_input = input("Please enter your selection: ")

if user_input == "1":
    DecisionTree()
elif user_input == "2":
    randomForest()
elif user_input == "3":
    knn()
elif user_input == "4":
    Support_Vector_Machine()
elif user_input == "5":
    Neural_Network()
elif user_input == "6":
    randomForestOptimised()