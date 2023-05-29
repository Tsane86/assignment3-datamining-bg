
import pandas as pd
import math
import sklearn as sklearn
import matplotlib as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from decisionTrees import DecisionTree
from knn import knn
from randomForest import randomForest

# Use to troubleshoot
print(sklearn.__version__)

# Menu
print("Welcome to the Model building app")
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