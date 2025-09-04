import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset = pd.read_csv("data/Salary_Data.csv")
X = dataset.iloc[:, :-1].values # all columns except the last
y = dataset.iloc[:, -1].values # last column
print("Read the data successfully")

# displaying the data (optional)
# print(X)
# print(y)
##################################################
##################################################





# Splitting the dataset into the Training set and Test set
##################################################
##################################################
from sklearn.model_selection import train_test_split

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print the new data sets (optional)
# print("X_train:")
# print(X_train)
# print("X_test:")
# print(X_test)
# print("y_train:")
# print(y_train)
# print("y_test:")
# print(y_test)
##################################################
##################################################

