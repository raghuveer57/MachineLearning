import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Imported the libraries successfully")

# importing the dataset
# read the data set using pandas into a data frame
dataset = pd.read_csv("data/Data1.csv")
X = dataset.iloc[:, :-1].values # all columns except the last
y = dataset.iloc[:, -1].values # last column
print("Read the data successfully")

# displaying the data
print("X: ",X)
print("y: ",y)