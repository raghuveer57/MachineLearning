import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Imported the libraries successfully")

# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset = pd.read_csv("data/Data1.csv")
X = dataset.iloc[:, :-1].values # all columns except the last
y = dataset.iloc[:, -1].values # last column
print("Read the data successfully")

# displaying the data
print(X)
print(y)
##################################################
##################################################



# handling missing values
##################################################
##################################################
print("Handling the missing values \n")
from sklearn.impute import SimpleImputer

# Create the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) # fit only on the columns that are numbers
X[:, 1:3] = imputer.transform(X[:, 1:3]) # transform the data

print(X)
##################################################
##################################################




#Encoding categorical variables


#Encoding independent variable X
##################################################
##################################################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create the column transformer
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Fit and transform the data
X = np.array(ct.fit_transform(X)) # force the ouput into a numpy array

print(X)
##################################################
##################################################


#Encoding dependant variable y
##################################################
##################################################
from sklearn.preprocessing import LabelEncoder

# Create the label encoder
le = LabelEncoder()

# Fit and transform the data
y = le.fit_transform(y)

print(y)
##################################################
##################################################
