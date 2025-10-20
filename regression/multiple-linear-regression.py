import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset = pd.read_csv("data/50_Startups.csv")
X = dataset.iloc[:, :-1].values # all columns except the last
y = dataset.iloc[:, -1].values # last column
print("Read the data successfully")

# displaying the data (optional)
# print(X)
# print(y)
##################################################
##################################################











#Encoding categorical variables.##################

#Encoding independent variable X
##################################################
##################################################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create the column transformer
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

# Fit and transform the data
X = np.array(ct.fit_transform(X)) # force the ouput into a numpy array

# print(X)
##################################################
##################################################













# Splitting the dataset into the Training set and Test set
##################################################
##################################################
from sklearn.model_selection import train_test_split

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print the new data sets
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






#  Training the multiple linear regression model
##################################################
##################################################

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # fit the model to the training data
print("Model trained successfully")

##################################################
##################################################



 

# Predicting the Test set results
##################################################
##################################################
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) # set precision for printing
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # compare predicted and actual values

##################################################
##################################################  