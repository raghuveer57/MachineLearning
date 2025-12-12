import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset     =   pd.         read_csv("data/Salary_Data.csv")
X           =   dataset.    iloc[:, :-1].   values # all columns except the last
y           =   dataset.    iloc[:, -1].    values # last column
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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





#  Training the simple linear regression model
##################################################
##################################################

from sklearn.linear_model import LinearRegression
regressor       =       LinearRegression()
regressor.  fit(X_train, y_train)       # fit the model to the training data
print       ("Model trained successfully")

##################################################
##################################################





# Predict the test set results
##################################################
##################################################

y_pred      =       regressor.predict(X_test)

# print(" values:")
# print(X_test)

# print("Predicted values:")
# print(y_pred)

##################################################
##################################################




# Plotting the training set results
##################################################
##################################################

plt.    scatter         (X_train, y_train, color='red') # actual values
plt.    plot            (X_train, regressor.    predict(X_train), color='blue') # predicted values
plt.    title           ('Salary vs Experience (Training set)')
plt.    xlabel          ('Years of Experience')
plt.    ylabel          ('Salary')
plt.    show            ()

##################################################
##################################################




# Plotting the test set results
##################################################
##################################################

plt.    scatter(X_test, y_test, color='red') # actual values
plt.    plot(X_test, y_pred, color='blue') # predicted values
plt.    title('Salary vs Experience (Test set)')
plt.    xlabel('Years of Experience')
plt.    ylabel('Salary')
plt.    show()
##################################################
##################################################