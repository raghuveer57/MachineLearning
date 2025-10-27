import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset = pd.read_csv("data/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values # we are only taking the levels (2nd column), positions (1st column) are not needed
y = dataset.iloc[:, -1].values # last column
print("Read the data successfully")

# displaying the data (optional)
# print(X)
# print(y)
##################################################
##################################################





# feature scaling
##################################################
##################################################
# We have to use two separate scalers for X and y because y is a different scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X) # scale X
y = y.reshape(len(y), 1) # reshape y to be a 2D array with one column
sc_y = StandardScaler()
y = sc_y.fit_transform(y) # scale y
y = y.flatten() # flatten y back to 1D array to be compatible with the SVR model

# displaying the scaled data (optional)
# print(X)
# print(y)

print("Feature scaling completed successfully")
##################################################
##################################################







# train the SVR model on the whole dataset
##################################################
##################################################
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # using the radial basis function kernel
regressor.fit(X, y)
print("Trained the SVR model successfully")
##################################################
##################################################






# predict a new result
##################################################
##################################################
# Predicting a new result with SVR
y_pred = regressor.predict(sc_X.transform([[6.5]])) # we need to scale the input value
y_pred = sc_y.inverse_transform([y_pred]).reshape(-1,1) # we need to inverse transform the output value
print("Predicted value for input 6.5 is:", y_pred)
##################################################
##################################################




# visualize the SVR results
##################################################
##################################################
plt.scatter(sc_X.inverse_transform(X.reshape(len(X),1)), sc_y.inverse_transform(y.reshape(len(y),1)), color='red') # original data points
plt.plot(sc_X.inverse_transform(X.reshape(len(X),1)), sc_y.inverse_transform(regressor.predict(X).reshape(len(X),1)), color='blue') # SVR model predictions
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
##################################################
##################################################



# visualize the SVR results (for higher resolution and smoother curve)
##################################################
##################################################
X_grid = np.arange(min(sc_X.inverse_transform(X.reshape(len(X),1))), max(sc_X.inverse_transform(X.reshape(len(X),1))), 0.01) # create a grid with a step of 0.01
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X.reshape(len(X),1)), sc_y.inverse_transform(y.reshape(len(y),1)), color='red') # original data points
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(len(X_grid),1)), color='blue') # SVR model predictions
plt.title('Truth or Bluff (SVR) - High Resolution')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
##################################################
##################################################