import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset     =   pd.         read_csv("data/Position_Salaries.csv")
X           =   dataset.    iloc[:, 1:-1].   values         # we are only taking the levels (2nd column), 
                                                            # positions (1st column) are not needed
y           =   dataset.    iloc[:, -1].     values         # last column
print       ("Read the data successfully")

# displaying the data (optional)
# print(X)
# print(y)
##################################################
##################################################







# train the decision tree regression model on the whole dataset
##################################################
##################################################
from sklearn.ensemble import  RandomForestRegressor
regressor   =   RandomForestRegressor(n_estimators=10, random_state=0)
regressor.      fit(X, y)
print           ("Trained the Random Forest Regression model successfully")
##################################################
##################################################









# predicting a new result with Decision Tree Regression
##################################################
##################################################
y_pred      =   regressor.      predict([[6.5]])
print           ("Predicted value for input 6.5 is:", y_pred)
##################################################
##################################################






# visualize the Random Forest Regression results (higher resolution)
##################################################
##################################################
X_grid      =   np.        arange   (min(X), max(X), 0.1)        # create a grid with an interval of 0.1
X_grid      =   X_grid.    reshape  ((len(X_grid), 1))          # reshape to be a 2D array with one column
plt.        scatter(X, y, color='red')
plt.        plot(X_grid, regressor.      predict(X_grid), color='blue')
plt.        title   ('Random Forest Regression Results')
plt.        xlabel  ('Position level')
plt.        ylabel  ('Salary')
plt.        show    ()
##################################################
##################################################
