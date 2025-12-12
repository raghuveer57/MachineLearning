import numpy                as      np
import matplotlib.pyplot    as      plt
import pandas               as      pd





# importing the dataset
##################################################
##################################################
# read the data set using pandas into a data frame
dataset     =   pd.         read_csv("data/Position_Salaries.csv")
X           =   dataset.    iloc[:, 1:-1].   values    # we are only taking the 
                                                       # levels (2nd column), positions
                                                       # (1st column) are not needed
y           =   dataset.    iloc[:, -1].    values     # last column
print("Read the data successfully")

# displaying the data (optional)
# print(X) 
# print(y)
##################################################
##################################################





# train the entire data set with Linear Regression
##################################################
##################################################
from sklearn.linear_model import LinearRegression
lin_reg     =   LinearRegression()
lin_reg.    fit(X, y) 
print       ("Trained the Linear Regression model successfully")
##################################################
##################################################





# train the entire data set with Polynomial Regression
##################################################
##################################################
from sklearn.preprocessing import PolynomialFeatures
poly_reg    =   PolynomialFeatures(degree=4) # you can change the degree here
X_poly      =   poly_reg.   fit_transform(X)
lin_reg_2   =   LinearRegression()
lin_reg_2.  fit(X_poly, y)
print       ("Trained the Polynomial Regression model successfully")
##################################################
##################################################





# visualize the Linear Regression results
##################################################
##################################################
plt.        scatter(X, y, color='red')
plt.        plot(X, lin_reg.    predict(X), color='blue')
plt.        title('Truth or Bluff (Linear Regression)')
plt.        xlabel('Position level')
plt.        ylabel('Salary')
plt.        show()
##################################################
##################################################





# visualize the Polynomial Regression results
##################################################
##################################################
X_grid      =   np.        arange(X.        min(), X.        max(), 0.1) # for higher resolution and smoother curve
X_grid      =   X_grid.    reshape((len(X_grid), 1))
plt.        scatter(X, y, color='red')
plt.        plot(X_grid, lin_reg_2.   predict(poly_reg.   transform(X_grid)), color='blue')
plt.        title('Truth or Bluff (Polynomial Regression)')
plt.        xlabel('Position level')
plt.        ylabel('Salary')
plt.        show()
##################################################
##################################################





# predict a new result with Linear Regression
##################################################
##################################################
level_to_predict     =   6.5
linear_pred          =   lin_reg.    predict([[level_to_predict]])
print(f"Linear Regression prediction for level {level_to_predict}: {linear_pred[0]}")
##################################################
##################################################





# predict a new result with Polynomial Regression
##################################################
##################################################
poly_pred    =   lin_reg_2.  predict(poly_reg.   transform([[level_to_predict]]))
print(f"Polynomial Regression prediction for level {level_to_predict}: {poly_pred[0]}")
##################################################
##################################################