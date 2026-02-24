import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error





# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1] .values  # All columns except the last (features)
y = dataset.iloc[:,  -1] .values   # Last column (target: PE)
print("Dataset loaded successfully. Shape:", dataset.shape)
print("Features:", dataset.columns[:-1].tolist())
print("Target:", dataset.columns[-1])



# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train/test sets.")



# Dictionary to store results
results = {}



# Function to evaluate and store results
def evaluate_model(name, y_true, y_pred):
    r2      = r2_score(y_true, y_pred)
    mse     = mean_squared_error(y_true, y_pred)
    mae     = mean_absolute_error(y_true, y_pred)
    rmse    = np.sqrt(mse)
    
    results[name] = {
        'R2 Score': r2,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }

    print(f"\n{name} Results:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")




# 1. Multiple Linear Regression
print       ("\n" + "="*50)
print       ("Training Multiple Linear Regression...")
mlr         = LinearRegression()
mlr         .fit(X_train, y_train)
y_pred_mlr  = mlr.predict(X_test)
evaluate_model("Multiple Linear Regression", y_test, y_pred_mlr)




# 2. Polynomial Regression (degree 2)
print           ("\n" + "="*50)
print           ("Training Polynomial Regression (degree 2)...")
poly_features   = PolynomialFeatures(degree=2)
X_train_poly    = poly_features.fit_transform(X_train)
X_test_poly     = poly_features.transform(X_test)
plr             = LinearRegression()
plr             .fit(X_train_poly, y_train)
y_pred_plr      = plr.predict(X_test_poly)
evaluate_model  ("Polynomial Regression (degree 2)", y_test, y_pred_plr)



# 3. Decision Tree Regression
print               ("\n" + "="*50)
print               ("Training Decision Tree Regression...")
dtr                 = DecisionTreeRegressor(random_state=42)
dtr                 .fit(X_train, y_train)
y_pred_dtr          = dtr.predict(X_test)
evaluate_model      ("Decision Tree Regression", y_test, y_pred_dtr)



# 4. Random Forest Regression
print               ("\n" + "="*50)
print               ("Training Random Forest Regression...")
rfr                 = RandomForestRegressor(n_estimators=100, random_state=42)
rfr                 .fit(X_train, y_train)
y_pred_rfr          = rfr.predict(X_test)
evaluate_model      ("Random Forest Regression", y_test, y_pred_rfr)




# 5. Support Vector Regression
print       ("\n" + "="*50)
print       ("Training Support Vector Regression...")
# Feature scaling for SVR
sc_X            = StandardScaler()
sc_y            = StandardScaler()
X_train_scaled  = sc_X.fit_transform(X_train)
X_test_scaled   = sc_X.transform(X_test)
y_train_scaled  = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

svr                 = SVR(kernel='rbf')
svr                 .fit(X_train_scaled, y_train_scaled)
y_pred_svr_scaled   = svr.predict(X_test_scaled)
y_pred_svr          = sc_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()
evaluate_model      ("Support Vector Regression", y_test, y_pred_svr)



# Summary of all results
print       ("\n" + "="*60)
print       ("SUMMARY OF ALL REGRESSION MODELS")
print       ("="*60)
print       (f"{'Model':<30} {'R2 Score':<10} {'RMSE':<10}")
print       ("-" * 50)
for model, metrics in results.items():
    print       (f"{model:<30} {metrics['R2 Score']:<10.4f} {metrics['RMSE']:<10.4f}")

# Find the best model
best_model = max(results, key=lambda x: results[x]['R2 Score'])
print       (f"\nBest performing model: {best_model} (R2 = {results[best_model]['R2 Score']:.4f})")

print       ("\nAll models trained and evaluated successfully!")