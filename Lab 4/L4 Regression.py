# Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

iris = datasets.load_iris()
X = iris.data[:, :2]  # Select the first two features
y = iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lr_model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("Coefficient of determination (R^2):", r2_score(y_test, y_pred))
