# Sales-Prediction-using-linear-regression-in-Machine-Learning

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration purposes
np.random.seed(42)
num_samples = 100
advertising_budget = np.random.uniform(0, 10, num_samples)
sales = 3 * advertising_budget + np.random.normal(0, 2, num_samples)

# Create a DataFrame
data = pd.DataFrame({'Advertising_Budget': advertising_budget, 'Sales': sales})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Advertising_Budget']], data['Sales'], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Advertising Budget')
plt.ylabel('Sales')
plt.title('Sales Prediction using Linear Regression')
plt.show()


Output - Mean Squared Error: 2.614798054868012

![image](https://github.com/Tejalwaghmare19/Sales-Prediction-using-linear-regression-in-Machine-Learning/assets/88417292/73ee1d5c-4039-4dc0-bec6-c44073f9c9fd)












