import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
bitcoin_data = pd.read_csv('bitcoin.csv')  

# Prepare data
x = bitcoin_data['Open'].values.reshape(-1, 1)  # Independent variable
y = bitcoin_data['Close'].values  # Dependent variable

# Linear regression model
model = LinearRegression()
model.fit(x, y)

# Predictions
y_pred = model.predict(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Values')  # Actual values in blue
plt.plot(x, y_pred, color='red', label='Predicted Values')  # Predicted values in red
plt.title('Linear Regression of Bitcoin Prices')
plt.xlabel('Open Prices')
plt.ylabel('Close Prices')
plt.legend()
plt.grid(True)
plt.show()
