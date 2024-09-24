import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import pickle

# Load the Boston housing dataset (for educational purposes only)
boston = datasets.load_boston()

# Convert the dataset into a DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Define feature matrix X and target vector y
X = data.drop(columns=['PRICE'])
y = data['PRICE']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions (optional, to check performance)
predictions = model.predict(X_test)
print(f"First 5 predictions: {predictions[:5]}")

# Save the trained model to a pickle file
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as linear_regression_model.pkl")
