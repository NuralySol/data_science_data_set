import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Reading the CSV file with try-except block for error handling
try:
    df = pd.read_csv("./carprices.csv")
    print("CSV file loaded successfully!")
    print(df.head())  # Display the first few rows to confirm it was loaded and check the columns!
except FileNotFoundError:
    print("Error: File not found. Please check the path to the 'carprices.csv' file.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file is corrupted or improperly formatted.")

# Prepare X and y for the linear regression model (Mileage vs Sell Price)
X_mileage = df["Mileage"].values.reshape(-1, 1)
y_price = df["Sell Price($)"].values

# Split the data into training and testing sets (80% training, 20% testing)
X_train_mileage, X_test_mileage, y_train_mileage, y_test_mileage = train_test_split(X_mileage, y_price, test_size=0.2, random_state=49)

# Create and train the Linear Regression model for Price vs Mileage
model_mileage = LinearRegression()
model_mileage.fit(X_train_mileage, y_train_mileage)

# Predict the prices for both training and test sets (Mileage vs Price)
y_train_mileage_pred = model_mileage.predict(X_train_mileage)
y_test_mileage_pred = model_mileage.predict(X_test_mileage)

# Define the evaluation function to calculate R2 score and Mean Squared Error
def evaluate_model(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"{set_name} - R^2 Score: {r2:.4f}")
    print(f"{set_name} - Mean Squared Error: {mse:.4f}")
    return r2, mse

# Evaluate the model performance for Price vs Mileage
print("\nMileage vs Price (Training Set)")
train_mileage_score = evaluate_model(y_train_mileage, y_train_mileage_pred, "Training Set")

print("\nMileage vs Price (Test Set)")
test_mileage_score = evaluate_model(y_test_mileage, y_test_mileage_pred, "Test Set")

# Plot: Price vs Mileage with Regression Line on Test Set
plt.figure(figsize=(10, 5))
plt.scatter(X_test_mileage, y_test_mileage, color="blue", label="Actual Prices (Test Set)")
plt.plot(X_test_mileage, y_test_mileage_pred, color="red", label="Regression Line (Test Set)")
plt.xlabel("Mileage")
plt.ylabel("Sell Price ($)")
plt.title("Car Price vs Mileage with Regression Line (Test Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Prepare X and y for Price vs Age prediction
X_age = df["Age(yrs)"].values.reshape(-1, 1)

# Split the data into training and testing sets for Price vs Age
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y_price, test_size=0.2, random_state=49)

# Create and train the Linear Regression model on training data for Price vs Age
model_age = LinearRegression()
model_age.fit(X_train_age, y_train_age)

# Predict the prices for both training and test sets (Age vs Price)
y_train_age_pred = model_age.predict(X_train_age)
y_test_age_pred = model_age.predict(X_test_age)

# Evaluate the model performance for Price vs Age
print("\nAge vs Price (Training Set)")
train_age_score = evaluate_model(y_train_age, y_train_age_pred, "Training Set")

print("\nAge vs Price (Test Set)")
test_age_score = evaluate_model(y_test_age, y_test_age_pred, "Test Set")

# Plot: Price vs Age with Regression Line on Test Set
plt.figure(figsize=(10, 5))
plt.scatter(X_test_age, y_test_age, color="green", label="Actual Prices (Test Set)")
plt.plot(X_test_age, y_test_age_pred, color="red", label="Regression Line (Test Set)")
plt.xlabel("Age (yrs)")
plt.ylabel("Sell Price ($)")
plt.title("Car Price vs Age with Regression Line (Test Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print evaluation results for the test sets
print(f"\nFinal Model Evaluation:")
print(f"Test Set R^2 (Mileage vs Price): {test_mileage_score[0]:.4f}")
print(f"Test Set MSE (Mileage vs Price): {test_mileage_score[1]:.4f}")
print(f"Test Set R^2 (Age vs Price): {test_age_score[0]:.4f}")
print(f"Test Set MSE (Age vs Price): {test_age_score[1]:.4f}")

#! R^2
# R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model.
# In investing, R-squared is generally interpreted as the percentage of a fund’s or security’s price movements that can be explained by movements in a benchmark index.
# An R-squared of 100% means that all movements of a security (or other dependent variable) are completely explained by movements in the index (or whatever independent variable you are interested in).


#! MSE
# Mean squared error (MSE) measures the amount of error in statistical models. It assesses the average squared difference between the observed and predicted values. When a model has no error, the MSE equals zero. As model error increases, its value increases. The mean squared error is also known as the mean squared deviation (MSD).