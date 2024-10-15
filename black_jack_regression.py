import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Simulating a sample dataset for a blackjack game
# Features: card_count (favorable/unfavorable), bet_size, player_hand_value, dealer_hand_value
# Target: Outcome (1 for win, 0 for loss)
data = {
    "card_count": [1, -2, 0, 1, 2, -1, 0, -3, 2, 1],
    "bet_size": [50, 100, 50, 25, 75, 50, 100, 200, 25, 50],
    "player_hand_value": [18, 21, 17, 20, 19, 16, 21, 15, 20, 19],
    "dealer_hand_value": [19, 18, 17, 20, 21, 17, 19, 20, 18, 17],
    "outcome": [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # 1 for win, 0 for loss
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare X and y for the regression model
X = df[["card_count", "bet_size", "player_hand_value", "dealer_hand_value"]]
y = df["outcome"]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the outcomes for both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Custom function to evaluate the model
def evaluate_model(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"{set_name} - R^2 Score: {r2:.4f}")
    print(f"{set_name} - Mean Squared Error: {mse:.4f}")
    return r2, mse

# Evaluate the model performance on both training and test sets
print("\nBlackjack Outcome Prediction (Training Set)")
train_score = evaluate_model(y_train, y_train_pred, "Training Set")

print("\nBlackjack Outcome Prediction (Test Set)")
test_score = evaluate_model(y_test, y_test_pred, "Test Set")

# Visualize actual vs predicted outcomes for the test set
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Outcomes (Test Set)")
plt.scatter(range(len(y_test)), y_test_pred, color="red", label="Predicted Outcomes (Test Set)", marker='x')
plt.xlabel("Game Index")
plt.ylabel("Outcome (1 = Win, 0 = Loss)")
plt.title("Actual vs Predicted Outcomes for Blackjack Test Set")
plt.legend()
plt.tight_layout()
plt.show()