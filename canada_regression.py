import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


try:
    df = pd.read_csv("./canada_per_capita_income.csv")
    print("CSV file loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print(
        "Error: File not found. Please check the path to the 'canada_per_capita_income.csv' file."
    )
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file is corrupted or improperly formatted.")

# Sort by year
canada_per_capita = df.sort_values(by="year", ascending=True)

# Prepare X and y for the linear regression model and store them in variables
X = canada_per_capita["year"].values.reshape(-1, 1)
y = canada_per_capita["per capita income (US$)"].values

# Create and fit the linear regression model using X, and y 
model = LinearRegression()
model.fit(X, y)
predicted_income = model.predict(X)

# Predict the income for 2024 and 2025 using np.array hold two values
future_year = np.array([[2024], [2025]])
predicted_income_2024_2025 = model.predict(future_year)

# Combine the original years with future years with flatten like a spreader in JS
all_years = np.concatenate([X.flatten(), future_year.flatten()])
all_predicted_income = np.concatenate([predicted_income, predicted_income_2024_2025])

# Plot the original data points
plt.figure(figsize=(8, 6))
plt.scatter(
    canada_per_capita["year"],
    canada_per_capita["per capita income (US$)"],
    color="blue",
    label="Per Capita Income (US$)",
)

# Origional plot style
plt.plot(
    canada_per_capita["year"], predicted_income, color="red", linestyle="-", label="Regression Line"
)

# Continuation of the plot using dashed line for model prediction up to 2025 year
plt.plot(
    np.concatenate([X.flatten()[-1:], future_year.flatten()]),
    np.concatenate([predicted_income[-1:], predicted_income_2024_2025]),
    color="green",
    linestyle="--",
    label="Prediction Line (2024-2025)",
)

#! two distinct scatters for the model for easier data vis
plt.scatter(
    future_year[0],
    predicted_income_2024_2025[0],
    color="orange",
    marker="$\u21E0$",
    s=200,
    label="Predicted 2024",
)
plt.scatter(
    future_year[1],
    predicted_income_2024_2025[1],
    color="purple",
    marker="$\u21E0$",
    s=200,
    label="Predicted 2025",
)

# for loop with enumirate and position of labels on the plot 
for i, year in enumerate(future_year.flatten()):
    offset = (20, -15) if i % 2 == 0 else (-15, 10)

    plt.annotate(
        f"${predicted_income_2024_2025[i]:,.2f}",
        (year, predicted_income_2024_2025[i]),
        textcoords="offset points",
        xytext=offset,
        ha="right" if i % 2 == 0 else "left",
        va="bottom",
        fontsize=8,
        color="black",
        bbox=dict(boxstyle="round,pad=0.1", edgecolor="gray", facecolor="lightyellow"),
    )

plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Per Capita Income of Canada with Regression Line and Predictions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print the predicted values for 2024 and 2025 for print output
print(f"Predicted per capita income for 2024: ${predicted_income_2024_2025[0]:,.2f}")
print(f"Predicted per capita income for 2025: ${predicted_income_2024_2025[1]:,.2f}")