import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import seaborn as sns

# Reading the CSV file
try:
    df = pd.read_csv("./housing.csv")
    print("CSV file loaded successfully!")
    print(df.head())  # Display the first few rows to confirm it was loaded
except FileNotFoundError:
    print("Error: File not found. Please check the path to the 'housing.csv' file.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file is corrupted or improperly formatted.")

# Fill missing values in total_bedrooms with the mean
mean_total_bedrooms = df["total_bedrooms"].mean()
df["total_bedrooms"] = df["total_bedrooms"].fillna(mean_total_bedrooms)

# Feature Engineering: Creating new features
df["average_bedrooms_per_house"] = df["total_bedrooms"] / df["households"]
df["people_per_household"] = df["population"] / df["households"]

# Scatter plot for population with longitude and latitude
plt.figure(figsize=(10, 6))
plt.scatter(df["longitude"], df["latitude"], s=df["population"]/100, cmap="plasma", c='blue', alpha=0.5, label="Population")
plt.scatter(-118.2437, 34.0522, color='red', marker='*', s=400, label="Los Angeles", edgecolors="black")
plt.scatter(-122.4194, 37.7749, color='orange', marker='*', s=400, label="San Francisco", edgecolors="black")
plt.annotate("Los Angeles", (-118.2437, 34.0522), xytext=(10, 0), textcoords='offset points', color="orange", fontsize=10)
plt.annotate("San Francisco", (-122.4194, 37.7749), xytext=(10, 0), textcoords='offset points', color="orange", fontsize=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Population Distribution with Longitude and Latitude")

plt.legend()
plt.tight_layout()
plt.show()


# # Preprocess categorical data: one-hot encoding for 'ocean_proximity'
# df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# # Define X (independent variables) and y (dependent variable)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
# Drop rows where 'median_house_value' is greater than 500000 permanently
df.drop(df[df['median_house_value'] > 500000].index, inplace=True)

#! Plotting the histogram for median house values
plt.figure(figsize=(10, 6))
plt.hist(df["median_house_value"], bins=50, ec='black', color='orange', edgecolor='black')
plt.xlabel('House Values')
plt.ylabel('Frequency')
plt.title('Histogram of Median House Values')

# Calculate the mean of the median_house_value
mean_value = df["median_house_value"].mean()

# Add a vertical line at the mean value
plt.axvline(mean_value, color="black", linestyle='dashed', linewidth=3, label="Average Price in a block")

# Annotate the mean value in the plot
plt.annotate(f'Mean: ${mean_value:,.0f}', xy=(mean_value, 25), xytext=(mean_value + 30000, 30), 
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color="white")

# Add a legend to the plot
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Calculate the correlation between 'median_house_value' and 'median_income'
correlation = df['median_income'].corr(df['median_house_value'])
print(f"Correlation between median house value and median income: {correlation:.2f}")

# Remove non-numeric columns before calculating the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Seaborn heatmap for correlation matrix 
plt.figure(figsize=(10, 6))
correlation_matrix = numeric_df.corr()

# Create a heatmap using Seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

# Add title to the heatmap
plt.title("Correlation Matrix of Housing Data")

# Rotate x and y axis ticks for better readability
plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
plt.yticks(rotation=0)   # Ensure y-axis labels are horizontal

# Display the heatmap
plt.tight_layout()
plt.show()

#! Bar chart based on ocean proximity. 
# Group the data by 'ocean_proximity' and calculate the sum of the number of households
households_by_proximity = df.groupby('ocean_proximity')['households'].sum()

# Get the number of households in the 'NEAR BAY' area
households_near_bay = households_by_proximity['NEAR BAY']
print(f"Number of households in NEAR BAY area: {households_near_bay}")

# Create a bar plot for all ocean proximity categories
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(households_by_proximity.index, households_by_proximity, color='blue', edgecolor='black')

# Highlight the 'NEAR BAY' bar by changing its color
bars[list(households_by_proximity.index).index('NEAR BAY')].set_color('orange')

# Add bar labels for the values
ax.bar_label(bars, fmt='%d')

# Add labels and title
ax.set_xlabel('Ocean Proximity')
ax.set_ylabel('Number of Households')
ax.set_title('Number of Households Based on Ocean Proximity')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()