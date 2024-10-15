import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Reading the CSV file
try:
    df = pd.read_csv("./chipotle.csv")
    print("CSV file loaded successfully!")
    print(df.head())  # Display the first few rows to confirm it was loaded
except FileNotFoundError:
    print("Error: File not found. Please check the path to the 'chipotle.csv' file.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file is corrupted or improperly formatted.")

# Check if 'item_price' exists in the DataFrame
if 'item_price' in df.columns:
    # Remove the dollar sign ($) and convert to float
    df['item_price'] = df['item_price'].str.replace('$', '').astype(float)
    print("\n'item_price' column successfully converted to float!")
else:
    print("\n'item_price' column not found!")

# Display summary statistics for the DataFrame
print(df.describe())

# Find the most expensive item
most_expensive_item = df[df["item_price"] == df["item_price"].max()][["item_name", "item_price"]]
print("Most expensive item:")
print(most_expensive_item)

# How many unique items in the dataset
unique_items = df["item_name"].nunique()
print(f"Unique items in the dataset: {unique_items}")

# Chicken Bowl Analysis
chicken_bowl_orders = df[df["item_name"] == "Chicken Bowl"]
total_chicken_bowls = chicken_bowl_orders["quantity"].sum()
print(f"Total number of Chicken Bowls ordered: {total_chicken_bowls}")
chicken_bowl_revenue = (chicken_bowl_orders['quantity'] * chicken_bowl_orders['item_price']).sum()
print(f"Total revenue from Chicken Bowls: ${chicken_bowl_revenue:.2f}")

# Total store revenue
sum_of_revenue = df["item_price"].sum()
print(f"Total revenue of the store: ${sum_of_revenue:.2f}")
percentage_of_chicken = (chicken_bowl_revenue / sum_of_revenue) * 100
print(f"Percentage of revenue from Chicken Bowls: {percentage_of_chicken:.2f}%")

# Canned Soda orders with quantity greater than 1
can_soda_more_than_one = df[(df["item_name"] == "Canned Soda") & (df["quantity"] > 1)]
print(f"Canned Soda orders with quantity greater than one:")
print(can_soda_more_than_one)

# Group by item_name to see the total quantity and revenue per item The .agg() method in Pandas is used to apply one or more aggregation functions (such as sum(), mean(), min(), max(), etc.) to grouped data. It is often used in combination with the groupby() method to summarize data for each group in a flexible way. To use .agg pass in keys of .method such as "sum", "mean", "min", "max" , "nunique" etc.
grouped_items = df.groupby("item_name").agg({
    "quantity": "sum", 
    "item_price": "sum"  
})
# Custom function example
grouped_custom_lambda = df.groupby('item_name').agg({
    'quantity': ['sum', 'mean', lambda x: x.sum() * 2],  #! Custom lambda function 
    'item_price': ['sum', 'mean', lambda x: x.max() - x.min()]  #! Custom lambda
})

# Display the grouped results
print(f"\nGrouped data (total quantity and total revenue for each item): {grouped_items}")
print(f"\nGrouped data using the cumstom functions: {grouped_custom_lambda}")

# Get concise summary of the DataFrame
df.info()

# Display the DataFrame
print(df)

#! pass in chaining methods
# Corrected chaining of methods: Group by 'item_name', sum 'item_price', and sort in descending order
df_revenue = df.groupby("item_name")["item_price"].sum().sort_values(ascending=False)

# Print the result
print(f"Revenue of each item in descending order:\n{df_revenue}")


#! revenue by percentage of the total sales
df_revenue = df.groupby("item_name")[["item_price"]].sum()
total_revenue = df_revenue["item_price"].sum()
df_revenue["percent_of_total"] = (df_revenue["item_price"] / total_revenue) * 100
print(f"\nTotal revenue of the store: ${total_revenue:.2f}\nPercentage of total revenue for each item:")
print(df_revenue.to_string(formatters={"percent_of_total": "{:.2f}%".format}))

#! plot the five order sellers by percentage, .head() with 5 param.

top_5_revenue = df_revenue.sort_values(by="percent_of_total", ascending=False).head(5)
plt.figure(figsize=(8, 8))
#! autopct is needed for display of percentages of each segment of pie chart. 
plt.pie(top_5_revenue["percent_of_total"], labels=top_5_revenue.index, autopct='%1.2f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Top 5 Items by Percentage of Total Revenue") # title of the plotly
# display the plotly 
plt.show()


#! need a cusmtom function to display % and the prices alongside the pie chart.
def autopct_format(values):
    def inner_autopct(pct):
        total = sum(values)
        absolute = int(pct/100.*total)
        return f"${absolute:,}\n({pct:.2f}%)"
    return inner_autopct

#! call in the custom function in the params of .pie() method.
top_5_revenue_by_dollars = df_revenue.sort_values(by="item_price", ascending=False).head(5)
plt.figure(figsize=(8, 8))
plt.pie(top_5_revenue_by_dollars["item_price"], labels=top_5_revenue_by_dollars.index,
        autopct=autopct_format(top_5_revenue_by_dollars["item_price"]), startangle=140, colors=plt.cm.Paired.colors)

plt.title("Top 5 Items by Total Revenue (Dollar Amount and Percentage)")
plt.show()

