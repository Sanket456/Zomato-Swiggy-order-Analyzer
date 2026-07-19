import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# STEP 1: GENERATE REALISTIC TRANSACTION DATA
# ==========================================
# Instead of hunting for a dirty CSV file, we are writing a localized data generator 
# that builds a rich, realistic history of Zomato & Swiggy orders over the last year.

np.random.seed(42)  # Ensures the random data looks exactly the same every time you run it
num_orders = 200

# Generate a list of dates spanning the last 365 days
start_date = datetime(2025, 7, 1)
date_list = [start_date + timedelta(days=int(np.random.randint(0, 365))) for _ in range(num_orders)]

# Realistic Indian food ordering choices
restaurants_cuisines = {
    "Biryani By Kilo": "North Indian",
    "Haldiram's": "North Indian",
    "Wow! Momo": "Chinese",
    "Burger King": "Fast Food",
    "Domino's": "Fast Food",
    "Sagar Ratna": "South Indian",
    "Baskin Robbins": "Desserts",
    "Local Dhaba": "North Indian"
}
restaurant_names = list(restaurants_cuisines.keys())

# Construct the raw dictionary dataset
raw_data = {
    "Order_ID": [f"ORD{1000 + i}" for i in range(num_orders)],
    "Date": date_list,
    "Platform": np.random.choice(["Zomato", "Swiggy"], size=num_orders, p=[0.55, 0.45]),
    "Restaurant": np.random.choice(restaurant_names, size=num_orders),
    "Bill_Amount": np.random.randint(250, 1200, size=num_orders),  # Bill in INR (₹)
    "Delivery_Charge": np.random.choice([0, 30, 45, 60], size=num_orders, p=[0.4, 0.3, 0.2, 0.1]),
    "Discount_Applied": np.random.choice([0, 50, 75, 100], size=num_orders, p=[0.3, 0.4, 0.2, 0.1]),
    "Payment_Mode": np.random.choice(["UPI", "Credit Card", "Net Banking"], size=num_orders, p=[0.7, 0.2, 0.1])
}

# Load the raw data dictionary directly into a beautiful Pandas DataFrame table
df = pd.DataFrame(raw_data)

# Map the correct cuisines to the randomly chosen restaurants
df["Cuisine"] = df["Restaurant"].map(restaurants_cuisines)

# THIS IS THE LINE THAT WAS MISSING! It calculates the 'Final_Paid' column.
df["Final_Paid"] = df["Bill_Amount"] + df["Delivery_Charge"] - df["Discount_Applied"]

print("====================================================")
print("📡 RAW ORDER TRANSACTION LOGS GENERATED SUCCESSFULLY")
print("====================================================")
print(df.head(5))  # Prints the first 5 rows to show what our dataset looks like
print("\n")

# ==========================================
# STEP 2: DATA ENGINEERING & FEATURE EXTRACTION
# ==========================================
# To analyze habits by month and days of the week, we need to extract details from the 'Date' column.

# Convert the Date column to an official Pandas Datetime object so Pandas understands time
df["Date"] = pd.to_datetime(df["Date"])

# Extract the Month Name (e.g., January, February)
df["Month"] = df["Date"].dt.strftime('%B')

# Extract the Day Name (e.g., Monday, Sunday)
df["Day_Of_Week"] = df["Date"].dt.day_name()

# Determine if the order happened on a Weekend vs Weekday using a quick conditional check
df["Is_Weekend"] = df["Day_Of_Week"].isin(["Saturday", "Sunday"]).map({True: "Weekend", False: "Weekday"})


# ==========================================
# STEP 3: ADVANCED DATA AGGREGATION (.groupby)
# ==========================================
print("====================================================")
print("📊 PERFORMANCE REPORT: METRICS & AGGREGATIONS")
print("====================================================")

# --- INSIGHT 1: Platform Battle (Zomato vs Swiggy) ---
# We group by the 'Platform' column and sum the 'Final_Paid' column to see where most cash goes.
platform_spend = df.groupby("Platform")["Final_Paid"].sum()
print("1. Total Lifetime Spend by Platform:")

# Loop through the platform_spend results correctly
for platform, amount in platform_spend.items():
    print(f"   • {platform}: ₹{amount:,}")
    
print(f"   💸 Total Food Delivery Cost: ₹{df['Final_Paid'].sum():,}\n")

# --- INSIGHT 2: Top 3 Most Frequented Restaurants ---
# Group by Restaurant, count the number of Order_IDs, and sort from highest to lowest.
fav_restaurants = df.groupby("Restaurant").agg(
    Total_Orders=("Order_ID", "count"),
    Total_Spent=("Final_Paid", "sum")
).sort_values(by="Total_Orders", ascending=False).head(3)

print("2. Top 3 Favorite Restaurants (By Order Count):")
print(fav_restaurants)
print("\n")

# --- INSIGHT 3: Cuisine Breakdown ---
# Find out which type of food you crave the most
cuisine_breakdown = df.groupby("Cuisine")["Final_Paid"].agg(["count", "sum", "mean"]).rename(
    columns={"count": "Total Orders", "sum": "Total Spent", "mean": "Avg Order Value"}
)
print("3. Expense Breakdown by Cuisine Type:")
print(cuisine_breakdown.round(2))
print("\n")

# --- INSIGHT 4: Weekend Laxity vs Weekday Discipline ---
# Group by Weekend vs Weekday to see if order behavior changes when the weekend hits.
weekend_analysis = df.groupby("Is_Weekend").agg(
    Avg_Bill_Size=("Final_Paid", "mean"),
    Total_Orders=("Order_ID", "count")
)
print("4. Ordering Patterns: Weekdays vs Weekends:")
print(weekend_analysis.round(2))
print("\n")

# --- INSIGHT 5: Delivery Charges vs Discounts (Payment Mode Analysis) ---
# Do certain payment modes attract better savings or lower friction?
payment_savings = df.groupby("Payment_Mode").agg(
    Avg_Discount=("Discount_Applied", "mean"),
    Avg_Delivery_Fee=("Delivery_Charge", "mean")
)
print("5. Efficiency Metrics by Payment Method:")
print(payment_savings.round(2))
print("====================================================")
