import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load user interactions dataset
user_interactions_df = pd.read_csv('user_interactions.csv')

# Load purchase data dataset
purchase_data_df = pd.read_csv('purchase_data.csv')

# Load website performance dataset
website_performance_df = pd.read_csv('website_performance.csv')


print("User Interactions Data:")
print(user_interactions_df.head())

print("\nData Shapes:")
print("User Interactions:", user_interactions_df.shape)

print("\nData Statistics:")
print(user_interactions_df.describe())

print("\nMissing Values:")
print(user_interactions_df.isnull().sum())

print("\nPurchase Data:")
print(purchase_data_df.head())

print("\nData Shapes:")
print("Purchase Data:", purchase_data_df.shape)

print("\nData Statistics:")
print(purchase_data_df.describe())

print("\nMissing Values:")
print(purchase_data_df.isnull().sum())

print("\nMissing Values:")
print(purchase_data_df.isnull().sum())

print("\nData Shapes:")
print("Website Performance:", website_performance_df.shape)

print("\nData Statistics:")
print(website_performance_df.describe())

print("\nMissing Values:")
print(website_performance_df.isnull().sum())

page_views = user_interactions_df['Page_Viewed'].value_counts()

average_time_per_page = user_interactions_df.groupby('Page_Viewed')['Time_Spent_Seconds'].mean()

most_visited_pages = page_views.sort_values(ascending=False)

print("Top 10 Most Visited Pages:")
print(most_visited_pages.head(10))

print("\nAverage Time Spent on Each Page:")
print(average_time_per_page)

clicks_by_section = user_interactions_df.groupby('Page_Viewed')['User_ID'].count()

# Calculate the number of page views for each section
page_views_by_section = user_interactions_df['Page_Viewed'].value_counts()

# Calculate the click-through rate (CTR) for each section
ctr_by_section = (clicks_by_section / page_views_by_section) * 100

print("\nClick-Through Rate for Different Sections:")
print(ctr_by_section)

# Visualize the most visited pages using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(most_visited_pages.index[:10], most_visited_pages[:10])
plt.xlabel('Page')
plt.ylabel('Number of Page Views')
plt.title('Top 10 Most Visited Pages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Visualize the average time spent on each page using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(average_time_per_page.index, average_time_per_page)
plt.xlabel('Page')
plt.ylabel('Average Time Spent (seconds)')
plt.title('Average Time Spent on Each Page')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize the click-through rate for different sections using a bar chart
plt.figure(figsize=(8, 6))
plt.bar(ctr_by_section.index, ctr_by_section)
plt.xlabel('Section')
plt.ylabel('Click-Through Rate (%)')
plt.title('Click-Through Rate for Different Sections')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

total_purchases = purchase_data_df['Purchase_ID'].nunique()

total_users_with_purchase = purchase_data_df['User_ID'].nunique()


conversion_rate = (total_users_with_purchase / total_purchases) * 100

print("Overall Conversion Rate: {:.2f}%".format(conversion_rate))

product_sales = purchase_data_df.groupby('Product_ID')['Transaction_Amount'].sum()

top_selling_products = product_sales.sort_values(ascending=False)

print("\nTop 10 Top-Selling Products:")
print(top_selling_products.head(10))

data = {
    'Product_ID': ['prod001', 'prod002', 'prod003', 'prod004'],
    'Product_Category': ['Category1', 'Category2', 'Category1', 'Category3']
}
product_id_to_category_df = pd.DataFrame(data)

merged_df = purchase_data_df.merge(product_id_to_category_df, on='Product_ID')

category_sales = merged_df.groupby('Product_Category')['Transaction_Amount'].sum()


top_selling_categories = category_sales.sort_values(ascending=False)

print("\nTop 5 Top-Selling Product Categories:")
print(top_selling_categories.head(5))

total_interactions = user_interactions_df.groupby('User_ID').size().reset_index(name='Total_Interactions')

merged_interactions_df = purchase_data_df.merge(total_interactions, on='User_ID')

conversion_by_interactions = (
    merged_interactions_df.groupby('User_ID')['Purchase_ID'].nunique() / merged_interactions_df['Total_Interactions']
)

plt.figure(figsize=(8, 6))
plt.scatter(merged_interactions_df['Total_Interactions'], merged_interactions_df['Purchase_ID'], alpha=0.5)
plt.xlabel('Number of Interactions')
plt.ylabel('Likelihood of Purchase')
plt.title('Relationship Between Interactions and Likelihood of Purchase')
plt.tight_layout()
plt.show()

earliest_interaction_df = user_interactions_df.groupby('User_ID')['Time_Spent_Seconds'].idxmin()
user_interactions_df['SignUpDate'] = user_interactions_df.loc[earliest_interaction_df, 'Time_Spent_Seconds']
user_interactions_df['SignUpMonth'] = pd.to_datetime(user_interactions_df['SignUpDate']).dt.to_period('M')


cohort_df = user_interactions_df.groupby('SignUpMonth')['User_ID'].nunique().reset_index()

cohort_df.columns = ['SignUpMonth', 'TotalUsers']

user_total_amount = purchase_data_df.groupby('User_ID')['Transaction_Amount'].sum()


average_transaction_amount = user_total_amount.mean()

average_order_frequency = purchase_data_df.groupby('User_ID')['Purchase_ID'].nunique().mean()

average_lifetime_span = (
    website_performance_df['Month'].nunique() - cohort_df['SignUpMonth'].nunique()
) + 1

cohort_df['LTV'] = average_transaction_amount * average_order_frequency * average_lifetime_span


print(cohort_df)

print(website_performance_df.describe())

average_page_load_time = website_performance_df['Page_Load_Time'].mean()
average_bounce_rate = website_performance_df['Bounce_Rate'].mean()

plt.figure(figsize=(10, 6))
plt.plot(website_performance_df['Month'], website_performance_df['Page_Load_Time'], marker='o')
plt.xlabel('Month')
plt.ylabel('Average Page Load Time')
plt.title('Average Page Load Time Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(website_performance_df['Month'], website_performance_df['Bounce_Rate'], marker='o', color='red')
plt.xlabel('Month')
plt.ylabel('Bounce Rate')
plt.title('Bounce Rate Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(website_performance_df['Month'], website_performance_df['Page_Load_Time'], marker='o')
plt.xlabel('Month')
plt.ylabel('Average Page Load Time')
plt.title('Average Page Load Time Over Time')
plt.xticks(rotation=45)
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(website_performance_df['Month'], website_performance_df['Bounce_Rate'], marker='o', color='red')
plt.xlabel('Month')
plt.ylabel('Bounce Rate')
plt.title('Bounce Rate Over Time')
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()

