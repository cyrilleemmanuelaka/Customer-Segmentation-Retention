# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
print("📥 Loading dataset...")
df = pd.read_csv('online_retail.csv', encoding='latin1')
print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print("\n" + "="*60 + "\n")

# Display first few rows
print("📋 First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*60 + "\n")

# Basic info
print("📊 Dataset Information:")
print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

print("🔍 EXPLORATORY DATA ANALYSIS")
print("="*60)

# 1. Check for duplicate rows
print(f"1. Duplicate rows: {df.duplicated().sum():,}")

# 2. Check unique values
print("\n2. Unique values in key columns:")
for col in ['InvoiceNo', 'StockCode', 'CustomerID', 'Country']:
    print(f"   {col}: {df[col].nunique():,} unique values")

# 3. Check date range
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"\n3. Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"   Time span: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")

# 4. Check for cancelled transactions
df['IsCancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
print(f"\n4. Cancelled transactions: {df['IsCancelled'].sum():,} ({df['IsCancelled'].mean()*100:.2f}%)")

# 5. Check for negative quantities
negative_qty = (df['Quantity'] < 0).sum()
print(f"\n5. Negative quantities: {negative_qty:,} ({negative_qty/len(df)*100:.2f}%)")

# 6. Check for zero/negative prices
invalid_price = (df['UnitPrice'] <= 0).sum()
print(f"6. Invalid prices (≤ 0): {invalid_price:,} ({invalid_price/len(df)*100:.2f}%)")

# 7. Top countries
print("\n7. Top 10 Countries by transaction count:")
country_counts = df['Country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"   {country}: {count:,} ({count/len(df)*100:.2f}%)")

# 8. Basic statistics for numerical columns
print("\n8. Basic Statistics:")
print(df[['Quantity', 'UnitPrice']].describe())

print("🧹 DATA CLEANING")
print("="*60)

# Create a copy for cleaning
df_clean = df.copy()

# 1. Remove cancelled transactions
initial_rows = len(df_clean)
df_clean = df_clean[~df_clean['IsCancelled']]
print(f"1. Removed cancelled transactions: {initial_rows - len(df_clean):,} rows removed")
print(f"   Remaining rows: {len(df_clean):,}")

# 2. Remove rows with negative quantities
initial_rows = len(df_clean)
df_clean = df_clean[df_clean['Quantity'] > 0]
print(f"\n2. Removed negative quantities: {initial_rows - len(df_clean):,} rows removed")
print(f"   Remaining rows: {len(df_clean):,}")

# 3. Remove rows with zero or negative prices
initial_rows = len(df_clean)
df_clean = df_clean[df_clean['UnitPrice'] > 0]
print(f"\n3. Removed invalid prices: {initial_rows - len(df_clean):,} rows removed")
print(f"   Remaining rows: {len(df_clean):,}")

# 4. Handle missing CustomerIDs
missing_customers = df_clean['CustomerID'].isnull().sum()
print(f"\n4. Missing CustomerIDs: {missing_customers:,} ({missing_customers/len(df_clean)*100:.2f}%)")

# Option 1: Remove rows with missing CustomerID (for customer segmentation)
df_clean = df_clean.dropna(subset=['CustomerID'])
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

print(f"   After removing missing CustomerIDs: {len(df_clean):,} rows")

# 5. Calculate TotalSpend
df_clean['TotalSpend'] = df_clean['Quantity'] * df_clean['UnitPrice']

# 6. Extract date features
df_clean['Year'] = df_clean['InvoiceDate'].dt.year
df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean['Day'] = df_clean['InvoiceDate'].dt.day
df_clean['Weekday'] = df_clean['InvoiceDate'].dt.weekday
df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
df_clean['Date'] = df_clean['InvoiceDate'].dt.date

# Show cleaning summary
print("\n" + "="*60)
print("✅ DATA CLEANING SUMMARY:")
print("="*60)
print(f"Original dataset: {len(df):,} rows")
print(f"Cleaned dataset: {len(df_clean):,} rows")
print(f"Data retained: {len(df_clean)/len(df)*100:.2f}%")
print(f"\nSample of cleaned data:")
print(df_clean.head())

print("👥 CUSTOMER-LEVEL ANALYSIS")
print("="*60)

# Set a snapshot date (one day after the last transaction)
snapshot_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
print(f"Snapshot date for RFM analysis: {snapshot_date}")

# Create customer-level dataframe
customer_data = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency (number of transactions)
    'TotalSpend': 'sum',     # Monetary value
    'Quantity': 'sum',       # Total items purchased
    'StockCode': 'nunique',  # Unique products purchased
    'Country': 'first'       # Customer's country
}).reset_index()

# Rename columns
customer_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                         'TotalQuantity', 'UniqueProducts', 'Country']

# Calculate additional metrics
customer_data['AvgTransactionValue'] = customer_data['Monetary'] / customer_data['Frequency']
customer_data['AvgItemsPerTransaction'] = customer_data['TotalQuantity'] / customer_data['Frequency']

print(f"Total unique customers: {len(customer_data):,}")
print("\nCustomer Data Sample:")
print(customer_data.head())
print("\nCustomer Data Statistics:")
print(customer_data[['Recency', 'Frequency', 'Monetary']].describe())

print("📊 VISUALIZING CUSTOMER DISTRIBUTIONS")
print("="*60)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Customer Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Recency Distribution
axes[0, 0].hist(customer_data['Recency'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Recency Distribution (Days since last purchase)')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].axvline(customer_data['Recency'].median(), color='red', linestyle='--', 
                  label=f'Median: {customer_data["Recency"].median():.0f} days')
axes[0, 0].legend()

# 2. Frequency Distribution (log scale for better visualization)
axes[0, 1].hist(customer_data['Frequency'], bins=50, edgecolor='black', alpha=0.7, log=True)
axes[0, 1].set_title('Frequency Distribution (Log Scale)')
axes[0, 1].set_xlabel('Number of Transactions')
axes[0, 1].set_ylabel('Number of Customers (log)')
axes[0, 1].axvline(customer_data['Frequency'].median(), color='red', linestyle='--',
                  label=f'Median: {customer_data["Frequency"].median():.0f}')

# 3. Monetary Value Distribution (log scale)
axes[0, 2].hist(customer_data['Monetary'], bins=50, edgecolor='black', alpha=0.7, log=True)
axes[0, 2].set_title('Monetary Value Distribution (Log Scale)')
axes[0, 2].set_xlabel('Total Spend (£)')
axes[0, 2].set_ylabel('Number of Customers (log)')
axes[0, 2].axvline(customer_data['Monetary'].median(), color='red', linestyle='--',
                  label=f'Median: £{customer_data["Monetary"].median():.2f}')

# 4. Scatter: Frequency vs Monetary
scatter = axes[1, 0].scatter(customer_data['Frequency'], customer_data['Monetary'], 
                           alpha=0.5, s=10, c=customer_data['Recency'], cmap='viridis')
axes[1, 0].set_title('Frequency vs Monetary Value')
axes[1, 0].set_xlabel('Frequency (Number of Transactions)')
axes[1, 0].set_ylabel('Monetary Value (£)')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
plt.colorbar(scatter, ax=axes[1, 0], label='Recency (Days)')

# 5. Scatter: Recency vs Monetary
scatter = axes[1, 1].scatter(customer_data['Recency'], customer_data['Monetary'], 
                           alpha=0.5, s=10, c=customer_data['Frequency'], cmap='plasma')
axes[1, 1].set_title('Recency vs Monetary Value')
axes[1, 1].set_xlabel('Recency (Days)')
axes[1, 1].set_ylabel('Monetary Value (£)')
axes[1, 1].set_yscale('log')
plt.colorbar(scatter, ax=axes[1, 1], label='Frequency')

# 6. Country Distribution (Top 10)
top_countries = customer_data['Country'].value_counts().head(10)
axes[1, 2].barh(range(len(top_countries)), top_countries.values)
axes[1, 2].set_yticks(range(len(top_countries)))
axes[1, 2].set_yticklabels(top_countries.index)
axes[1, 2].set_title('Top 10 Countries by Customer Count')
axes[1, 2].set_xlabel('Number of Customers')
axes[1, 2].invert_yaxis()

plt.tight_layout()
plt.show()

# Print key statistics
print("\n📈 KEY CUSTOMER STATISTICS:")
print("="*60)
print(f"Total Customers: {len(customer_data):,}")
print(f"Total Revenue: £{customer_data['Monetary'].sum():,.2f}")
print(f"Average Revenue per Customer: £{customer_data['Monetary'].mean():,.2f}")
print(f"\nRecency Statistics:")
print(f"  Mean: {customer_data['Recency'].mean():.1f} days")
print(f"  Median: {customer_data['Recency'].median():.0f} days")
print(f"  25th percentile: {customer_data['Recency'].quantile(0.25):.0f} days")
print(f"  75th percentile: {customer_data['Recency'].quantile(0.75):.0f} days")
print(f"\nFrequency Statistics:")
print(f"  Mean: {customer_data['Frequency'].mean():.1f} transactions")
print(f"  Median: {customer_data['Frequency'].median():.0f} transactions")
print(f"  Top 10% of customers make {customer_data['Frequency'].quantile(0.9):.0f}+ transactions")
print(f"\nMonetary Statistics:")
print(f"  Mean: £{customer_data['Monetary'].mean():,.2f}")
print(f"  Median: £{customer_data['Monetary'].median():,.2f}")
print(f"  Top 10% spend £{customer_data['Monetary'].quantile(0.9):,.2f}+")

print("💡 KEY INSIGHTS FROM INITIAL ANALYSIS")
print("="*60)

# Insight 1: Pareto Principle (80/20 rule)
total_revenue = customer_data['Monetary'].sum()
top_20_percent = int(len(customer_data) * 0.2)
top_customers = customer_data.nlargest(top_20_percent, 'Monetary')
revenue_from_top_20 = top_customers['Monetary'].sum()
percentage_revenue = (revenue_from_top_20 / total_revenue) * 100

print(f"1. PARETO PRINCIPLE (80/20 Rule):")
print(f"   • Top 20% of customers ({top_20_percent:,}) generate {percentage_revenue:.1f}% of revenue")
print(f"   • Average spend of top 20%: £{top_customers['Monetary'].mean():,.2f}")
print(f"   • Average spend of bottom 80%: £{customer_data.nsmallest(len(customer_data)-top_20_percent, 'Monetary')['Monetary'].mean():,.2f}")

# Insight 2: Customer Engagement Levels
print(f"\n2. CUSTOMER ENGAGEMENT LEVELS:")
active_customers = customer_data[customer_data['Recency'] <= 30].shape[0]
dormant_customers = customer_data[(customer_data['Recency'] > 30) & (customer_data['Recency'] <= 90)].shape[0]
at_risk_customers = customer_data[customer_data['Recency'] > 90].shape[0]

print(f"   • Active customers (≤30 days): {active_customers:,} ({active_customers/len(customer_data)*100:.1f}%)")
print(f"   • Dormant customers (31-90 days): {dormant_customers:,} ({dormant_customers/len(customer_data)*100:.1f}%)")
print(f"   • At-risk customers (>90 days): {at_risk_customers:,} ({at_risk_customers/len(customer_data)*100:.1f}%)")

# Insight 3: Geographic concentration
print(f"\n3. GEOGRAPHIC CONCENTRATION:")
uk_customers = customer_data[customer_data['Country'] == 'United Kingdom'].shape[0]
uk_revenue = customer_data[customer_data['Country'] == 'United Kingdom']['Monetary'].sum()
print(f"   • UK customers: {uk_customers:,} ({uk_customers/len(customer_data)*100:.1f}% of total)")
print(f"   • UK revenue: £{uk_revenue:,.2f} ({uk_revenue/total_revenue*100:.1f}% of total)")

# Insight 4: Purchase frequency patterns
print(f"\n4. PURCHASE FREQUENCY PATTERNS:")
one_time_buyers = customer_data[customer_data['Frequency'] == 1].shape[0]
repeat_buyers = customer_data[customer_data['Frequency'] > 1].shape[0]
print(f"   • One-time buyers: {one_time_buyers:,} ({one_time_buyers/len(customer_data)*100:.1f}%)")
print(f"   • Repeat buyers: {repeat_buyers:,} ({repeat_buyers/len(customer_data)*100:.1f}%)")
print(f"   • Average transactions per repeat buyer: {customer_data[customer_data['Frequency'] > 1]['Frequency'].mean():.1f}")

# Insight 5: High-value vs Low-value segments
print(f"\n5. VALUE SEGMENT ANALYSIS:")
high_value_threshold = customer_data['Monetary'].quantile(0.75)
low_value_threshold = customer_data['Monetary'].quantile(0.25)

high_value = customer_data[customer_data['Monetary'] >= high_value_threshold]
low_value = customer_data[customer_data['Monetary'] <= low_value_threshold]

print(f"   • High-value customers (top 25%):")
print(f"     - Count: {len(high_value):,}")
print(f"     - Avg Recency: {high_value['Recency'].mean():.1f} days")
print(f"     - Avg Frequency: {high_value['Frequency'].mean():.1f} transactions")
print(f"   • Low-value customers (bottom 25%):")
print(f"     - Count: {len(low_value):,}")
print(f"     - Avg Recency: {low_value['Recency'].mean():.1f} days")
print(f"     - Avg Frequency: {low_value['Frequency'].mean():.1f} transactions")

print("\n" + "="*60)
print("🎯 APPROACH 1: TRADITIONAL RFM SCORE SEGMENTATION")
print("="*60)

def calculate_rfm_scores(customer_data):
    """Calculate RFM scores and segments"""
    rfm = customer_data.copy()
    
    # Create scoring (1-4, where 4 is best)
    # Recency: Lower is better (recent purchase = higher score)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    
    # Frequency: Higher is better
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    # Monetary: Higher is better
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    # Combine scores
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Sum'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
    
    # Create segments based on RFM scores
    segment_map = {
        r'111|112|121|131|141|151': 'Hibernating',
        r'[1-2][1-3][1-2]': 'At Risk',
        r'[1-2]4[1-2]': 'Cant Lose',
        r'[1-2][1-2][3-4]': 'About to Sleep',
        r'[1-2][3-4][3-4]': 'Need Attention',
        r'3[1-2][1-3]': 'Promising',
        r'33[1-3]': 'Potential Loyalists',
        r'[3-4][1-3]4': 'Loyal',
        r'4[1-2]4': 'Champions',
        r'[3-4][3-4][3-4]': 'Loyal Customers',
        r'[1-2]4[3-4]': 'High Spenders',
        r'[3-4]4[1-2]': 'New Customers'
    }
    
    rfm['Segment'] = 'Others'
    for pattern, segment in segment_map.items():
        mask = rfm['RFM_Score'].str.match(pattern)
        rfm.loc[mask, 'Segment'] = segment
    
    return rfm

# Calculate RFM scores
rfm_scored = calculate_rfm_scores(customer_data)

print("✅ RFM Scores Calculated!")
print(f"\nSegment Distribution:")
segment_counts = rfm_scored['Segment'].value_counts()
for segment, count in segment_counts.items():
    percentage = count / len(rfm_scored) * 100
    print(f"  • {segment}: {count:,} customers ({percentage:.1f}%)")

# Visualize RFM segments
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Segment Distribution
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
axes[0].set_title('Customer Segments Distribution', fontweight='bold')

# 2. Segment Performance
segment_stats = rfm_scored.groupby('Segment').agg({
    'Monetary': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean',
    'CustomerID': 'count'
}).round(2)

# Sort by monetary value
segment_stats = segment_stats.sort_values('Monetary', ascending=False)
x = range(len(segment_stats))
width = 0.35

axes[1].bar(x, segment_stats['Monetary'], width, label='Avg Spend (£)', color='skyblue')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Average Spend (£)', color='skyblue')
axes[1].tick_params(axis='y', labelcolor='skyblue')
axes[1].set_xticks(x)
axes[1].set_xticklabels(segment_stats.index, rotation=45, ha='right')

ax2 = axes[1].twinx()
ax2.bar([i + width for i in x], segment_stats['Frequency'], width, label='Avg Frequency', color='salmon')
ax2.set_ylabel('Average Frequency', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

axes[1].set_title('Segment Performance', fontweight='bold')
axes[1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Detailed segment analysis
print("\n📊 SEGMENT PERFORMANCE ANALYSIS:")
print("="*60)
for segment in segment_stats.index:
    seg_data = segment_stats.loc[segment]
    print(f"\n{segment.upper()}:")
    print(f"  • Customers: {seg_data['CustomerID']:,}")
    print(f"  • Avg Spend: £{seg_data['Monetary']:,.2f}")
    print(f"  • Avg Frequency: {seg_data['Frequency']:.1f} transactions")
    print(f"  • Avg Recency: {seg_data['Recency']:.1f} days")

print("\n" + "="*60)
print("🤖 APPROACH 2: K-MEANS CLUSTERING")
print("="*60)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Prepare data for clustering
cluster_data = customer_data[['Recency', 'Frequency', 'Monetary']].copy()

# Log transform to handle skewness
cluster_data['Frequency_log'] = np.log1p(cluster_data['Frequency'])
cluster_data['Monetary_log'] = np.log1p(cluster_data['Monetary'])
cluster_data['Recency_log'] = np.log1p(cluster_data['Recency'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data[['Recency_log', 'Frequency_log', 'Monetary_log']])

# Determine optimal number of clusters using Elbow Method
print("🔍 Finding optimal number of clusters...")
wcss = []  # Within-cluster sum of squares
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Plot Elbow Method and Silhouette Scores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Elbow Method
axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True)

# Silhouette Scores
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Scores for Different k')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (usually where elbow bends)
# Let's automatically determine optimal k
optimal_k = 4  # You can also use silhouette scores to choose automatically
# Or: optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"✅ Optimal number of clusters: {optimal_k}")

# Apply K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Analyze clusters
print(f"\n📊 CLUSTER ANALYSIS:")
print("="*60)

# Calculate cluster statistics
cluster_stats = customer_data.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'TotalQuantity': 'mean',
    'UniqueProducts': 'mean'
}).round(2)

# Calculate percentage of customers and revenue
cluster_stats['%_Customers'] = (cluster_stats['CustomerID'] / len(customer_data) * 100).round(1)
cluster_stats['Total_Revenue'] = customer_data.groupby('Cluster')['Monetary'].sum()
cluster_stats['%_Revenue'] = (cluster_stats['Total_Revenue'] / customer_data['Monetary'].sum() * 100).round(1)

# Sort clusters by monetary value
cluster_stats = cluster_stats.sort_values('Monetary', ascending=False)

print("\nCluster Characteristics:")
for cluster in cluster_stats.index:
    stats = cluster_stats.loc[cluster]
    print(f"\n🏷️ CLUSTER {cluster}:")
    print(f"  • Customers: {stats['CustomerID']:,} ({stats['%_Customers']}%)")
    print(f"  • Revenue Contribution: {stats['%_Revenue']}%")
    print(f"  • Avg Recency: {stats['Recency']:.1f} days")
    print(f"  • Avg Frequency: {stats['Frequency']:.1f} transactions")
    print(f"  • Avg Spend: £{stats['Monetary']:,.2f}")
    print(f"  • Avg Unique Products: {stats['UniqueProducts']:.1f}")

# Visualize clusters - FIXED VERSION
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Cluster Size
axes[0, 0].bar(cluster_stats.index, cluster_stats['CustomerID'], color=plt.cm.Set3(cluster_stats.index))
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].set_title('Cluster Size Distribution')
for i, v in enumerate(cluster_stats['CustomerID']):
    axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')

# 2. Revenue Contribution
axes[0, 1].pie(cluster_stats['%_Revenue'], labels=[f'Cluster {i}' for i in cluster_stats.index],
              autopct='%1.1f%%', colors=plt.cm.Set3(cluster_stats.index))
axes[0, 1].set_title('Revenue Contribution by Cluster')

# 3. FIXED Radar Chart - Alternative: Bar chart comparison
# Instead of radar chart, let's use grouped bar chart which is clearer
metrics = ['Recency', 'Frequency', 'Monetary']
x = np.arange(len(metrics))
width = 0.2

# Normalize for better visualization (0-1 scale)
normalized_stats = cluster_stats.copy()
for metric in metrics:
    min_val = cluster_stats[metric].min()
    max_val = cluster_stats[metric].max()
    normalized_stats[f'{metric}_norm'] = (cluster_stats[metric] - min_val) / (max_val - min_val)

for idx, cluster in enumerate(cluster_stats.index):
    values = [normalized_stats.loc[cluster, f'{metric}_norm'] for metric in metrics]
    axes[1, 0].bar(x + idx*width, values, width, label=f'Cluster {cluster}', 
                   color=plt.cm.Set3(idx))
    
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Normalized Value (0-1)')
axes[1, 0].set_title('Cluster RFM Profile Comparison')
axes[1, 0].set_xticks(x + width*(len(cluster_stats)-1)/2)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()

# 4. 3D Scatter plot - FIXED (using 2D scatter since 3D is complex)
# Instead of trying 3D, let's do 2 informative 2D plots
scatter1 = axes[1, 1].scatter(cluster_data['Recency_log'], 
                             cluster_data['Frequency_log'], 
                             c=customer_data['Cluster'], 
                             cmap='viridis', alpha=0.6, s=10)
axes[1, 1].set_xlabel('Recency (log)')
axes[1, 1].set_ylabel('Frequency (log)')
axes[1, 1].set_title('Customer Clusters: Recency vs Frequency')
plt.colorbar(scatter1, ax=axes[1, 1], label='Cluster')

plt.tight_layout()
plt.show()

# Alternative: Create a separate figure for the radar chart if you really want it
print("\n🌀 Creating Radar Chart (Alternative Visualization)...")

# FIXED Radar Chart Implementation
fig_radar = plt.figure(figsize=(10, 8))
ax_radar = fig_radar.add_subplot(111, projection='polar')

# Now we can use polar plot methods
categories = ['Recency', 'Frequency', 'Monetary']
N = len(categories)

# What each angle is at
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

for idx, cluster in enumerate(cluster_stats.index):
    values = normalized_stats.loc[cluster, ['Recency_norm', 'Frequency_norm', 'Monetary_norm']].values.tolist()
    values += values[:1]  # Complete the loop
    
    # Actually plot
    ax_radar.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
    ax_radar.fill(angles, values, alpha=0.1)

# Fix axis
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Cluster RFM Profile (Normalized Radar Chart)', size=15, y=1.1)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.show()

# Name the clusters based on characteristics
cluster_names = {
    0: 'High-Value Loyal Customers',
    1: 'Mid-Value Regulars',
    2: 'Low-Value Occasional Buyers',
    3: 'At-Risk Customers'
}

customer_data['Cluster_Name'] = customer_data['Cluster'].map(cluster_names)
print(f"\n🎯 CLUSTER NAMES BASED ON CHARACTERISTICS:")
for cluster, name in cluster_names.items():
    cluster_customers = customer_data[customer_data['Cluster'] == cluster]
    print(f"  • Cluster {cluster}: {name}")
    print(f"    - Count: {len(cluster_customers):,}")
    print(f"    - Avg Recency: {cluster_customers['Recency'].mean():.1f} days")
    print(f"    - Avg Spend: £{cluster_customers['Monetary'].mean():,.2f}")

print("\n" + "="*60)
print("🔄 COMPARISON: RFM vs K-MEANS APPROACHES")
print("="*60)

# Merge both segmentation results
comparison_df = customer_data[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Cluster_Name']].copy()
comparison_df = comparison_df.merge(
    rfm_scored[['CustomerID', 'Segment', 'RFM_Sum']], 
    on='CustomerID', 
    how='left'
)

print(f"✅ Combined {len(comparison_df):,} customers from both segmentation methods")

# Check what segments/clusters we have
print(f"\n📋 RFM Segments: {rfm_scored['Segment'].nunique()} unique segments")
print(f"📋 K-means Clusters: {comparison_df['Cluster_Name'].nunique()} unique clusters")

# Cross-tabulation
print("\n📊 CROSS-TABULATION OF SEGMENTS:")
cross_tab = pd.crosstab(comparison_df['Cluster_Name'], comparison_df['Segment'])
print(cross_tab)

# Visualize the relationship
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Heatmap of cross-tabulation
im = axes[0].imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(len(cross_tab.columns)))
axes[0].set_xticklabels(cross_tab.columns, rotation=45, ha='right')
axes[0].set_yticks(range(len(cross_tab.index)))
axes[0].set_yticklabels(cross_tab.index)
axes[0].set_title('Segment Overlap Heatmap')
plt.colorbar(im, ax=axes[0], label='Number of Customers')

# 2. FIXED: Comparison of segment sizes - Use separate charts
# Since they have different numbers of segments, plot them separately

# Clear the second axes for new plots
axes[1].clear()

# Create subplots within the second axes using inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# RFM segment sizes (top part)
rfm_segment_sizes = rfm_scored['Segment'].value_counts().sort_index()
rfm_ax = inset_axes(axes[1], width="40%", height="40%", loc='upper left')
bars_rfm = rfm_ax.bar(range(len(rfm_segment_sizes)), rfm_segment_sizes.values, 
                     color='skyblue', alpha=0.8)
rfm_ax.set_title('RFM Segments', fontsize=10)
rfm_ax.set_ylabel('Count', fontsize=8)
rfm_ax.set_xticks(range(len(rfm_segment_sizes)))
rfm_ax.set_xticklabels(rfm_segment_sizes.index, rotation=45, ha='right', fontsize=7)
rfm_ax.tick_params(axis='y', labelsize=7)

# Add count labels on bars
for bar in bars_rfm:
    height = bar.get_height()
    rfm_ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}', ha='center', va='bottom', fontsize=6)

# K-means cluster sizes (bottom part)
kmeans_segment_sizes = comparison_df['Cluster_Name'].value_counts().sort_index()
kmeans_ax = inset_axes(axes[1], width="40%", height="40%", loc='lower right')
bars_kmeans = kmeans_ax.bar(range(len(kmeans_segment_sizes)), kmeans_segment_sizes.values,
                           color='lightcoral', alpha=0.8)
kmeans_ax.set_title('K-means Clusters', fontsize=10)
kmeans_ax.set_ylabel('Count', fontsize=8)
kmeans_ax.set_xticks(range(len(kmeans_segment_sizes)))
kmeans_ax.set_xticklabels(kmeans_segment_sizes.index, rotation=45, ha='right', fontsize=7)
kmeans_ax.tick_params(axis='y', labelsize=7)

# Add count labels on bars
for bar in bars_kmeans:
    height = bar.get_height()
    kmeans_ax.text(bar.get_x() + bar.get_width()/2., height,
                  f'{int(height):,}', ha='center', va='bottom', fontsize=6)

axes[1].set_title('Segment Size Comparison (Separate Charts)', fontsize=12)
axes[1].axis('off')  # Hide the main axes since we're using inset axes

plt.tight_layout()
plt.show()

# Alternative: Create a separate figure for side-by-side comparison
print("\n📊 ALTERNATIVE VISUALIZATION: Side-by-side comparison")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# RFM segments pie chart
rfm_counts = rfm_scored['Segment'].value_counts()
colors_rfm = plt.cm.Set3(np.linspace(0, 1, len(rfm_counts)))
axes[0].pie(rfm_counts.values, labels=rfm_counts.index, autopct='%1.1f%%',
           colors=colors_rfm, startangle=90)
axes[0].set_title('RFM Segments Distribution', fontweight='bold')

# K-means clusters pie chart
kmeans_counts = comparison_df['Cluster_Name'].value_counts()
colors_kmeans = plt.cm.Set2(np.linspace(0, 1, len(kmeans_counts)))
axes[1].pie(kmeans_counts.values, labels=kmeans_counts.index, autopct='%1.1f%%',
           colors=colors_kmeans, startangle=90)
axes[1].set_title('K-means Clusters Distribution', fontweight='bold')

plt.tight_layout()
plt.show()

# Evaluate which approach gives better separation
print("\n📏 EVALUATION METRICS:")
print("-"*40)

# Calculate separation metrics for K-means
# Make sure scaled_data and customer_data['Cluster'] are aligned
if 'Cluster' in customer_data.columns and hasattr(scaled_data, 'shape'):
    if len(customer_data) == scaled_data.shape[0]:
        kmeans_silhouette = silhouette_score(scaled_data, customer_data['Cluster'])
        print(f"K-means Silhouette Score: {kmeans_silhouette:.3f}")
        print("(Closer to 1 = better separation)")
    else:
        print("⚠️  Cannot calculate silhouette score: data mismatch")
else:
    print("⚠️  Cannot calculate silhouette score: missing cluster labels or scaled data")

# For RFM, we can calculate intra/inter segment variance
def calculate_rfm_separation(rfm_data):
    """Calculate separation quality for RFM segments"""
    # Check if we have the necessary columns
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Segment']
    missing_cols = [col for col in required_cols if col not in rfm_data.columns]
    
    if missing_cols:
        print(f"⚠️  Missing columns for RFM separation: {missing_cols}")
        return 0
    
    segments = rfm_data['Segment'].unique()
    total_ss = 0
    
    for segment in segments:
        segment_data = rfm_data[rfm_data['Segment'] == segment]
        segment_mean = segment_data[['Recency', 'Frequency', 'Monetary']].mean()
        total_ss += ((segment_data[['Recency', 'Frequency', 'Monetary']] - segment_mean) ** 2).sum().sum()
    
    overall_mean = rfm_data[['Recency', 'Frequency', 'Monetary']].mean()
    between_ss = ((rfm_data[['Recency', 'Frequency', 'Monetary']] - overall_mean) ** 2).sum().sum()
    
    separation_ratio = (between_ss - total_ss) / between_ss if between_ss > 0 else 0
    return separation_ratio

# Check if rfm_scored has the necessary columns
if all(col in rfm_scored.columns for col in ['Recency', 'Frequency', 'Monetary', 'Segment']):
    rfm_separation = calculate_rfm_separation(rfm_scored)
    print(f"RFM Segment Separation Ratio: {rfm_separation:.3f}")
    print("(Higher = better separation between segments)")
else:
    print("⚠️  Cannot calculate RFM separation: missing required columns")

# Calculate agreement between methods
print("\n🤝 AGREEMENT ANALYSIS:")
print("-"*40)

# Check if a customer is consistently classified as high/medium/low
def calculate_segment_agreement(comparison_df):
    """Calculate how often both methods agree on customer value tier"""
    
    # Define value tiers for each method
    # For K-means: Use Cluster_Name
    # For RFM: Map Segment names to tiers
    tier_mapping = {
        # High value
        'Champions': 'High',
        'Loyal Customers': 'High',
        'High Spenders': 'High',
        'High-Value Loyal Customers': 'High',
        
        # Medium value
        'Potential Loyalists': 'Medium',
        'Need Attention': 'Medium',
        'Promising': 'Medium',
        'Mid-Value Regulars': 'Medium',
        
        # Low value
        'At Risk': 'Low',
        'Cant Lose': 'Low',
        'About to Sleep': 'Low',
        'Hibernating': 'Low',
        'Low-Value Occasional Buyers': 'Low',
        'At-Risk Customers': 'Low'
    }
    
    # Map segments to tiers
    comparison_df['RFM_Tier'] = comparison_df['Segment'].map(tier_mapping)
    comparison_df['Kmeans_Tier'] = comparison_df['Cluster_Name'].map(tier_mapping)
    
    # Calculate agreement
    agreement = (comparison_df['RFM_Tier'] == comparison_df['Kmeans_Tier']).mean() * 100
    
    # Create confusion matrix of tiers
    if 'RFM_Tier' in comparison_df.columns and 'Kmeans_Tier' in comparison_df.columns:
        tier_crosstab = pd.crosstab(comparison_df['RFM_Tier'], comparison_df['Kmeans_Tier'], 
                                   normalize='all') * 100
        
        print(f"Tier Agreement: {agreement:.1f}%")
        print("\nTier Confusion Matrix (% of total customers):")
        print(tier_crosstab.round(1))
    
    return agreement

# Try to calculate agreement
try:
    agreement_rate = calculate_segment_agreement(comparison_df)
except Exception as e:
    print(f"⚠️  Could not calculate agreement: {e}")

print("\n💡 RECOMMENDATION:")
print("-"*40)

# Count segments and clusters
rfm_segment_count = rfm_scored['Segment'].nunique()
kmeans_cluster_count = comparison_df['Cluster_Name'].nunique()

print(f"""
• Use RFM Segmentation ({rfm_segment_count} segments) for:
  - Marketing campaigns (easy to understand, business-friendly)
  - Business reporting and dashboards
  - Simple customer tiering (High/Medium/Low)
  - Quick segmentation without technical expertise

• Use K-means Clustering ({kmeans_cluster_count} clusters) for:
  - Advanced analytics and data exploration
  - Predictive modeling and machine learning
  - Discovering hidden patterns in customer behavior
  - Data-driven strategy and optimization
  - When you want purely data-driven segments

• Recommendation: Use BOTH!
  - Start with RFM for immediate business impact
  - Use K-means to validate and refine segments
  - Combine insights from both approaches
""")

print("🎯 CUSTOMER LIFETIME VALUE PREDICTION")
print("="*60)

# Import additional libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
import warnings
warnings.filterwarnings('ignore')

print("📊 PREPARING DATA FOR CLV MODELS (Alternative Method)")
print("="*60)

# Use the cleaned data
clv_data = df_clean.copy()

# Basic customer metrics (same as before)
customer_features = clv_data.groupby('CustomerID').agg({
    'InvoiceDate': ['min', 'max', 'count'],
    'TotalSpend': 'sum',
    'Quantity': 'sum',
    'StockCode': 'nunique',
    'InvoiceNo': 'nunique',
    'Country': 'first'
}).reset_index()

customer_features.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase', 
                            'PurchaseCount', 'TotalMonetary', 'TotalQuantity',
                            'UniqueProducts', 'TransactionCount', 'Country']

# Calculate additional features
customer_features['CustomerLifespan'] = (customer_features['LastPurchase'] - 
                                        customer_features['FirstPurchase']).dt.days + 1
customer_features['AvgTransactionValue'] = customer_features['TotalMonetary'] / customer_features['TransactionCount']
customer_features['AvgItemsPerTransaction'] = customer_features['TotalQuantity'] / customer_features['TransactionCount']
customer_features['PurchaseFrequency'] = customer_features['TransactionCount'] / customer_features['CustomerLifespan']

# Handle division by zero
customer_features['AvgTransactionValue'] = customer_features['AvgTransactionValue'].replace([np.inf, -np.inf], 0)
customer_features['AvgItemsPerTransaction'] = customer_features['AvgItemsPerTransaction'].replace([np.inf, -np.inf], 0)
customer_features['PurchaseFrequency'] = customer_features['PurchaseFrequency'].replace([np.inf, -np.inf], 0)

# Calculate recency
snapshot_date = clv_data['InvoiceDate'].max() + pd.Timedelta(days=1)
customer_features['Recency'] = (snapshot_date - customer_features['LastPurchase']).dt.days

# Day of week patterns using pivot_table with fill_value
print("\nAdding day of week patterns...")
day_pattern = pd.pivot_table(
    clv_data,
    values='InvoiceNo',
    index='CustomerID',
    columns='Weekday',
    aggfunc='count',
    fill_value=0
)

# Ensure all 7 weekdays are present
for i in range(7):
    if i not in day_pattern.columns:
        day_pattern[i] = 0

# Rename columns
day_pattern = day_pattern.reindex(sorted(day_pattern.columns), axis=1)
day_pattern.columns = [f'Day_{i}_Purchases' for i in range(7)]

# Hour of day patterns
print("Adding hour of day patterns...")
hour_pattern = pd.pivot_table(
    clv_data,
    values='InvoiceNo',
    index='CustomerID',
    columns='Hour',
    aggfunc='count',
    fill_value=0
)

# Ensure all 24 hours are present
for i in range(24):
    if i not in hour_pattern.columns:
        hour_pattern[i] = 0

# Rename columns
hour_pattern = hour_pattern.reindex(sorted(hour_pattern.columns), axis=1)
hour_pattern.columns = [f'Hour_{i}_Purchases' for i in range(24)]

# Merge patterns with customer features
customer_features = customer_features.merge(day_pattern, on='CustomerID', how='left')
customer_features = customer_features.merge(hour_pattern, on='CustomerID', how='left')

# Fill any remaining NaN values
customer_features = customer_features.fillna(0)

print(f"✅ Customer features created for {len(customer_features):,} customers")

print("\n" + "="*60)
print("📈 APPROACH 1: HISTORICAL CLV")
print("="*60)

def calculate_historical_clv(customer_data, prediction_period_days=365):
    """
    Calculate historical CLV based on past behavior
    Simple but effective for established customers
    """
    historical_clv = customer_data.copy()
    
    # Method 1: Average daily spend * prediction period
    historical_clv['DailySpend'] = historical_clv['TotalMonetary'] / historical_clv['CustomerLifespan']
    historical_clv['CLV_DailyAvg'] = historical_clv['DailySpend'] * prediction_period_days
    
    # Method 2: Extrapolate based on frequency and average transaction value
    historical_clv['CLV_FrequencyBased'] = (
        historical_clv['TransactionCount'] / 
        historical_clv['CustomerLifespan'] * 
        prediction_period_days * 
        historical_clv['AvgTransactionValue']
    )
    
    # Method 3: Simple moving average (last 3 months behavior)
    # We'll use last 90 days activity for this
    last_90_days = snapshot_date - pd.Timedelta(days=90)
    recent_data = clv_data[clv_data['InvoiceDate'] >= last_90_days]
    
    recent_spend = recent_data.groupby('CustomerID')['TotalSpend'].sum()
    recent_transactions = recent_data.groupby('CustomerID')['InvoiceNo'].nunique()
    
    historical_clv = historical_clv.merge(
        recent_spend.rename('Recent90DaySpend'), 
        on='CustomerID', 
        how='left'
    ).merge(
        recent_transactions.rename('Recent90DayTransactions'), 
        on='CustomerID', 
        how='left'
    )
    
    historical_clv['Recent90DaySpend'] = historical_clv['Recent90DaySpend'].fillna(0)
    historical_clv['Recent90DayTransactions'] = historical_clv['Recent90DayTransactions'].fillna(0)
    
    # Project last 90 days to yearly CLV
    historical_clv['CLV_RecentBased'] = historical_clv['Recent90DaySpend'] * 4  # Quarterly to yearly
    
    # Weighted average of all methods
    historical_clv['CLV_Historical'] = (
        historical_clv['CLV_DailyAvg'] * 0.3 +
        historical_clv['CLV_FrequencyBased'] * 0.4 +
        historical_clv['CLV_RecentBased'] * 0.3
    )
    
    # Categorize customers by CLV
    clv_percentiles = historical_clv['CLV_Historical'].quantile([0.33, 0.66])
    historical_clv['CLV_Tier'] = pd.cut(
        historical_clv['CLV_Historical'],
        bins=[-np.inf, clv_percentiles[0.33], clv_percentiles[0.66], np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    return historical_clv

# Calculate historical CLV
historical_clv = calculate_historical_clv(customer_features)

print("✅ Historical CLV Calculated!")
print(f"\n📊 CLV Distribution:")
print(historical_clv['CLV_Historical'].describe())

print(f"\n🎯 CLV Tiers:")
tier_counts = historical_clv['CLV_Tier'].value_counts()
for tier, count in tier_counts.items():
    percentage = count / len(historical_clv) * 100
    avg_clv = historical_clv[historical_clv['CLV_Tier'] == tier]['CLV_Historical'].mean()
    print(f"  • {tier} Tier: {count:,} customers ({percentage:.1f}%) - Avg CLV: £{avg_clv:,.2f}")

# Visualize historical CLV
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. CLV Distribution
axes[0].hist(historical_clv['CLV_Historical'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Predicted CLV (£)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Historical CLV Distribution')
axes[0].axvline(historical_clv['CLV_Historical'].median(), color='red', linestyle='--',
               label=f'Median: £{historical_clv["CLV_Historical"].median():,.2f}')
axes[0].legend()

# 2. CLV by Tier
colors = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}
for tier in historical_clv['CLV_Tier'].unique():
    tier_data = historical_clv[historical_clv['CLV_Tier'] == tier]['CLV_Historical']
    axes[1].hist(tier_data, bins=30, alpha=0.6, label=tier, color=colors.get(tier, 'blue'))
axes[1].set_xlabel('Predicted CLV (£)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('CLV Distribution by Tier')
axes[1].legend()

# 3. Top 20 customers
top_customers = historical_clv.nlargest(20, 'CLV_Historical')[['CustomerID', 'CLV_Historical', 'TransactionCount']]
axes[2].bar(range(len(top_customers)), top_customers['CLV_Historical'], color='skyblue')
axes[2].set_xlabel('Customer Rank')
axes[2].set_ylabel('Predicted CLV (£)')
axes[2].set_title('Top 20 Customers by CLV')
axes[2].set_xticks(range(len(top_customers)))
axes[2].set_xticklabels(top_customers['CustomerID'], rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("\n💡 INSIGHTS FROM HISTORICAL CLV:")
print("-"*40)
print(f"1. Total predicted CLV (next year): £{historical_clv['CLV_Historical'].sum():,.2f}")
print(f"2. Average customer value: £{historical_clv['CLV_Historical'].mean():,.2f}")
print(f"3. Top 10% of customers contribute: £{historical_clv.nlargest(int(len(historical_clv)*0.1), 'CLV_Historical')['CLV_Historical'].sum():,.2f}")
print(f"4. Median CLV: £{historical_clv['CLV_Historical'].median():,.2f}")

print("\n" + "="*60)
print("🤖 APPROACH 2: PREDICTIVE CLV (MACHINE LEARNING)")
print("="*60)

def prepare_ml_clv_data(customer_features, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning CLV prediction
    """
    # Create target variable: Future 90-day spend (for validation)
    # We'll use last 30 days as holdout period
    validation_start = snapshot_date - pd.Timedelta(days=90)
    validation_end = snapshot_date - pd.Timedelta(days=1)
    
    validation_data = clv_data[
        (clv_data['InvoiceDate'] >= validation_start) & 
        (clv_data['InvoiceDate'] <= validation_end)
    ]
    
    # Calculate actual spend in validation period
    actual_future_spend = validation_data.groupby('CustomerID')['TotalSpend'].sum().reset_index()
    actual_future_spend.columns = ['CustomerID', 'Actual_Future_Spend']
    
    # Merge with customer features
    ml_data = customer_features.merge(actual_future_spend, on='CustomerID', how='left')
    ml_data['Actual_Future_Spend'] = ml_data['Actual_Future_Spend'].fillna(0)
    
    # Create features for ML
    features = [
        'TransactionCount', 'TotalMonetary', 'TotalQuantity', 'UniqueProducts',
        'CustomerLifespan', 'AvgTransactionValue', 'AvgItemsPerTransaction',
        'PurchaseFrequency', 'Recency'
    ]
    
    # Add day and hour features (top 5 most important)
    day_features = [f'Day_{i}_Purchases' for i in range(7)]
    hour_features = [f'Hour_{i}_Purchases' for i in [9, 10, 11, 12, 13, 14, 15]]  # Business hours
    
    all_features = features + day_features + hour_features
    
    # Prepare X and y
    X = ml_data[all_features]
    y = ml_data['Actual_Future_Spend']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, ml_data, all_features

# Prepare ML data
X_train, X_test, y_train, y_test, ml_data, feature_names = prepare_ml_clv_data(customer_features)

print(f"✅ ML Data Prepared!")
print(f"   Training samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Target: Future 90-day spend")

print("\n🔍 Target Variable Statistics:")
print(f"   Mean future spend: £{y_train.mean():.2f}")
print(f"   Max future spend: £{y_train.max():.2f}")
print(f"   % Zero future spend: {(y_train == 0).sum() / len(y_train) * 100:.1f}%")

def train_clv_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple ML models for CLV prediction"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': y_pred
        }
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
        
        print(f"  MAE: £{mae:.2f}, RMSE: £{rmse:.2f}, R²: {r2:.3f}")
    
    return results, feature_importances

# Train models
print("\n🚀 Training ML Models...")
results, feature_importances = train_clv_models(X_train, X_test, y_train, y_test)

# Compare model performance
print("\n📊 MODEL PERFORMANCE COMPARISON:")
print("="*50)

performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'R2': [results[m]['R2'] for m in results.keys()]
}).sort_values('R2', ascending=False)

print(performance_df)

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# MAE and RMSE comparison
x = np.arange(len(performance_df))
width = 0.35

axes[0].bar(x - width/2, performance_df['MAE'], width, label='MAE', color='skyblue', alpha=0.7)
axes[0].bar(x + width/2, performance_df['RMSE'], width, label='RMSE', color='salmon', alpha=0.7)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Error (£)')
axes[0].set_title('Model Error Comparison (Lower is Better)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(performance_df['Model'], rotation=45, ha='right')
axes[0].legend()

# R² comparison
colors = plt.cm.Set3(np.linspace(0, 1, len(performance_df)))
bars = axes[1].bar(performance_df['Model'], performance_df['R2'], color=colors)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Model R² Comparison (Higher is Better)')
axes[1].set_xticklabels(performance_df['Model'], rotation=45, ha='right')
axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Select best model
best_model_name = performance_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {performance_df.iloc[0]['R2']:.3f}")
print(f"   MAE: £{performance_df.iloc[0]['MAE']:.2f}")

print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

if feature_importances:
    # Get feature importance from best tree-based model
    if best_model_name in feature_importances:
        importance_scores = feature_importances[best_model_name]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False).head(15)
        
        print(f"\nTop 15 Important Features for {best_model_name}:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
        
        # Visualize feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['Importance'], color='teal')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top 15 Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.show()
        
        # Insight: What drives high CLV?
        print("\n💡 INSIGHTS FROM FEATURE IMPORTANCE:")
        print("-"*40)
        top_features = importance_df.head(5)['Feature'].tolist()
        print(f"High CLV customers are characterized by:")
        for feature in top_features:
            if 'TotalMonetary' in feature:
                print(f"  • High total historical spend")
            elif 'TransactionCount' in feature:
                print(f"  • High purchase frequency")
            elif 'AvgTransactionValue' in feature:
                print(f"  • High average transaction value")
            elif 'Recency' in feature:
                print(f"  • Recent purchases (low recency)")
            elif 'Day_' in feature:
                day_num = feature.split('_')[1]
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                print(f"  • Frequent purchases on {days[int(day_num)]}")
            elif 'Hour_' in feature:
                hour = feature.split('_')[1]
                print(f"  • Frequent purchases at {hour}:00")

def predict_clv_ml(customer_features, model, feature_names, prediction_period_days=365):
    """Predict CLV for all customers using ML model"""
    
    # Prepare features for prediction
    X_all = customer_features[feature_names]
    
    # Predict 90-day spend
    predicted_90day_spend = model.predict(X_all)
    
    # Convert to yearly CLV (extrapolate)
    # Simple extrapolation: multiply by 4 (quarterly to yearly)
    predicted_clv = predicted_90day_spend * (prediction_period_days / 90)
    
    # Create results dataframe
    clv_predictions = customer_features[['CustomerID']].copy()
    clv_predictions['Predicted_90Day_Spend'] = predicted_90day_spend
    clv_predictions['Predicted_CLV_1Year'] = predicted_clv
    
    # Categorize predictions
    clv_percentiles = clv_predictions['Predicted_CLV_1Year'].quantile([0.33, 0.66])
    clv_predictions['CLV_Tier_ML'] = pd.cut(
        clv_predictions['Predicted_CLV_1Year'],
        bins=[-np.inf, clv_percentiles[0.33], clv_percentiles[0.66], np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    return clv_predictions

# Make predictions
print("\n🎯 MAKING CLV PREDICTIONS FOR ALL CUSTOMERS...")
ml_clv_predictions = predict_clv_ml(customer_features, best_model, feature_names)

print(f"✅ CLV Predictions Complete!")
print(f"\n📊 ML-Based CLV Distribution:")
print(ml_clv_predictions['Predicted_CLV_1Year'].describe())

print(f"\n🎯 CLV Tiers (ML Prediction):")
tier_counts_ml = ml_clv_predictions['CLV_Tier_ML'].value_counts()
for tier, count in tier_counts_ml.items():
    percentage = count / len(ml_clv_predictions) * 100
    avg_clv = ml_clv_predictions[ml_clv_predictions['CLV_Tier_ML'] == tier]['Predicted_CLV_1Year'].mean()
    print(f"  • {tier} Tier: {count:,} customers ({percentage:.1f}%) - Avg CLV: £{avg_clv:,.2f}")

# Visualize ML-based CLV predictions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Distribution of predicted CLV
axes[0].hist(ml_clv_predictions['Predicted_CLV_1Year'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Predicted CLV (£)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('ML-Based CLV Distribution')
axes[0].axvline(ml_clv_predictions['Predicted_CLV_1Year'].median(), color='red', linestyle='--',
               label=f'Median: £{ml_clv_predictions["Predicted_CLV_1Year"].median():,.2f}')
axes[0].legend()

# 2. Actual vs Predicted for test set
test_indices = X_test.index
test_predictions = ml_clv_predictions.loc[test_indices]
test_actual = ml_data.loc[test_indices, 'Actual_Future_Spend']

# Convert 90-day actual to yearly for comparison
test_actual_yearly = test_actual * 4

axes[1].scatter(test_actual_yearly, test_predictions['Predicted_CLV_1Year'], alpha=0.5, s=20)
axes[1].plot([0, max(test_actual_yearly.max(), test_predictions['Predicted_CLV_1Year'].max())],
            [0, max(test_actual_yearly.max(), test_predictions['Predicted_CLV_1Year'].max())],
            color='red', linestyle='--')
axes[1].set_xlabel('Actual CLV (Extrapolated from 90-day) (£)')
axes[1].set_ylabel('Predicted CLV (£)')
axes[1].set_title('Actual vs Predicted CLV (Test Set)')
axes[1].grid(True, alpha=0.3)

# 3. Top 20 high-CLV customers
top_ml_customers = ml_clv_predictions.nlargest(20, 'Predicted_CLV_1Year')
axes[2].bar(range(len(top_ml_customers)), top_ml_customers['Predicted_CLV_1Year'], color='green', alpha=0.7)
axes[2].set_xlabel('Customer Rank')
axes[2].set_ylabel('Predicted CLV (£)')
axes[2].set_title('Top 20 Customers by ML-Predicted CLV')
axes[2].set_xticks(range(len(top_ml_customers)))
axes[2].set_xticklabels(top_ml_customers['CustomerID'], rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("\n💡 INSIGHTS FROM ML-BASED CLV:")
print("-"*40)
print(f"1. Total predicted CLV (next year): £{ml_clv_predictions['Predicted_CLV_1Year'].sum():,.2f}")
print(f"2. Average predicted CLV: £{ml_clv_predictions['Predicted_CLV_1Year'].mean():,.2f}")
print(f"3. Top 10% predicted value: £{ml_clv_predictions.nlargest(int(len(ml_clv_predictions)*0.1), 'Predicted_CLV_1Year')['Predicted_CLV_1Year'].sum():,.2f}")
print(f"4. Model confidence (R²): {performance_df.iloc[0]['R2']:.3f}")


print("\n" + "="*60)
print("📊 APPROACH 3: PROBABILISTIC CLV (BG/NBD & GAMMA-GAMMA)")
print("="*60)

def prepare_lifetimes_data(transaction_data):
    """Prepare data for lifetimes library"""
    
    # Create RFM summary for lifetimes
    rfm_summary = summary_data_from_transaction_data(
        transaction_data,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalSpend',
        observation_period_end=snapshot_date
    )
    
    return rfm_summary

# Prepare data for lifetimes
print("Preparing data for probabilistic models...")
lifetimes_data = prepare_lifetimes_data(clv_data)

print(f"\n📋 Lifetimes Data Summary:")
print(f"• Total customers: {len(lifetimes_data):,}")
print(f"• Frequency stats: {lifetimes_data['frequency'].describe()[['mean', '50%', 'max']].round(2).to_dict()}")
print(f"• Recency stats: {lifetimes_data['recency'].describe()[['mean', '50%', 'max']].round(2).to_dict()}")
print(f"• Monetary stats: {lifetimes_data['monetary_value'].describe()[['mean', '50%', 'max']].round(2).to_dict()}")

# Split into calibration and holdout
print("\n📅 Creating calibration and holdout periods...")
calibration_end = snapshot_date - pd.Timedelta(days=90)
calibration_data = calibration_and_holdout_data(
    clv_data,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    monetary_value_col='TotalSpend',
    calibration_period_end=calibration_end,
    observation_period_end=snapshot_date,
    freq='D'  # Daily frequency
)

print(f"Calibration period: up to {calibration_end}")
print(f"Holdout period: {calibration_end} to {snapshot_date}")

print("\n🔮 TRAINING BG/NBD MODEL (WITH MANUAL VISUALIZATIONS)")
print("="*60)

# Filter out customers with only 1 purchase (frequency = 0)
bgf_data = lifetimes_data[lifetimes_data['frequency'] > 0].copy()

print(f"📊 Customers with frequency > 0: {len(bgf_data):,} out of {len(lifetimes_data):,}")

# Train BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(bgf_data['frequency'], bgf_data['recency'], bgf_data['T'])

print("✅ BG/NBD Model Trained!")
print(f"\n📊 Model Parameters:")
for param, value in bgf.params_.items():
    print(f"  {param}: {value:.4f}")

# Create manual visualizations (no lifetimes plotting functions)
print("\n📊 Creating Manual Visualizations...")

# 1. Create Frequency-Recency heatmap manually
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap 1: Predicted purchases
# Create grid
freq_grid = np.arange(0, 51, 5)  # Frequency from 0 to 50
rec_grid = np.arange(0, 366, 30)  # Recency from 0 to 365

# Create meshgrid
F, R = np.meshgrid(freq_grid, rec_grid)
Z_pred = np.zeros_like(F, dtype=float)

# Fill with predictions (assuming T = recency + average customer lifespan)
avg_T = bgf_data['T'].mean()
for i in range(len(freq_grid)):
    for j in range(len(rec_grid)):
        if rec_grid[j] < avg_T:  # Recency must be less than T
            Z_pred[j, i] = bgf.predict(t=90, frequency=freq_grid[i], recency=rec_grid[j], T=avg_T)
        else:
            Z_pred[j, i] = 0

# Plot predicted purchases heatmap
im1 = axes[0].imshow(Z_pred, cmap='YlOrRd', aspect='auto', origin='lower', 
                     extent=[0, 50, 0, 365])
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Recency (days)')
axes[0].set_title('Predicted Purchases (Next 90 Days)', fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='Predicted Purchases')

# Heatmap 2: Probability alive
Z_prob = np.zeros_like(F, dtype=float)
for i in range(len(freq_grid)):
    for j in range(len(rec_grid)):
        if rec_grid[j] < avg_T:
            Z_prob[j, i] = bgf.conditional_probability_alive(frequency=freq_grid[i], 
                                                            recency=rec_grid[j], 
                                                            T=avg_T)
        else:
            Z_prob[j, i] = 0

# Plot probability alive heatmap
im2 = axes[1].imshow(Z_prob, cmap='RdYlGn', aspect='auto', origin='lower', 
                     extent=[0, 50, 0, 365], vmin=0, vmax=1)
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Recency (days)')
axes[1].set_title('Probability Customer is Alive', fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='Probability')

plt.tight_layout()
plt.show()

# Predict future transactions
print("\n🎯 PREDICTING FUTURE TRANSACTIONS...")

# Predict transactions for next 90 days
t = 90  # Predict for next 90 days
bgf_data['predicted_purchases_90d'] = bgf.predict(t, bgf_data['frequency'], bgf_data['recency'], bgf_data['T'])

print(f"\n📊 Purchase Predictions (Next 90 days):")
print(bgf_data['predicted_purchases_90d'].describe())
print(f"\nExpected total transactions in next 90 days: {bgf_data['predicted_purchases_90d'].sum():.0f}")

# Calculate additional metrics
bgf_data['probability_alive'] = bgf.conditional_probability_alive(
    bgf_data['frequency'], bgf_data['recency'], bgf_data['T']
)

# Create comprehensive analysis visualization
print("\n📈 COMPREHENSIVE ANALYSIS:")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distribution of predicted purchases
axes[0, 0].hist(bgf_data['predicted_purchases_90d'], bins=50, edgecolor='black', 
               alpha=0.7, color='skyblue')
axes[0, 0].set_xlabel('Predicted Purchases (90 days)')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].set_title('Distribution of Predicted Purchases', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Probability alive distribution
axes[0, 1].hist(bgf_data['probability_alive'], bins=50, edgecolor='black', 
               alpha=0.7, color='lightcoral')
axes[0, 1].set_xlabel('Probability Alive')
axes[0, 1].set_ylabel('Number of Customers')
axes[0, 1].set_title('Distribution of Alive Probability', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Predicted vs Probability alive
scatter = axes[0, 2].scatter(bgf_data['probability_alive'], bgf_data['predicted_purchases_90d'],
                            c=bgf_data['frequency'], alpha=0.6, s=20, cmap='viridis')
axes[0, 2].set_xlabel('Probability Alive')
axes[0, 2].set_ylabel('Predicted Purchases (90 days)')
axes[0, 2].set_title('Probability Alive vs Predicted Purchases', fontweight='bold')
plt.colorbar(scatter, ax=axes[0, 2], label='Historical Frequency')
axes[0, 2].grid(True, alpha=0.3)

# 4. RFM vs Predicted purchases
scatter1 = axes[1, 0].scatter(bgf_data['frequency'], bgf_data['predicted_purchases_90d'],
                             c=bgf_data['recency'], alpha=0.6, s=20, cmap='plasma')
axes[1, 0].set_xlabel('Historical Frequency')
axes[1, 0].set_ylabel('Predicted Purchases (90 days)')
axes[1, 0].set_title('Frequency vs Predicted Purchases', fontweight='bold')
plt.colorbar(scatter1, ax=axes[1, 0], label='Recency (days)')
axes[1, 0].grid(True, alpha=0.3)

# 5. Monetary value vs Predicted
scatter2 = axes[1, 1].scatter(bgf_data['monetary_value'], bgf_data['predicted_purchases_90d'],
                             c=bgf_data['probability_alive'], alpha=0.6, s=20, cmap='coolwarm')
axes[1, 1].set_xlabel('Monetary Value')
axes[1, 1].set_ylabel('Predicted Purchases (90 days)')
axes[1, 1].set_title('Monetary Value vs Predicted Purchases', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1, 1], label='Probability Alive')
axes[1, 1].grid(True, alpha=0.3)

# 6. Customer segments by predictions
# Create 4 quadrants based on probability alive and predicted purchases
bgf_data['segment'] = 'Medium'
bgf_data.loc[(bgf_data['probability_alive'] > 0.7) & 
            (bgf_data['predicted_purchases_90d'] > bgf_data['predicted_purchases_90d'].median()), 'segment'] = 'High Potential'
bgf_data.loc[(bgf_data['probability_alive'] < 0.3) & 
            (bgf_data['predicted_purchases_90d'] < bgf_data['predicted_purchases_90d'].median()), 'segment'] = 'At Risk'
bgf_data.loc[(bgf_data['probability_alive'] > 0.7) & 
            (bgf_data['predicted_purchases_90d'] < bgf_data['predicted_purchases_90d'].median()), 'segment'] = 'Loyal Low Spender'
bgf_data.loc[(bgf_data['probability_alive'] < 0.3) & 
            (bgf_data['predicted_purchases_90d'] > bgf_data['predicted_purchases_90d'].median()), 'segment'] = 'At Risk High Spender'

segment_counts = bgf_data['segment'].value_counts()
colors_segments = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
bars = axes[1, 2].bar(segment_counts.index, segment_counts.values, color=colors_segments)
axes[1, 2].set_xlabel('Customer Segment')
axes[1, 2].set_ylabel('Number of Customers')
axes[1, 2].set_title('Customer Segments by BG/NBD Model', fontweight='bold')
axes[1, 2].tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    percentage = (height / len(bgf_data)) * 100
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Print segment insights
print("\n💡 BG/NBD CUSTOMER SEGMENTS:")
print("="*50)
for segment in segment_counts.index:
    segment_data = bgf_data[bgf_data['segment'] == segment]
    print(f"\n🔹 {segment.upper()}:")
    print(f"   • Customers: {len(segment_data):,} ({len(segment_data)/len(bgf_data)*100:.1f}%)")
    print(f"   • Avg predicted purchases: {segment_data['predicted_purchases_90d'].mean():.2f}")
    print(f"   • Avg probability alive: {segment_data['probability_alive'].mean():.2f}")
    print(f"   • Avg historical spend: £{segment_data['monetary_value'].mean():.2f}")
    
    # Recommendations
    if 'High Potential' in segment:
        print(f"   📌 ACTION: Invest in retention, offer loyalty rewards")
    elif 'At Risk' in segment:
        print(f"   📌 ACTION: Win-back campaigns, special offers")
    elif 'Loyal Low Spender' in segment:
        print(f"   📌 ACTION: Upsell opportunities, product recommendations")
    elif 'At Risk High Spender' in segment:
        print(f"   📌 ACTION: Priority retention, personalized outreach")

print("\n💰 TRAINING GAMMA-GAMMA MODEL...")

# Filter for customers with frequency > 0
ggf_data = lifetimes_data[lifetimes_data['frequency'] > 0].copy()

# Train Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(ggf_data['frequency'], ggf_data['monetary_value'])

print("✅ Gamma-Gamma Model Trained!")
print(f"\n📊 Model Parameters:")
for param, value in ggf.params_.items():
    print(f"  {param}: {value:.4f}")

# Predict expected average transaction value
ggf_data['predicted_avg_value'] = ggf.conditional_expected_average_profit(
    ggf_data['frequency'],
    ggf_data['monetary_value']
)

print(f"\n📊 Predicted Average Transaction Value:")
print(ggf_data['predicted_avg_value'].describe())

print("\n🧮 CALCULATING PROBABILISTIC CLV...")

# Combine predictions for CLV calculation
# CLV = Expected purchases * Expected average transaction value

# Predict for different time periods
time_periods = [30, 90, 365]  # 30 days, 90 days, 1 year

probabilistic_clv = ggf_data.copy()

for days in time_periods:
    # Predict number of purchases
    probabilistic_clv[f'predicted_purchases_{days}d'] = bgf.predict(
        days,
        probabilistic_clv['frequency'],
        probabilistic_clv['recency'],
        probabilistic_clv['T']
    )
    
    # Calculate CLV
    probabilistic_clv[f'CLV_{days}d'] = (
        probabilistic_clv[f'predicted_purchases_{days}d'] *
        probabilistic_clv['predicted_avg_value']
    )

# Also calculate customer lifetime value (discounted)
# Using standard discount rate of 10% annually
discount_rate = 0.10
monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1

# Calculate CLV for 3 years with discounting
probabilistic_clv['CLV_3y_discounted'] = 0
for month in range(1, 37):  # 36 months = 3 years
    days = month * 30
    predicted_purchases = bgf.predict(
        days,
        probabilistic_clv['frequency'],
        probabilistic_clv['recency'],
        probabilistic_clv['T']
    )
    monthly_clv = predicted_purchases * probabilistic_clv['predicted_avg_value']
    discounted_clv = monthly_clv / ((1 + monthly_discount_rate) ** month)
    probabilistic_clv['CLV_3y_discounted'] += discounted_clv

print("✅ Probabilistic CLV Calculated!")

print(f"\n📊 PROBABILISTIC CLV SUMMARY:")
print("="*50)
for days in time_periods:
    clv_col = f'CLV_{days}d'
    total_clv = probabilistic_clv[clv_col].sum()
    avg_clv = probabilistic_clv[clv_col].mean()
    print(f"\n{int(days/30)}-month CLV ({days} days):")
    print(f"  • Total predicted value: £{total_clv:,.2f}")
    print(f"  • Average per customer: £{avg_clv:,.2f}")
    print(f"  • Median per customer: £{probabilistic_clv[clv_col].median():,.2f}")

print(f"\n3-Year Discounted CLV (10% discount rate):")
print(f"  • Total: £{probabilistic_clv['CLV_3y_discounted'].sum():,.2f}")
print(f"  • Average: £{probabilistic_clv['CLV_3y_discounted'].mean():,.2f}")

# Categorize customers by probabilistic CLV
probabilistic_clv['CLV_Tier_Prob'] = pd.qcut(
    probabilistic_clv['CLV_365d'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

print(f"\n🎯 PROBABILISTIC CLV TIERS:")
tier_counts_prob = probabilistic_clv['CLV_Tier_Prob'].value_counts()
for tier, count in tier_counts_prob.items():
    percentage = count / len(probabilistic_clv) * 100
    avg_clv = probabilistic_clv[probabilistic_clv['CLV_Tier_Prob'] == tier]['CLV_365d'].mean()
    print(f"  • {tier} Tier: {count:,} customers ({percentage:.1f}%) - Avg CLV: £{avg_clv:,.2f}")

# Visualize probabilistic CLV
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Distribution of 1-year CLV
axes[0].hist(probabilistic_clv['CLV_365d'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Probabilistic CLV (£)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Probabilistic CLV Distribution (1 Year)')
axes[0].axvline(probabilistic_clv['CLV_365d'].median(), color='red', linestyle='--',
               label=f'Median: £{probabilistic_clv["CLV_365d"].median():,.2f}')
axes[0].legend()

# 2. CLV by tier
colors = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}
for tier in probabilistic_clv['CLV_Tier_Prob'].unique():
    tier_data = probabilistic_clv[probabilistic_clv['CLV_Tier_Prob'] == tier]['CLV_365d']
    axes[1].hist(tier_data, bins=30, alpha=0.6, label=tier, color=colors.get(tier, 'blue'))
axes[1].set_xlabel('Probabilistic CLV (£)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Probabilistic CLV by Tier')
axes[1].legend()

# 3. Top 20 probabilistic CLV customers
top_prob_customers = probabilistic_clv.nlargest(20, 'CLV_365d')[['CLV_365d']]
axes[2].bar(range(len(top_prob_customers)), top_prob_customers['CLV_365d'], color='purple', alpha=0.7)
axes[2].set_xlabel('Customer Rank')
axes[2].set_ylabel('Probabilistic CLV (£)')
axes[2].set_title('Top 20 Customers by Probabilistic CLV')
axes[2].set_xticks(range(len(top_prob_customers)))
axes[2].set_xticklabels(top_prob_customers.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("\n💡 INSIGHTS FROM PROBABILISTIC CLV:")
print("-"*40)
print(f"1. Accounts for purchase probability and customer churn")
print(f"2. Provides uncertainty estimates (confidence intervals)")
print(f"3. Particularly useful for new/medium-frequency customers")
print(f"4. Total 1-year probabilistic CLV: £{probabilistic_clv['CLV_365d'].sum():,.2f}")
print(f"5. 3-year discounted CLV (more realistic): £{probabilistic_clv['CLV_3y_discounted'].sum():,.2f}")

print("\n" + "="*60)
print("🔄 COMPARING ALL THREE CLV APPROACHES")
print("="*60)

# Prepare comparison data
comparison_data = customer_features[['CustomerID']].copy()

# Merge all CLV predictions
comparison_data = comparison_data.merge(
    historical_clv[['CustomerID', 'CLV_Historical', 'CLV_Tier']],
    on='CustomerID',
    how='left'
)

comparison_data = comparison_data.merge(
    ml_clv_predictions[['CustomerID', 'Predicted_CLV_1Year', 'CLV_Tier_ML']],
    on='CustomerID',
    how='left'
)

comparison_data = comparison_data.merge(
    probabilistic_clv[['CLV_365d', 'CLV_Tier_Prob']],
    left_on='CustomerID',
    right_index=True,
    how='left'
)

# Rename for clarity
comparison_data = comparison_data.rename(columns={
    'CLV_365d': 'CLV_Probabilistic',
    'Predicted_CLV_1Year': 'CLV_ML'
})

print(f"✅ Combined data for {len(comparison_data):,} customers")

# Calculate correlations
print("\n📊 CORRELATION BETWEEN APPROACHES:")
correlation_matrix = comparison_data[['CLV_Historical', 'CLV_ML', 'CLV_Probabilistic']].corr()
print(correlation_matrix)

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Correlation heatmap
im = axes[0, 0].imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto')
axes[0, 0].set_xticks(range(len(correlation_matrix.columns)))
axes[0, 0].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
axes[0, 0].set_yticks(range(len(correlation_matrix.columns)))
axes[0, 0].set_yticklabels(correlation_matrix.columns)
axes[0, 0].set_title('Correlation Between CLV Methods')
plt.colorbar(im, ax=axes[0, 0])

# 2. Scatter: Historical vs ML
axes[0, 1].scatter(comparison_data['CLV_Historical'], comparison_data['CLV_ML'], alpha=0.5, s=10)
axes[0, 1].set_xlabel('Historical CLV (£)')
axes[0, 1].set_ylabel('ML CLV (£)')
axes[0, 1].set_title('Historical vs ML CLV')
max_val = max(comparison_data['CLV_Historical'].max(), comparison_data['CLV_ML'].max())
axes[0, 1].plot([0, max_val], [0, max_val], color='red', linestyle='--', alpha=0.5)

# 3. Scatter: Historical vs Probabilistic
axes[0, 2].scatter(comparison_data['CLV_Historical'], comparison_data['CLV_Probabilistic'], alpha=0.5, s=10)
axes[0, 2].set_xlabel('Historical CLV (£)')
axes[0, 2].set_ylabel('Probabilistic CLV (£)')
axes[0, 2].set_title('Historical vs Probabilistic CLV')
max_val = max(comparison_data['CLV_Historical'].max(), comparison_data['CLV_Probabilistic'].max())
axes[0, 2].plot([0, max_val], [0, max_val], color='red', linestyle='--', alpha=0.5)

# 4. CLV distributions comparison
for method in ['CLV_Historical', 'CLV_ML', 'CLV_Probabilistic']:
    axes[1, 0].hist(comparison_data[method].dropna(), bins=50, alpha=0.5, label=method)
axes[1, 0].set_xlabel('CLV (£)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('CLV Distribution Comparison')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, comparison_data['CLV_Historical'].quantile(0.99))

# 5. Tier agreement between methods
tier_agreement = comparison_data[['CLV_Tier', 'CLV_Tier_ML', 'CLV_Tier_Prob']].apply(
    lambda x: x.nunique() == 1, axis=1
)
agreement_rate = tier_agreement.mean() * 100

axes[1, 1].pie([agreement_rate, 100 - agreement_rate], 
              labels=['Agree', 'Disagree'],
              colors=['green', 'red'],
              autopct='%1.1f%%')
axes[1, 1].set_title(f'Tier Agreement: {agreement_rate:.1f}%')

# 6. Top customers comparison
top_10_historical = comparison_data.nlargest(10, 'CLV_Historical')['CustomerID']
top_10_ml = comparison_data.nlargest(10, 'CLV_ML')['CustomerID']
top_10_prob = comparison_data.nlargest(10, 'CLV_Probabilistic')['CustomerID']

# Calculate overlap
overlap_hist_ml = len(set(top_10_historical) & set(top_10_ml))
overlap_hist_prob = len(set(top_10_historical) & set(top_10_prob))
overlap_ml_prob = len(set(top_10_ml) & set(top_10_prob))

overlaps = [overlap_hist_ml, overlap_hist_prob, overlap_ml_prob]
labels = ['Hist-ML', 'Hist-Prob', 'ML-Prob']

bars = axes[1, 2].bar(labels, overlaps, color=['skyblue', 'salmon', 'lightgreen'])
axes[1, 2].set_ylabel('Number of Overlapping Customers')
axes[1, 2].set_title('Top 10 Customers Overlap')
axes[1, 2].set_ylim(0, 10)

for bar in bars:
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n📈 CLV APPROACH COMPARISON SUMMARY:")
print("="*50)

summary_stats = pd.DataFrame({
    'Method': ['Historical', 'Machine Learning', 'Probabilistic'],
    'Total CLV (£)': [
        comparison_data['CLV_Historical'].sum(),
        comparison_data['CLV_ML'].sum(),
        comparison_data['CLV_Probabilistic'].sum()
    ],
    'Avg CLV (£)': [
        comparison_data['CLV_Historical'].mean(),
        comparison_data['CLV_ML'].mean(),
        comparison_data['CLV_Probabilistic'].mean()
    ],
    'Median CLV (£)': [
        comparison_data['CLV_Historical'].median(),
        comparison_data['CLV_ML'].median(),
        comparison_data['CLV_Probabilistic'].median()
    ],
    'Std Dev (£)': [
        comparison_data['CLV_Historical'].std(),
        comparison_data['CLV_ML'].std(),
        comparison_data['CLV_Probabilistic'].std()
    ]
})

print(summary_stats)

print("\n💡 WHEN TO USE EACH APPROACH:")
print("-"*40)
print("""
1. HISTORICAL CLV:
   • Best for: Established, frequent buyers
   • Pros: Simple, easy to explain
   • Cons: Assumes past = future, ignores churn

2. MACHINE LEARNING CLV:
   • Best for: All customer types, especially with rich features
   • Pros: Can capture complex patterns, uses all available data
   • Cons: Requires more data, black-box nature

3. PROBABILISTIC CLV:
   • Best for: New/moderate frequency customers
   • Pros: Accounts for churn probability, provides uncertainty
   • Cons: Assumes specific statistical distributions
""")

print("\n" + "="*60)
print("🏆 SIMPLE ENSEMBLE CLV CREATION")
print("="*60)

def create_simple_ensemble(comparison_data):
    """Simple ensemble CLV creation"""
    
    # Create a fresh dataframe with just CustomerID
    ensemble = pd.DataFrame({'CustomerID': comparison_data['CustomerID']})
    
    # Add basic CLV columns if they exist
    for col in ['CLV_Historical', 'CLV_ML', 'CLV_Probabilistic']:
        if col in comparison_data.columns:
            ensemble[col] = pd.to_numeric(comparison_data[col], errors='coerce').fillna(0)
        else:
            print(f"⚠️  {col} not found, using zeros")
            ensemble[col] = 0
    
    # Add transaction count if available
    if 'TransactionCount' in customer_features.columns:
        ensemble = ensemble.merge(
            customer_features[['CustomerID', 'TransactionCount']], 
            on='CustomerID', 
            how='left'
        )
        ensemble['TransactionCount'] = ensemble['TransactionCount'].fillna(0)
    else:
        ensemble['TransactionCount'] = 1  # Default
    
    # Simple weighted average
    # Higher weight for historical for frequent customers
    ensemble['weight_historical'] = np.clip(ensemble['TransactionCount'] / 20, 0, 0.6)
    ensemble['weight_probabilistic'] = np.clip(1 - ensemble['TransactionCount'] / 20, 0, 0.6)
    ensemble['weight_ml'] = 0.4  # Fixed weight for ML
    
    # Ensure weights sum to 1
    total_weight = ensemble[['weight_historical', 'weight_probabilistic', 'weight_ml']].sum(axis=1)
    ensemble['weight_historical'] /= total_weight
    ensemble['weight_probabilistic'] /= total_weight
    ensemble['weight_ml'] /= total_weight
    
    # Calculate ensemble CLV
    ensemble['CLV_Final'] = (
        ensemble['weight_historical'] * ensemble['CLV_Historical'] +
        ensemble['weight_probabilistic'] * ensemble['CLV_Probabilistic'] +
        ensemble['weight_ml'] * ensemble['CLV_ML']
    )
    
    # Create tiers
    clv_percentiles = ensemble['CLV_Final'].quantile([0.33, 0.66])
    ensemble['CLV_Tier_Final'] = pd.cut(
        ensemble['CLV_Final'],
        bins=[-np.inf, clv_percentiles[0.33], clv_percentiles[0.66], np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    return ensemble

# Create simple ensemble
simple_ensemble = create_simple_ensemble(comparison_data)

print("✅ Simple Ensemble CLV Created!")
print(f"\n📊 Basic Statistics:")
print(simple_ensemble['CLV_Final'].describe())

print(f"\n🎯 Customer Tiers:")
tier_counts = simple_ensemble['CLV_Tier_Final'].value_counts()
for tier, count in tier_counts.items():
    percentage = count / len(simple_ensemble) * 100
    avg_clv = simple_ensemble[simple_ensemble['CLV_Tier_Final'] == tier]['CLV_Final'].mean()
    print(f"  • {tier}: {count:,} customers ({percentage:.1f}%) - Avg: £{avg_clv:,.2f}")

# Simple visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. CLV Distribution
axes[0].hist(simple_ensemble['CLV_Final'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[0].set_xlabel('Ensemble CLV (£)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Ensemble CLV Distribution', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 2. Tier Distribution
tier_counts.plot(kind='bar', ax=axes[1], color=['red', 'orange', 'green'])
axes[1].set_xlabel('CLV Tier')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Customer Distribution by CLV Tier', fontweight='bold')
axes[1].tick_params(axis='x', rotation=0)

# Add counts on bars
for i, v in enumerate(tier_counts.values):
    axes[1].text(i, v + 5, f'{v:,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("🚀 ACTIONABLE RECOMMENDATIONS & ROI ANALYSIS")
print("="*60)

# Calculate ROI for different customer tiers
print("\n💰 ROI ANALYSIS BY CLV TIER:")
print("-"*40)

# Assumptions (these would come from business data)
assumptions = {
    'acquisition_cost': {
        'Low': 50,     # £50 to acquire low-tier customer
        'Medium': 100, # £100 to acquire medium-tier
        'High': 200    # £200 to acquire high-tier
    },
    'retention_cost_multiplier': 0.3,  # 30% of acquisition cost to retain
    'discount_rate': 0.10,  # 10% annual discount rate
    'retention_lift': 0.20  # 20% lift from retention efforts
}

roi_analysis = []

for tier in ['Low', 'Medium', 'High']:
    tier_customers = simple_ensemble[simple_ensemble['CLV_Tier_Final'] == tier]
    avg_clv = tier_customers['CLV_Final'].mean()
    num_customers = len(tier_customers)
    acquisition_cost = assumptions['acquisition_cost'][tier]
    retention_cost = acquisition_cost * assumptions['retention_cost_multiplier']
    
    # Calculate ROI for acquisition
    acquisition_roi = (avg_clv - acquisition_cost) / acquisition_cost
    
    # Calculate ROI for retention (assuming we can increase CLV by retention_lift)
    clv_with_retention = avg_clv * (1 + assumptions['retention_lift'])
    retention_roi = (clv_with_retention - avg_clv - retention_cost) / retention_cost
    
    roi_analysis.append({
        'Tier': tier,
        'Avg_CLV': avg_clv,
        'Customers': num_customers,
        'Acquisition_Cost': acquisition_cost,
        'Acquisition_ROI': acquisition_roi,
        'Retention_Cost': retention_cost,
        'Retention_ROI': retention_roi
    })

roi_df = pd.DataFrame(roi_analysis)
print("\n📊 ROI ANALYSIS:")
print(roi_df.to_string(index=False))

# Visualize ROI
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Acquisition ROI
x = range(len(roi_df))
axes[0].bar(x, roi_df['Acquisition_ROI'], color=['red', 'orange', 'green'])
axes[0].set_xlabel('Customer Tier')
axes[0].set_ylabel('ROI (Return per £1 spent)')
axes[0].set_title('Acquisition ROI by Tier')
axes[0].set_xticks(x)
axes[0].set_xticklabels(roi_df['Tier'])
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

for i, v in enumerate(roi_df['Acquisition_ROI']):
    axes[0].text(i, v, f'£{v:.1f}', ha='center', va='bottom' if v > 0 else 'top')

# Retention ROI
axes[1].bar(x, roi_df['Retention_ROI'], color=['red', 'orange', 'green'])
axes[1].set_xlabel('Customer Tier')
axes[1].set_ylabel('ROI (Return per £1 spent)')
axes[1].set_title('Retention ROI by Tier')
axes[1].set_xticks(x)
axes[1].set_xticklabels(roi_df['Tier'])
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

for i, v in enumerate(roi_df['Retention_ROI']):
    axes[1].text(i, v, f'£{v:.1f}', ha='center', va='bottom' if v > 0 else 'top')

plt.tight_layout()
plt.show()

print("\n🎯 ACTIONABLE RECOMMENDATIONS:")
print("-"*40)

for tier in ['High', 'Medium', 'Low']:
    tier_data = roi_df[roi_df['Tier'] == tier].iloc[0]
    
    print(f"\n🔹 {tier.upper()} TIER CUSTOMERS:")
    print(f"   • Count: {tier_data['Customers']:,}")
    print(f"   • Average CLV: £{tier_data['Avg_CLV']:,.2f}")
    print(f"   • Acquisition ROI: £{tier_data['Acquisition_ROI']:.2f} per £1 spent")
    print(f"   • Retention ROI: £{tier_data['Retention_ROI']:.2f} per £1 spent")
    
    print(f"   📌 RECOMMENDED ACTIONS:")
    
    if tier == 'High':
        print("     - Priority 1: Retention & loyalty programs")
        print("     - Dedicated account managers")
        print("     - Exclusive offers and early access")
        print("     - Referral programs (they bring similar customers)")
    
    elif tier == 'Medium':
        print("     - Focus on upsell and cross-sell")
        print("     - Frequency-based rewards")
        print("     - Personalized product recommendations")
        print("     - Test different engagement strategies")
    
    else:  # Low tier
        if tier_data['Acquisition_ROI'] > 0:
            print("     - Cost-effective acquisition channels")
            print("     - Entry-level product promotions")
            print("     - Educational content to increase engagement")
            print("     - Low-cost retention efforts")
        else:
            print("     - Re-evaluate acquisition strategy")
            print("     - Focus on quality over quantity")
            print("     - Improve targeting to find higher-value customers")

# Identify specific high-value opportunities
print("\n🎯 HIGH-VALUE OPPORTUNITIES:")
print("-"*40)

# 1. High CLV but low frequency (upsell opportunity)
high_clv_low_freq = simple_ensemble[
    (simple_ensemble['CLV_Tier_Final'] == 'High') & 
    (simple_ensemble['TransactionCount'] < 5)
].sort_values('CLV_Final', ascending=False).head(10)

print(f"\n1. High CLV, Low Frequency Customers (Upsell Opportunity):")
print(f"   • {len(high_clv_low_freq):,} customers spending high amounts infrequently")
print(f"   • Average CLV: £{high_clv_low_freq['CLV_Final'].mean():,.2f}")
print(f"   • Action: Product bundles, subscription models")

# 2. High frequency but low CLV (increase transaction value)
high_freq_low_clv = simple_ensemble[
    (simple_ensemble['TransactionCount'] >= 10) & 
    (simple_ensemble['CLV_Tier_Final'] == 'Low')
].sort_values('CLV_Final', ascending=False).head(10)

print(f"\n2. High Frequency, Low CLV Customers (Increase Transaction Value):")
print(f"   • {len(high_freq_low_clv):,} customers buying frequently but low value")
print(f"   • Average transactions: {high_freq_low_clv['TransactionCount'].mean():.0f}")
print(f"   • Action: Premium product recommendations, loyalty tiers")

# 3. High confidence predictions (most reliable)
# Note: CLV_Confidence column not created in simple_ensemble, using CLV_Final percentile instead
high_confidence = simple_ensemble[
    simple_ensemble['CLV_Final'] >= simple_ensemble['CLV_Final'].quantile(0.8)
].sort_values('CLV_Final', ascending=False)

print(f"\n3. High Value Predictions (Top 20%):")
print(f"   • {len(high_confidence):,} customers in top 20% by CLV")
print(f"   • Use for: Budget planning, priority targeting")
print(f"   • Total predicted value: £{high_confidence['CLV_Final'].sum():,.2f}")

def create_production_outputs(customer_data, clv_predictions, segments):
    """Create outputs ready for deployment"""
    
    # 1. Customer master file
    customer_master = customer_data.merge(clv_predictions, on='CustomerID')
    customer_master = customer_master.merge(segments, on='CustomerID')
    
    # 2. Action recommendations
    def generate_recommendations(row):
        recommendations = []
        
        if row['CLV_Tier'] == 'High':
            recommendations.append('Priority retention program')
            recommendations.append('Personalized offers')
            if row['Recency'] > 30:
                recommendations.append('Win-back campaign')
        
        if row['Frequency'] > 10 and row['AvgTransactionValue'] < 50:
            recommendations.append('Upsell premium products')
        
        if row['Recency'] < 7:
            recommendations.append('Cross-sell complementary items')
        
        return '; '.join(recommendations[:3])  # Top 3 recommendations
    
    customer_master['Recommendations'] = customer_master.apply(generate_recommendations, axis=1)
    
    # 3. Create marketing lists
    high_value_list = customer_master[customer_master['CLV_Tier'] == 'High'][['CustomerID', 'Email', 'Recommendations']]
    at_risk_list = customer_master[customer_master['Recency'] > 90][['CustomerID', 'Email', 'Recommendations']]
    
    return {
        'customer_master': customer_master,
        'high_value_list': high_value_list,
        'at_risk_list': at_risk_list
    }

class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self):
        self.performance_history = []
    
    def track_performance(self, model_name, metrics, timestamp):
        """Track model performance"""
        self.performance_history.append({
            'timestamp': timestamp,
            'model': model_name,
            **metrics
        })
    
    def get_performance_trends(self):
        """Analyze performance trends"""
        if not self.performance_history:
            return None
        
        history_df = pd.DataFrame(self.performance_history)
        trends = history_df.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(4)
        
        return trends