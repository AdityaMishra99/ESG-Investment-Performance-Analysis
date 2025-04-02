#!/usr/bin/env python
# coding: utf-8

# # ESG (Environmental, Social, Governance) Investment Performance Analysis
# 
# ### Let's import the necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# ### Let's Load the ESG File

# In[2]:


esg = pd.read_csv("C:/Users/DELL/Downloads/archive (15)/SP 500 ESG Risk Ratings.csv")


# In[3]:


esg.head()


# ### Let's Load the Stock File using YFinance

# In[4]:


import yfinance as yf

# Define stock tickers
tickers = ["AAPL", "TSLA", "MSFT", "^GSPC"]  # Apple, Tesla, Microsoft, S&P 500 (benchmark)

# Define the date range
start_date = "2015-01-01"
end_date = "2024-03-28"

# Download historical stock data with adjusted prices
stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# Display the first few rows
print(stock_data.head())


# ### Let's Clean UP the ESG File

# ### First check the misssing values

# In[5]:


print(esg.isnull().sum())


# ### Let's drop the unnecessary Columns

# In[6]:


esg_cleaned = esg.drop(columns=["Address", "Description", "Full Time Employees"])


# ### Handle Missing Values
# Drop Rows with Many Missing Values
# The Controversy Score has 100 missing values, which might make it unreliable.
# If a row has missing values in most ESG-related columns, drop it.

# In[7]:


esg_cleaned.dropna(subset=["Total ESG Risk score", "Environment Risk Score",
                                "Governance Risk Score", "Social Risk Score", 
                                "ESG Risk Percentile"], inplace=True)


# ### Fill Missing Values for Controversy Score
# Since the Controversy Score is missing in all dropped rows, we can fill it with the mean value:

# In[8]:


esg_cleaned["Controversy Score"].fillna(esg_cleaned["Controversy Score"].mean(), inplace=True)


# ### Convert Data Types
# ### Convert Percentile to Percentage
# To fix this, we need to:
# 
# Extract the numeric part (e.g., "50th percentile" → 50).
# 
# Divide by 100 to convert it into a decimal (percentage format).
# 
# Rename the column from "ESG Risk Percentile" to "ESG Risk Percentage".

# In[9]:


numeric_columns = [
    "Total ESG Risk score", "Environment Risk Score", "Governance Risk Score",
    "Social Risk Score", "Controversy Score", "ESG Risk Percentile"
]

# Convert "ESG Risk Percentile" into percentage (e.g., "50th percentile" → 0.50)
esg_cleaned["ESG Risk Percentile"] = (
    esg_cleaned["ESG Risk Percentile"]
    .astype(str)  # Ensure it's a string for regex
    .str.extract(r"(\d+)")  # Extract numeric part
    .astype(float) / 100  # Convert to percentage
)

# Rename the column
esg_cleaned.rename(columns={"ESG Risk Percentile": "ESG Risk Percentage"}, inplace=True)

# Convert all other numeric columns to float
esg_cleaned[numeric_columns[:-1]] = esg_cleaned[numeric_columns[:-1]].apply(pd.to_numeric, errors="coerce")

# Print first few rows to verify
print(esg_cleaned.head())


# ### Let's save the cleaned data of ESG in csv formate

# In[10]:


# Save cleaned ESG data (optional)
esg_cleaned.to_csv("cleaned_data.csv", index=False)


# ### Let's Reshape Stock Data
# Since stock data has Ticker symbols as column headers, we need to convert it into a long format where Ticker is a column.

# In[11]:


# Reset index so Date becomes a column
stock_data_reset = stock_data.reset_index()

# Convert from wide to long format (unpivot tickers)
stock_data_long = stock_data_reset.melt(id_vars=["Date"], var_name=["Metric", "Ticker"], value_name="Value")

# Pivot to get each Metric as a separate column
stock_data_final = stock_data_long.pivot_table(index=["Date", "Ticker"], columns="Metric", values="Value").reset_index()

# Display the transformed stock data
print(stock_data_final.head())


# ### Let' Merge Stock Data with ESG Data
# We merge ESG data on Symbol and Ticker

# In[12]:


# Merge ESG data with stock data on 'Ticker' and 'Symbol'
merged_data = stock_data_final.merge(esg_cleaned, left_on="Ticker", right_on="Symbol", how="inner")

# Drop redundant Symbol column
merged_data.drop(columns=["Symbol"], inplace=True)

# Display merged dataset
print(merged_data.head())


# ### Let's Calculate Daily Returns
# Before we visualize correlations, we need to calculate daily stock returns using the Close Price

# In[13]:


# Convert 'Date' column to datetime format
merged_data["Date"] = pd.to_datetime(merged_data["Date"])

# Now sort data by Date for proper return calculation
merged_data.sort_values(by=["Ticker", "Date"], inplace=True)

# Calculate daily percentage change (returns) for each stock
merged_data["Daily Return"] = merged_data.groupby("Ticker")["Close"].pct_change()

# Drop NA values (first row per stock will have NaN)
merged_data.dropna(subset=["Daily Return"], inplace=True)

print(merged_data.head())


# ### Let's make Correlation Matrix (ESG Scores vs. Returns)
# 
# ✅ This heatmap will show which ESG factors are most correlated with stock returns.

# In[14]:


# Select relevant columns for correlation
correlation_data = merged_data[[
    "Daily Return", "Total ESG Risk score", "Environment Risk Score",
    "Governance Risk Score", "Social Risk Score", "Controversy Score", "ESG Risk Percentage"
]]

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between ESG Scores and Stock Returns")
plt.show()


# ### Scatter Plots (Individual Relationships)
# To visualize how each ESG score relates to stock returns, let's create scatter plots.
# 
# ✅ This will reveal if there are any patterns (positive or negative relationships) between ESG scores and stock returns.

# In[15]:


# List of ESG metrics to plot against returns
esg_factors = ["Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", 
               "Social Risk Score", "Controversy Score"]

# Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, factor in enumerate(esg_factors):
    sns.scatterplot(data=merged_data, x=factor, y="Daily Return", ax=axes[i], alpha=0.5)
    axes[i].set_title(f"Stock Returns vs {factor}")

plt.tight_layout()
plt.show()


# ### Define "Sustainable" and "Non-Sustainable" Stocks
# 
# We need a threshold to classify stocks into two groups:
# 
# ✅ High ESG (Sustainable Investments) → Low ESG Risk Score
# 
# ❌ Low ESG (Non-Sustainable Investments) → High ESG Risk Score
# 
# We can use median ESG risk score as the cutoff

# In[16]:


# Calculate the median ESG risk score
median_esg = merged_data["Total ESG Risk score"].median()

# Categorize companies based on ESG risk
merged_data["ESG Category"] = np.where(merged_data["Total ESG Risk score"] <= median_esg, "High ESG", "Low ESG")

# Display categorized data
print(merged_data[["Ticker", "Total ESG Risk score", "ESG Category"]].head())  ##Now, each stock is classified as either High ESG or Low ESG.


# ### Compare Average Returns
# 
# We compute the mean daily return for each group
# 
#  If High ESG stocks have a higher return than Low ESG stocks, sustainable investments may perform better.

# In[17]:


# Group by ESG Category and calculate average daily return
esg_return_comparison = merged_data.groupby("ESG Category")["Daily Return"].mean()

print(esg_return_comparison)


# ### Let's perform the T-Test (Statistical Significance)
# 
# To check if the difference is statistically significant, we use a t-test
# 
# ✅ Interpretation of p-value:
# 
# p < 0.05 → ESG investments significantly outperform or underperform.
# 
# p > 0.05 → No significant difference between ESG and non-ESG returns.

# In[18]:


from scipy.stats import ttest_ind

# Separate returns for High ESG and Low ESG stocks
high_esg_returns = merged_data[merged_data["ESG Category"] == "High ESG"]["Daily Return"]
low_esg_returns = merged_data[merged_data["ESG Category"] == "Low ESG"]["Daily Return"]

# Perform independent t-test
t_stat, p_value = ttest_ind(high_esg_returns, low_esg_returns, equal_var=False)

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")


# ### Visualizing Return Distributions
# 
# We compare the return distributions of High ESG vs. Low ESG stocks.
# 
# This helps us visualize if High ESG stocks have less volatility or higher returns.

# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(high_esg_returns, label="High ESG", kde=True, color="green", bins=30)
sns.histplot(low_esg_returns, label="Low ESG", kde=True, color="red", bins=30)
plt.title("Return Distribution: High ESG vs Low ESG Stocks")
plt.xlabel("Daily Return")
plt.legend()
plt.show()


# ## Now Let's Perform a regression analysis to quantify the impact of ESG scores on stock returns

# ### Prepare Data for Regression
# 
# We set daily stock returns as the dependent variable (Y) and ESG scores as independent variables (X)

# In[20]:


import statsmodels.api as sm

# Define independent variables (ESG factors)
X = merged_data[[
    "Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", 
    "Social Risk Score", "Controversy Score"
]]

# Add constant for intercept in regression
X = sm.add_constant(X)

# Define dependent variable (Stock Returns)
Y = merged_data["Daily Return"]

# Drop missing values (if any)
X = X.dropna()
Y = Y.loc[X.index]


# ### Let's Run the Regression Model
# Now, we run an Ordinary Least Squares (OLS) regression to quantify the impact of ESG scores on stock returns.
# 
# Interpret Results
# The regression output provides:
# 
# R² Value → If it's high, ESG factors explain a good portion of stock returns.
# 
# Significance Levels:
# 
# p < 0.05 → ESG factors have a significant impact.
# 
# p > 0.05 → No significant relationship.
# 
# Coefficient Signs:
# 
# Positive → Higher ESG score → Higher returns.
# 
# Negative → Higher ESG score → Lower returns.

# In[21]:


# Fit OLS regression model
model = sm.OLS(Y, X).fit()

# Display regression results
print(model.summary())


# ### Let's Check for Multicollinearity
# 
# Since ESG factors may be correlated, we check for multicollinearity using Variance Inflation Factor (VIF).
# 
# ✅ VIF < 5 → No multicollinearity issues.
# 
# ✅ VIF > 10 → Strong correlation between independent variables (remove highly correlated ones).

# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF scores
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# ## Feature Engineering & Alternative Models
# 
# Now, let's improve our model using Principal Component Analysis (PCA) to reduce collinearity and then apply Non-Linear Models (Random Forest, Neural Networks) to capture potential non-linear relationships.

# ### Let's Apply Principal Component Analysis (PCA)
# 
# PCA helps reduce multicollinearity by transforming correlated ESG factors into independent components.
# 
# ✅ PCA1 & PCA2 replace original ESG scores, reducing multicollinearity
# 
# ✅ Explained Variance Ratio shows how much ESG data is retained in PCA

# In[23]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select ESG features for PCA
esg_features = merged_data[[
    "Total ESG Risk score", "Environment Risk Score", 
    "Governance Risk Score", "Social Risk Score", "Controversy Score"
]]

# Standardize the data
scaler = StandardScaler()
esg_scaled = scaler.fit_transform(esg_features)

# Apply PCA
pca = PCA(n_components=2)  # Keep 2 principal components
esg_pca = pca.fit_transform(esg_scaled)

# Add PCA components to dataset
merged_data["PCA1"] = esg_pca[:, 0]
merged_data["PCA2"] = esg_pca[:, 1]

# Display explained variance ratio
print("Explained Variance:", pca.explained_variance_ratio_)


# ### Train a Random Forest Model
# 
# Now, let's train a Random Forest Regressor to capture potential non-linear relationships.
# 
# ✅ R² Score → Measures how well ESG scores explain stock returns
# 
# ✅ MSE (Mean Squared Error) → Measures prediction error

# In[24]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define features (PCA components) and target variable
X = merged_data[["PCA1", "PCA2"]]
Y = merged_data["Daily Return"]

# Split data into training and testing sets (80%-20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions
Y_pred = rf_model.predict(X_test)

# Evaluate model performance
print("Random Forest R² Score:", r2_score(Y_test, Y_pred))
print("Random Forest MSE:", mean_squared_error(Y_test, Y_pred))


# ### Train a Neural Network (MLP Regressor)
# 
# Let's try a Neural Network (Multi-Layer Perceptron) for potential non-linear interactions.
# 
# ✅ Hidden Layers (64,32) capture non-linear interactions
# 
# ✅ ReLU Activation helps in complex pattern learning

# In[25]:


from sklearn.neural_network import MLPRegressor

# Train Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
nn_model.fit(X_train, Y_train)

# Make predictions
Y_pred_nn = nn_model.predict(X_test)

# Evaluate model performance
print("Neural Network R² Score:", r2_score(Y_test, Y_pred_nn))
print("Neural Network MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### Check for High Noise in Daily Returns
# 
# Stock returns are highly volatile, and ESG scores may not have a strong direct relationship. Let's visualize their distribution

# In[26]:


# Check daily return distribution
sns.histplot(merged_data["Daily Return"], bins=50, kde=True)
plt.title("Distribution of Daily Returns")
plt.show()


# Returns are highly skewed or have extreme values, consider using log returns instead

# In[27]:


merged_data["Log Return"] = np.log(1 + merged_data["Daily Return"])


# In[28]:


print(merged_data[["PCA1", "PCA2", "Daily Return"]].corr())
sns.pairplot(merged_data[["PCA1", "PCA2", "Daily Return"]])
plt.show()


# We can see correlation is close to 0, then ESG scores may not directly impact daily returns.

# ### Let's Improve Feature Engineering

# Using Raw ESG Scores Instead of PCA

# In[29]:


X = merged_data[[
    "Total ESG Risk score", "Environment Risk Score", 
    "Governance Risk Score", "Social Risk Score", "Controversy Score"
]]


# ### Include Stock-Specific Features
# 
# ESG alone might not be enough. Add stock volatility, momentum, or fundamental metrics.
# 
# This adds short-term stock behavior into the model.

# In[30]:


merged_data["Volatility"] = merged_data["Daily Return"].rolling(10).std()
merged_data["Momentum"] = merged_data["Daily Return"].rolling(10).mean()
merged_data.dropna(inplace=True)  # Remove NaN rows due to rolling calculations

X = merged_data[["Total ESG Risk score", "Momentum", "Volatility"]]


# ### Split Data into Train and Test Sets
# 
# We need to update the dataset and split it into training and testing sets.

# In[31]:


from sklearn.model_selection import train_test_split

# Define target variable (Y) and features (X)
Y = merged_data["Daily Return"]
X = merged_data[["Total ESG Risk score", "Momentum", "Volatility"]]

# Split data into training and testing sets (80%-20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# ### Train and Evaluate Models Again
# 
#  Random Forest Model
# 
#  ✅ Why this should work better?
# 
# We added Momentum & Volatility, which capture market trends.
# 
# More estimators (500) and deeper trees (max_depth=10) help model complex relationships.

# In[32]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train Random Forest model with improved parameters
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions
Y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
print("Random Forest R² Score:", r2_score(Y_test, Y_pred_rf))
print("Random Forest MSE:", mean_squared_error(Y_test, Y_pred_rf))


# In[33]:


from sklearn.neural_network import MLPRegressor

# Train Neural Network with optimized hyperparameters
nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='tanh', solver='adam', max_iter=1000, random_state=42)
nn_model.fit(X_train, Y_train)

# Make predictions
Y_pred_nn = nn_model.predict(X_test)

# Evaluate performance
print("Neural Network R² Score:", r2_score(Y_test, Y_pred_nn))
print("Neural Network MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### Let's Switch to Weekly & Monthly Returns
# 
# Since ESG factors likely influence long-term trends, we’ll calculate weekly and monthly returns instead of daily.
# 

# In[34]:


# Convert daily stock prices to weekly and monthly returns
merged_data["Weekly Return"] = merged_data["Close"].pct_change(5)  # 5 trading days
merged_data["Monthly Return"] = merged_data["Close"].pct_change(21)  # ~21 trading days

# Drop NaN values after pct_change calculations
merged_data.dropna(inplace=True)


# Now, we’ll test Random Forest on these returns instead of daily returns.
# 
# ### Train Random Forest Again (on Weekly & Monthly Returns)
# 
# Modify the Random Forest model to use weekly or monthly returns as the target variable

# In[35]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Define features
X = merged_data[["Total ESG Risk score", "Momentum", "Volatility"]]

# Test with Weekly Return
Y_weekly = merged_data["Weekly Return"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_weekly, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)

print("Random Forest Weekly R² Score:", r2_score(Y_test, Y_pred))
print("Random Forest Weekly MSE:", mean_squared_error(Y_test, Y_pred))

# Test with Monthly Return
Y_monthly = merged_data["Monthly Return"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_monthly, test_size=0.2, random_state=42)

rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)

print("Random Forest Monthly R² Score:", r2_score(Y_test, Y_pred))
print("Random Forest Monthly MSE:", mean_squared_error(Y_test, Y_pred))


# ### Further Refinements
# 🔹 Try More Features:
# Let’s improve R² further by adding fundamental & macroeconomic factors (e.g., Market Cap, P/E Ratio, Interest Rates).
# 
# Get Stock Fundamentals using yfinance

# In[36]:


print(merged_data.columns)


# In[37]:


# Use "Ticker" instead of "Symbol"
tickers = merged_data["Ticker"].unique()

fundamentals = []
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals.append({
            "Ticker": ticker,  # Use "Ticker" instead of "Symbol"
            "Market Cap": info.get("marketCap", None),
            "P/E Ratio": info.get("trailingPE", None)
        })
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Convert to DataFrame
stock_fundamental_data = pd.DataFrame(fundamentals)

# Merge with `merged_data`
merged_data = merged_data.merge(stock_fundamental_data, on="Ticker", how="left")

print(merged_data.head())  # Verify the result


# ### Let's Train Random Forest with New Features
# 
# Now, update your feature set (X) to include Market Cap & P/E Ratio

# In[38]:


# Define features and target variable
X = merged_data[["Total ESG Risk score", "Momentum", "Volatility", "Market Cap", "P/E Ratio"]]
Y = merged_data["Weekly Return"]  # You can also try Monthly Return

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Predictions
Y_pred = rf_model.predict(X_test)

# Evaluate Model Performance
print("Random Forest Weekly R² Score:", r2_score(Y_test, Y_pred))
print("Random Forest Weekly MSE:", mean_squared_error(Y_test, Y_pred))


# ###  Train Neural Network with New Features
# Let's see how a Neural Network performs with these new inputs

# In[39]:


# Train Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
nn_model.fit(X_train, Y_train)

# Predictions
Y_pred_nn = nn_model.predict(X_test)

# Evaluate Model Performance
print("Neural Network Weekly R² Score:", r2_score(Y_test, Y_pred_nn))
print("Neural Network Weekly MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### That Neural Network R² Score is way off! 🚨
# 
# The Random Forest model improved slightly (from 0.344 to 0.352), but the Neural Network is completely broken.

# Before training the Neural Network, let's normalize the features using StandardScaler

# In[40]:


from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train Neural Network again
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
nn_model.fit(X_train, Y_train)

# Predictions
Y_pred_nn = nn_model.predict(X_test)

# Evaluate Model Performance
print("Neural Network Weekly R² Score:", r2_score(Y_test, Y_pred_nn))
print("Neural Network Weekly MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ###  Hyperparameter Tuning Plan
# 
# We'll experiment with:
# 
# Hidden Layers & Neurons
# 
# Current: (64, 32), Let's try (128, 64, 32) for more depth.
# 
# Optimizer Variations
# 
# Current: adam, Let’s test lbfgs (good for small datasets) & sgd (good for large datasets).
# 
# Learning Rate Adjustments
# 
# Lower learning_rate_init=0.001 or 0.0005 for better convergence.
# 
# Increase Iterations
# 
# max_iter=1000 to ensure convergence.

# In[41]:


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define hyperparameter sets to test
hyperparams = [
    {"hidden_layer_sizes": (128, 64, 32), "activation": "relu", "solver": "adam", "max_iter": 1000, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64, 32), "activation": "relu", "solver": "lbfgs", "max_iter": 1000},
    {"hidden_layer_sizes": (256, 128, 64, 32), "activation": "relu", "solver": "adam", "max_iter": 1000, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "tanh", "solver": "sgd", "max_iter": 1000, "learning_rate_init": 0.01},
]

# Try different hyperparameters
for params in hyperparams:
    print(f"\nTesting Hyperparameters: {params}")
    nn_model = MLPRegressor(**params, random_state=42)
    nn_model.fit(X_train, Y_train)

    # Predictions
    Y_pred_nn = nn_model.predict(X_test)

    # Model Evaluation
    print("Neural Network R² Score:", r2_score(Y_test, Y_pred_nn))
    print("Neural Network MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### Collect More Fundamental Factors
# We'll try to fetch additional stock fundamentals using yfinance, such as:
# 
# ROE (Return on Equity)
# 
# ROA (Return on Assets)
# 
# Debt-to-Equity Ratio
# 
# EPS (Earnings Per Share)
# 
# Current Ratio (Liquidity Measure)
# 
# Book-to-Market Ratio

# In[42]:


# List of stock tickers from ESG data
tickers = merged_data["Ticker"].unique()

# Create an empty DataFrame for fundamentals
fundamentals = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals.append({
            "Ticker": ticker,
            "Market Cap": info.get("marketCap", None),
            "P/E Ratio": info.get("trailingPE", None),
            "ROE": info.get("returnOnEquity", None),
            "ROA": info.get("returnOnAssets", None),
            "Debt-to-Equity": info.get("debtToEquity", None),
            "EPS": info.get("trailingEps", None),
            "Current Ratio": info.get("currentRatio", None),
            "Book-to-Market": 1 / info.get("priceToBook", None) if info.get("priceToBook", None) else None
        })
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Convert to DataFrame
stock_fundamental_data = pd.DataFrame(fundamentals)

# Display sample data
print(stock_fundamental_data.head())

# Merge with existing dataset
merged_data = merged_data.merge(stock_fundamental_data, on="Ticker", how="left")


# ### Update Feature Set
# Once the new data is merged, we update the features used for training

# In[43]:


print(merged_data.columns)


# ### Let's Remove Duplicates and Keep Relevant Columns

# In[44]:


# List of possible duplicate columns
duplicate_columns = [
    "Market Cap_x", "P/E Ratio_x", "Market Cap_y", "P/E Ratio_y",
    "ROE_x", "ROA_x", "Debt-to-Equity_x", "EPS_x", "Current Ratio_x", "Book-to-Market_x",
    "ROE_y", "ROA_y", "Debt-to-Equity_y", "EPS_y", "Current Ratio_y", "Book-to-Market_y"
]

# Drop only the columns that exist
merged_data = merged_data.drop(columns=[col for col in duplicate_columns if col in merged_data.columns], errors="ignore")

# Print remaining columns to verify
print("Remaining columns:", merged_data.columns)


# In[45]:


# Get unique stock tickers
tickers = merged_data["Ticker"].unique()

# List to store fundamentals
fundamentals = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals.append({
            "Ticker": ticker,
            "Market Cap": info.get("marketCap", None),
            "P/E Ratio": info.get("trailingPE", None)
        })
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Convert to DataFrame
fundamental_data = pd.DataFrame(fundamentals)

# Merge with merged_data on "Ticker"
merged_data = merged_data.merge(fundamental_data, on="Ticker", how="left")

# Check if data was added
print(merged_data[["Ticker", "Market Cap", "P/E Ratio"]].head())


# ### Let's Run Model with Cleaned Data

# In[46]:


X = merged_data[[
    "Total ESG Risk score", "Momentum", "Volatility", 
    "Market Cap", "P/E Ratio", "ROE", "ROA", "Debt-to-Equity", 
    "EPS", "Current Ratio", "Book-to-Market"
]]

# Drop rows with NaN values
X.dropna(inplace=True)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Keep target variable the same
Y = merged_data["Weekly Return"].iloc[X.index]

# Train Neural Network again
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='lbfgs', max_iter=1000)
nn_model.fit(X_train, Y_train)

# Make predictions
Y_pred_nn = nn_model.predict(X_test)

# Evaluate the updated model
print("Updated Neural Network R² Score:", r2_score(Y_test, Y_pred_nn))
print("Updated Neural Network MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### Adding Technical Indicators
# Let's compute the following indicators and integrate them into your dataset:
# 
# Relative Strength Index (RSI) → Measures momentum (overbought/oversold levels).
# 
# Moving Average Convergence Divergence (MACD) → Identifies trend direction & momentum.
# 
# Bollinger Bands (Upper, Lower, Band Width) → Measures volatility.
# 
# Exponential Moving Averages (EMA 10, EMA 50, EMA 200) → Tracks price trends.

# In[47]:


# Function to calculate RSI
def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# Function to calculate MACD
def compute_MACD(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    MACD = short_ema - long_ema
    signal = MACD.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return MACD, signal

# Function to calculate Bollinger Bands
def compute_Bollinger_Bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Function to calculate EMAs
def compute_EMA(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

# Add technical indicators to merged_data
merged_data['RSI'] = compute_RSI(merged_data)
merged_data['MACD'], merged_data['MACD_Signal'] = compute_MACD(merged_data)
merged_data['Bollinger_Upper'], merged_data['Bollinger_Lower'] = compute_Bollinger_Bands(merged_data)
merged_data['EMA_10'] = compute_EMA(merged_data, 10)
merged_data['EMA_50'] = compute_EMA(merged_data, 50)
merged_data['EMA_200'] = compute_EMA(merged_data, 200)

# Drop NaN rows generated by rolling calculations
merged_data.dropna(inplace=True)

# Display the updated dataset with technical indicators
print(merged_data[['RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'EMA_10', 'EMA_50', 'EMA_200']].head())


# ### Retraining the Neural Network & Random Forest Models

# In[48]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define the updated feature set
X = merged_data[[
    "Total ESG Risk score", "Momentum", "Volatility", 
    "Market Cap", "P/E Ratio", "ROE", "ROA", "Debt-to-Equity", 
    "EPS", "Current Ratio", "Book-to-Market", 
    "RSI", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower",
    "EMA_10", "EMA_50", "EMA_200"
]]

# Drop NaN values
X.dropna(inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Keep target variable
merged_data_cleaned = merged_data.dropna(subset=[
    "Total ESG Risk score", "Momentum", "Volatility", 
    "Market Cap", "P/E Ratio", "ROE", "ROA", "Debt-to-Equity", 
    "EPS", "Current Ratio", "Book-to-Market", 
    "RSI", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower",
    "EMA_10", "EMA_50", "EMA_200", "Weekly Return"
])

# Define features and target again
X = merged_data_cleaned[[
    "Total ESG Risk score", "Momentum", "Volatility", 
    "Market Cap", "P/E Ratio", "ROE", "ROA", "Debt-to-Equity", 
    "EPS", "Current Ratio", "Book-to-Market", 
    "RSI", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower",
    "EMA_10", "EMA_50", "EMA_200"
]]

Y = merged_data_cleaned["Weekly Return"]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)

# Train Neural Network Model
nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='lbfgs', max_iter=1000)
nn_model.fit(X_train, Y_train)
Y_pred_nn = nn_model.predict(X_test)

# Evaluate Models
print("Updated Random Forest R² Score:", r2_score(Y_test, Y_pred_rf))
print("Updated Random Forest MSE:", mean_squared_error(Y_test, Y_pred_rf))

print("Updated Neural Network R² Score:", r2_score(Y_test, Y_pred_nn))
print("Updated Neural Network MSE:", mean_squared_error(Y_test, Y_pred_nn))


# ### Feature Importance Analysis (Random Forest)
# 
# We’ll use Random Forest’s feature importance scores to identify which features have the most impact on predicting stock returns.
# 
# ✅ Plan:
# 
# 1. Extract feature importance values from the trained Random Forest model.
# 
# 2. Rank features by their contribution.
# 
# 3. Visualize the results in a bar chart.

# In[49]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure feature set (X) and target variable (Y) are defined
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions
Y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model
print("Random Forest R² Score:", r2_score(Y_test, Y_pred_rf))
print("Random Forest MSE:", mean_squared_error(Y_test, Y_pred_rf))


# In[50]:


# Extract feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)

# Sort in descending order
feature_importance = feature_importance.sort_values(ascending=False)

# Plot the feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Analysis (Random Forest)")
plt.show()


# ### Now let's compare the Neural Network & Random Forest models against benchmark models like Linear Regression and Gradient Boosting to see how well your models perform

# ### Train Benchmark Models

# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
Y_pred_lr = lr_model.predict(X_test)

# Train Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, Y_train)
Y_pred_gb = gb_model.predict(X_test)


# ### Evaluate All Models

# In[52]:


# Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate scores
models = {
    "Linear Regression": (Y_test, Y_pred_lr),
    "Gradient Boosting": (Y_test, Y_pred_gb),
    "Random Forest": (Y_test, Y_pred_rf),
    "Neural Network": (Y_test, Y_pred_nn)
}

# Print R² Score & MSE for each model
for model_name, (y_true, y_pred) in models.items():
    print(f"{model_name} R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"{model_name} MSE: {mean_squared_error(y_true, y_pred):.6f}")
    print("-" * 40)


# ### Compare Performance Visually

# In[53]:


# Create bar plots for comparison
r2_scores = {name: r2_score(y_true, y_pred) for name, (y_true, y_pred) in models.items()}
mse_scores = {name: mean_squared_error(y_true, y_pred) for name, (y_true, y_pred) in models.items()}

plt.figure(figsize=(15, 10))

# R² Score comparison
plt.subplot(1, 2, 1)
plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison (R² Score)")

# MSE comparison
plt.subplot(1, 2, 2)
plt.bar(mse_scores.keys(), mse_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.ylabel("Mean Squared Error")
plt.title("Model Performance Comparison (MSE)")

plt.xticks(rotation=15)
plt.show()


# ## Perform the Visualize Predictions vs. Actual Returns

# ### Scatter Plot (Predicted vs. Actual)
# This will show how close the predicted returns are to actual returns.
# 
# 🔹 Red dashed line represents the ideal case where predicted = actual.
# 
# 🔹 If points align closely with the line, the model is performing well.

# In[54]:


# Define models for visualization
model_predictions = {
    "Random Forest": Y_pred_rf,
    "Neural Network": Y_pred_nn,
    "Linear Regression": Y_pred_lr,
    "Gradient Boosting": Y_pred_gb
}

plt.figure(figsize=(12, 8))

for i, (model, Y_pred) in enumerate(model_predictions.items(), 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.5)
    plt.plot(Y_test, Y_test, color='red', linestyle='dashed')  # Ideal line
    plt.xlabel("Actual Weekly Return")
    plt.ylabel("Predicted Weekly Return")
    plt.title(f"{model} Predictions vs. Actuals")

plt.tight_layout()
plt.show()


# ### Line Plot (Time-Series View)
# This will compare how the predicted returns track the actual returns over time.
# 
# 🔹 If predicted lines closely follow the actual returns, the model is making accurate forecasts.

# In[55]:


plt.figure(figsize=(14, 6))

# Plot actual vs. predicted returns for Neural Network
plt.plot(Y_test.values, label="Actual Returns", color='black', linewidth=2)
plt.plot(Y_pred_nn, label="Neural Network Predicted Returns", linestyle="dashed", color="red")
plt.plot(Y_pred_rf, label="Random Forest Predicted Returns", linestyle="dashed", color="blue")
plt.plot(Y_pred_gb, label="Gradient Boosting Predicted Returns", linestyle="dashed", color="green")

plt.xlabel("Time (Test Set Index)")
plt.ylabel("Weekly Return")
plt.title("Predicted vs. Actual Returns Over Time")
plt.legend()
plt.show()


# ### Histogram of Prediction Errors
# 
# This shows how far predictions deviate from actual returns.
# 
# 🔹 A narrow, centered distribution around 0 means fewer prediction errors and a better model.

# In[56]:


errors_nn = Y_test - Y_pred_nn
errors_rf = Y_test - Y_pred_rf

plt.figure(figsize=(10, 5))
sns.histplot(errors_nn, bins=50, kde=True, color="red", label="Neural Network Errors", alpha=0.6)
sns.histplot(errors_rf, bins=50, kde=True, color="blue", label="Random Forest Errors", alpha=0.6)

plt.axvline(0, color='black', linestyle="dashed")
plt.xlabel("Prediction Error")
plt.title("Distribution of Prediction Errors")
plt.legend()
plt.show()


# ## Backtest trading strategies using Predicted vs. Actual Returns.
# 
# We'll evaluate how well your model performs in a trading strategy by simulating trades based on predicted vs. actual returns.
# 
# We'll compare different strategies based on predicted vs. actual returns. Here are some approaches we can test:
# 
# 1. Buy & Hold Strategy – Invest in the stock and hold it throughout the period.
# 
# 2. Mean Reversion Strategy – Buy when predicted return is significantly lower than historical average and sell when it's higher.
# 
# 3. Momentum Strategy – Buy when predicted return is positive and sell when it's negative.
# 
# 4. Threshold-Based Strategy – Buy only if predicted return is above a certain threshold and sell if below another.

# In[57]:


print("Columns in merged_data:", merged_data.columns)


# In[58]:


# Create a date range starting from a specific date (e.g., '2020-01-01') assuming daily data
merged_data['Date'] = pd.date_range(start='2020-01-01', periods=len(merged_data), freq='D')


# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the index is in datetime format
if not isinstance(merged_data.index, pd.DatetimeIndex):
    merged_data.index = pd.to_datetime(merged_data.index)

# Define the required features for standardization
required_features = [
    "Total ESG Risk score", "Momentum", "Volatility", 
    "Market Cap", "P/E Ratio", "ROE", "ROA", "Debt-to-Equity", 
    "EPS", "Current Ratio", "Book-to-Market",
    "Bollinger_Lower", "Bollinger_Upper", "EMA_10", "EMA_50", "EMA_200",
    "RSI", "MACD", "MACD_Signal", "Year"  # Added "Year" as a feature
]

# Ensure only existing columns are used (avoid missing features error)
existing_features = [feature for feature in required_features if feature in merged_data.columns]

# Reorder existing_features to match the order used during scaler fitting
existing_features = [feature for feature in scaler.feature_names_in_ if feature in merged_data.columns]

# Drop rows with NaN values in the selected features
merged_data.dropna(subset=existing_features, inplace=True)

# Check if any features are missing from merged_data compared to the scaler
missing_features = set(scaler.feature_names_in_) - set(merged_data.columns)
if missing_features:
    raise ValueError(f"Missing Features: {missing_features}")

# Transform data using the fitted scaler
X_scaled = scaler.transform(merged_data[existing_features])

# Generate predictions using the trained Neural Network model
try:
    merged_data['Predicted Return'] = nn_model.predict(X_scaled)
except NameError:
    raise ValueError("Neural Network model (nn_model) is not defined or not trained.")

# Define 'Actual Return' if not present
if "Actual Return" not in merged_data.columns:
    merged_data["Actual Return"] = merged_data["Daily Return"]  # Use Daily Return as Actual Return

# Backtesting Function
def backtest(strategy_name, positions):
    capital = 100000  # Initial capital
    positions = pd.Series(positions, index=merged_data.index)
    
    # Shift positions by 1 day (simulate previous day execution)
    returns = positions.shift(1) * merged_data["Actual Return"]
    
    # Replace NaN with 0
    returns.fillna(0, inplace=True)
    
    # Compute cumulative portfolio value
    portfolio_value = (1 + returns).cumprod() * capital
    
    return portfolio_value

# Buy & Hold Strategy (Always Invested)
buy_hold_positions = np.ones(len(merged_data))

# Mean Reversion Strategy
merged_data["Mean Return"] = merged_data["Actual Return"].rolling(window=10, min_periods=1).mean()
mean_reversion_positions = np.where(merged_data["Predicted Return"] < merged_data["Mean Return"], 1, -1)

# Momentum Strategy (Invest if Predicted Return > 0)
momentum_positions = np.where(merged_data["Predicted Return"] > 0, 1, -1)

# Threshold-Based Strategy (Invest if above/below threshold)
threshold_positions = np.where(
    merged_data["Predicted Return"] > 0.01, 1, 
    np.where(merged_data["Predicted Return"] < -0.01, -1, 0)
)

# Backtest Strategies
results = {
    "Buy & Hold": backtest("Buy & Hold", buy_hold_positions),
    "Mean Reversion": backtest("Mean Reversion", mean_reversion_positions),
    "Momentum": backtest("Momentum", momentum_positions),
    "Threshold": backtest("Threshold", threshold_positions)
}

# Plot Results
plt.figure(figsize=(12, 6))
for strategy, portfolio_value in results.items():
    plt.plot(portfolio_value, label=strategy)

plt.legend()
plt.title("Backtest Results: Trading Strategies")
plt.xlabel("Year")  # Updated to show 'Year' instead of full Date
plt.ylabel("Portfolio Value")
plt.grid()
plt.show()

# Print Final Portfolio Values
for strategy, portfolio_value in results.items():
    print(f"Final Portfolio Value ({strategy}): ${portfolio_value.iloc[-1]:,.2f}")


# ## 📌 Monte Carlo Simulation Approach
# 
# 1. Generate multiple random scenarios for future stock returns.
# 
# 2. Simulate portfolio growth over a given time horizon.
# 
# 3. Analyze risk metrics such as VaR (Value at Risk) and Expected Shortfall.
# 
# ### 🔍 Explanation:
# ✅ Simulates 1000 possible portfolio paths for 1 year
# 
# ✅ Uses historical return mean & volatility to generate scenarios
# 
# ✅ Plots simulated paths to visualize potential outcomes
# 
# ✅ Calculates Value at Risk (VaR) at a 95% confidence level

# In[62]:


import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo parameters
simulations = 1000  # Number of simulated paths
time_horizon = 252  # Number of trading days (1 year)
initial_capital = 100000  # Starting portfolio value

# Extract the daily return distribution from historical data
returns = merged_data['Actual Return'].dropna()
mean_return = np.mean(returns)
std_dev = np.std(returns)

# Monte Carlo Simulation
simulated_portfolios = np.zeros((simulations, time_horizon))

for i in range(simulations):
    simulated_returns = np.random.normal(mean_return, std_dev, time_horizon)
    simulated_portfolios[i] = initial_capital * np.cumprod(1 + simulated_returns)

# Plot Monte Carlo Simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_portfolios.T, color='blue', alpha=0.1)
plt.title("Monte Carlo Simulation of Portfolio Value Over 1 Year")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.grid()
plt.show()

# Compute Value at Risk (VaR) at 95% confidence level
var_95 = np.percentile(simulated_portfolios[:, -1], 5)
expected_loss = initial_capital - var_95

print(f"95% Value at Risk (VaR): ${expected_loss:,.2f}")


# In[ ]:




