# ESG-Investment-Performance-Analysis
This project analyzes the relationship between Environmental, Social, and Governance (ESG) metrics and stock performance. Using S&P 500 ESG Risk Ratings combined with historical stock data from major companies (Apple, Tesla, Microsoft) and index benchmarks, the analysis explores whether ESG factors can predict returns and inform effective trading strategies.
Key Research Questions

Do ESG scores correlate with daily/weekly stock returns?
Do high-ESG stocks exhibit different risk-return profiles than low-ESG stocks?
Can machine learning models effectively predict returns using ESG data?
Which ESG components (Environmental, Social, Governance) have the greatest impact on performance?
How do ESG-based trading strategies perform compared to benchmarks?

Data Sources

ESG Data: S&P 500 ESG Risk Ratings from Kaggle
Stock Data: Historical price data for Apple, Tesla, Microsoft, and S&P 500 index from Yahoo Finance

Methodologies

Data Preprocessing

Cleaning, normalization, and feature engineering
Merging ESG metrics with financial data


Statistical Analysis

Correlation analysis between ESG factors and returns
T-tests comparing high vs. low ESG performance
Multicollinearity assessment using VIF


Machine Learning Models

Linear Regression
Random Forest
Neural Networks
Gradient Boosting


Advanced Analysis

Feature importance evaluation
Principal Component Analysis (PCA)
Time series prediction visualization
Monte Carlo simulation


Trading Strategy Backtesting

Buy & Hold
Mean Reversion
Momentum
Threshold-based approach



Key Findings

1. Minimal correlation between ESG metrics and daily stock returns
2. High ESG stocks demonstrate slightly lower volatility compared to low ESG stocks
3. Technical indicators dominate feature importance, with ESG metrics showing limited direct impact on short-term returns
4. Advanced ML models can achieve R² scores of up to 0.62-0.68 when combining ESG data with technical factors
5. Buy & Hold strategy outperformed other approaches in backtesting
6. Monte Carlo simulations reveal significant range of potential outcomes in ESG-focused portfolios
