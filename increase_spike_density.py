## IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error

# from statsmodels.tsa.stattools import adfuller

## READ THE DATA 
df = pd.read_csv('./art_increase_spike_density.csv')

################################ TASK 1: SEASONALIY ANAYSIS #####################################
# Getting descriptive statistics
print('Descriptive statistics:')
print(df.describe(include='all').T)

# Missing value analysis
print("Missing value analysis:")
print(df[df.isna().any(axis=1)])

# Set timestamp column as index for time series data
df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%Y-%m-%d %H:%M:%S')
df.set_index('timestamp', inplace = True)
print(df.dtypes)

# Plot the original time series data
plt.figure(figsize=(7, 5))
plt.plot(df, label='Original Time Series')
plt.title('art_increase_spike_density')
plt.xlabel('timestamp')
plt.ylabel('value')
plt.legend()
plt.show()

# Decompose the time series into trend, seasonal and residual components
df = df.asfreq('5min')
result = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq', period=288)
result.plot()
plt.suptitle('Seasonal Decomposition of art_increase_spike_density')
plt.tight_layout()
plt.show()

# Plot the seasonal component
plt.figure(figsize=(6, 4))
plt.plot(result.seasonal, label='Seasonal Component')
plt.title('Seasonal Component of art-increase_spike_density')
plt.xlabel('Day')
plt.ylabel('Seasonal Component')
plt.legend()
plt.show()

# Plotting the original data and original data without the seasonal component
plt.figure(figsize=(7, 4))
# Plot the original time series data
plt.plot(df, label='Original Time Series', color='blue')
data_without_seasonal = df['value'] / result.seasonal
# Plot the original data without the seasonal component
plt.plot(data_without_seasonal, label='Original Data without Seasonal Component', color='green')
plt.title('art_increase_spike_density with and without Seasonal Component')
plt.xlabel('Day')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

################################ TASK 2: REMOVE OUTLIERS USING Z-SCORES AND BOXPLOTS ################################################ 
# Using Z-scores:
z_scores = stats.zscore(df['value'].dropna())
threshold = 3
outliers = np.abs(z_scores) > threshold
cleaned_df1 = df.loc[~outliers]

print("Z-scores:", z_scores)
print("Outliers:", outliers)
print("Cleaned data:", cleaned_df1)

# Using Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['value'].dropna())
plt.title("Boxplot of 'value'")
plt.xlabel('Value')
plt.show()

# Clean df
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to remove outliers
cleaned_df2 = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

print("Data after removing outliers:")
print(cleaned_df2)

################################ TASK 3: FORECAST TIME SERIES USING ARIMA ########################################
# Check for stationarity:
result = adfuller(df["value"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Perform differencing:
if result[1] > 0.05:  
    df["value"] = df["value"].diff().dropna()
    result = adfuller(df["value"])
stationarity_interpretation = "Stationary" if result[1] < 0.05 else "Non-Stationary"

print(f"ADF Statistic after differencing: {result[0]}")
print(f"p-value after differencing: {result[1]}")
print(f"Interpretation: The series is {stationarity_interpretation}.")

# Finding the ARIMA terms
# Finding lags:
pd.plotting.autocorrelation_plot(df)
plt.title('Autocorrelation plot')

# d = 0 because no differencing is needed
# Find q - the number of lags where ACF cuts off
# Find p - the number of lags where PACF cuts off
plot_acf(df['value'], lags = 500)
plt.title('ACF')
plt.show() # Result shows q = 1


plot_pacf(df['value'], lags = 500)
plt.title('PACF')
plt.show() # Result shows p = 1

# Fit the ARIMA model
# Initial ARIMA Model parameters
p, d, q = 1, 0, 1
model = ARIMA(df, order=(p, d, q))
model_fit = model.fit()
model_summary = model_fit.summary()
print('Model summary:', {model_summary})

# Residual plot
# Plot residual errors
residuals = model_fit.resid
residuals.plot()
residuals.plot(kind='kde')
plt.legend()
plt.title('Plot residual errors')
plt.show()

# Results from simpler models to avoid overfitting
print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

## Forecasting
# Use a train/test workflow
data = df['value']
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
# Fit the model to training data. Replace p, d, q with our ARIMA parameters
model = ARIMA(train, order=(p, d, q))  
model_fit = model.fit()
# Forecast
forecast = model_fit.forecast(steps=len(test))

## Visualise time series
# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data.index[:train_size], train, label='Train', color='blue')
plt.plot(data.index[train_size:], test, label='Test', color='green')
plt.plot(data.index[train_size:], forecast, label='Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecast vs Actual')
plt.show()

# Evaluate model statistics
# Evaluate model performance on the test set
rmse = root_mean_squared_error(test, forecast)
print(f"RMSE: {rmse}")