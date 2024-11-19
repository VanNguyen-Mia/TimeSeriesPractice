# module_1.py
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sktime.split import temporal_train_test_split

# 1. Load data
def read_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset

    Args:
        file_path (str): path to the dataset folder

    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    df = pd.read_csv(file_path)
    return df

# 2. Preprocess data & EDA
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change timestamp to datatime format

    Args:
        df (pd.DataFrame): the dataset

    Returns:
        pd.DataFrame: new dataset with datetime object
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    print("Descriptive statistics:")
    print(df.describe(include='all').T)
    print("\nMissing values analysis:")
    print(df[df.isna().any(axis=1)])

    return df.asfreq('5min')

# 3. Seasonality analysis
def seasonality_analysis(df: pd.DataFrame, save_path: str, save_name: str) -> str:
    """
    Analyse seasonality and plot time series data

    Args:
        df (pd.DataFrame): the preprocessed dataset
        save_path (str): path of the folder to save the plots
        save_name (str): prefered name tagging for plots

    Returns:
        str: paths to the plots
    """
    # Create directories for saving plot images if they don't exist
    images_folder = os.path.join(save_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Plot the original time series
    plt.figure(figsize=(7, 5))
    plt.plot(df['value'], label='Original Time Series')
    plt.title('Original Time Series')
    plt.legend()
    original_time_series_path = os.path.join(images_folder, f"original_timeseries_{save_name}.png")
    plt.savefig(original_time_series_path)

    # Plot the time series decomposition
    decomposition = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq', period=288)
    decomposition.plot()
    plt.suptitle('Time Series Decomposition')
    plt.tight_layout()
    timeseries_decomposition_path = os.path.join(images_folder, f"timeseries_decomposition_{save_name}.png")
    plt.savefig(timeseries_decomposition_path)
    
    # Plot the seasonal component
    plt.figure(figsize=(6, 4))
    plt.plot(decomposition.seasonal, label='Seasonal Component')
    plt.title('Seasonal Component')
    plt.xlabel('Day')
    plt.ylabel('Seasonal Component')
    seasonal_component_path = os.path.join(images_folder, f"seasonal_component_{save_name}.png")
    plt.savefig(seasonal_component_path)

    # Plot original data with and without seasonal component
    plt.figure(figsize=(7, 4))
    plt.plot(df, label='Original Time Series', color='blue')
    data_without_seasonal = df['value'] / decomposition.seasonal
    plt.plot(data_without_seasonal, label='Original Data without Seasonal Component', color='green')
    plt.title('Time series with and without Seasonal Component')
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.legend()
    comparison_path = os.path.join(images_folder, f"without_seasonal_component{save_name}.png")
    plt.savefig(comparison_path)

    return original_time_series_path, timeseries_decomposition_path,  seasonal_component_path, comparison_path

# 4. Outlier removal
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the dataset using boxplot and z-scores

    Args:
        df (pd.DataFrame): the dataset

    Returns:
        pd.DataFrame: cleaned dataset
    """
    z_scores = stats.zscore(df['value'].dropna())
    outliers = np.abs(z_scores) > 3
    cleaned_df1 = df.loc[~outliers]

    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df2 = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return cleaned_df1, cleaned_df2

# 5. Stationarity check
def stationarity_check(cleaned_df2: pd.DataFrame) -> bool:
    """
    Check the stationarity of the timeseries

    Args:
        cleaned_df2 (pd.DataFrame): data cleaned with boxplot

    Returns:
        bool: check stationarity of data. 
            True: the data is stationary, 
            False: the data isn't stationary
    """
    result = adfuller(cleaned_df2["value"].dropna())
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] > 0.05:
        cleaned_df2["value"] = cleaned_df2["value"].diff().dropna()
    return result[1] < 0.05

# 6. Finding p and q
def autocorrelation_plot(cleaned_df2: pd.DataFrame, save_path, test_size = 36) -> str:
    """
    Plot auto-correlation to find p and q

    Args:
        cleaned_df2 (pd.DataFrame): the cleaned dataset

        test_size (int, optional): choose test size. Defaults to 36.

    Returns:
        str: _description_
    """
    images_folder = os.path.join(save_path, "images")
    os.makedirs(images_folder, exist_ok=True)

#############################################################

# 7. Forecasting with ARIMA
def arima_forecast(train, test, order=(1, 0, 0)):
    train, test = temporal_train_test_split(df, test_size=36)

    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='Forecast')
    plt.legend()
    plt.title('ARIMA Forecast')
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"RMSE: {rmse}")
    return forecast, rmse
