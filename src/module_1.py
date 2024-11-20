# module_1.py
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sktime.split import temporal_train_test_split
from math import sqrt
import pmdarima as pm

# Create directory
def create_plot_directory(save_path: str, subfolder: str = "images") -> str:
    """
    Create directory for saving plot images if they don't exist

    Args:
        save_path (str): path to the parent folder
        subfolder (str): subfolder name for plots

    Returns:
        str: path to the created subfolder
    """
    images_folder = os.path.join(save_path, subfolder)
    os.makedirs(images_folder, exist_ok=True)
    return images_folder

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
    print("Missing values analysis:")
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
    images_folder = create_plot_directory(save_path)

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

    Q1 = cleaned_df1['value'].quantile(0.25)
    Q3 = cleaned_df1['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df2 = cleaned_df1[(cleaned_df1['value'] >= lower_bound) & 
                              (cleaned_df1['value'] <= upper_bound)]
    return cleaned_df2

# 5. Stationarity check
def stationarity_check(df: pd.DataFrame) -> bool:
    """
    Check the stationarity of the timeseries

    Args:
        df (pd.DataFrame): data cleaned with boxplot

    Returns:
        bool: check stationarity of data. 
            True: the data is stationary, 
            False: the data isn't stationary
    """
    result = adfuller(df["value"].dropna())
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] > 0.05:
        df["value"] = df["value"].diff().dropna()
    return df

# 6. Forecasting with ARIMA
def auto_arima_forecast(df: pd.DataFrame, m: int, save_path: str, save_name: str) -> str:
    """
    Plot Arima forecasting

    Args:
        df (pd.DataFrame): Cleaned dataset
        m (int): frequecy of series, passed into auto_arima model
        save_path (str): path to save the images
        save_name (str): saved name of the images

    Returns:
        str: Path to the timeseries prediction graph
    """

    train, test = temporal_train_test_split(df, test_size=36)

    auto_model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      m=m, # frequency of series
                      d=None, # let model determine 'd'
                      seasonal=True, # Use Seasonality
                      information_criterion='aicc',
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
    print(auto_model.summary())

    images_folder = create_plot_directory(save_path)

    # Plot diagnostics graphs
    auto_model.plot_diagnostics(figsize=(14, 6))
    diagnostics_path = os.path.join(images_folder, f"diagnostics_plot_{save_name}.png")
    plt.savefig(diagnostics_path)
    plt.close()

    # Graph the prediction
    history = train
    auto_predictions = auto_model.predict(len(test))
    test.loc[:, 'auto_predictions'] = auto_predictions.copy()
    plt.plot(test['auto_predictions'], color='red', label='prediction')
    plt.plot(test['value'], color='blue', label='actual')
    plt.legend(loc="upper left")
    prediction_path = os.path.join(images_folder, f"prediction_{save_name}.png")
    plt.savefig(prediction_path) 

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test['auto_predictions'], test['value']))
    print('Test RMSE: %.3f' % rmse)

    return diagnostics_path, prediction_path
