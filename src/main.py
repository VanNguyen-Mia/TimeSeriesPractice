import argparse
from module_1 import (
    read_data, preprocess_data, seasonality_analysis,
    remove_outliers, stationarity_check, arima_forecast
)

def main():
    parser = argparse.ArgumentParser(description="Time Series Analysis and Forecasting")
    parser.add_argument('file_path', type=str, help="Path to the CSV file containing time series data.")
    args = parser.parse_args()

    # Load and preprocess data
    df = read_data(args.file_path)
    df = preprocess_data(df)

    # Task 1: Seasonality analysis
    decomposition = seasonality_analysis(df)

    # Task 2: Outlier removal
    cleaned_df1, cleaned_df2 = remove_outliers(df)

    # Task 3: ARIMA Forecasting
    df_stationary, is_stationary = stationarity_check(cleaned_df2)
    train_size = int(len(df_stationary) * 0.8)
    train, test = df_stationary['value'][:train_size], df_stationary['value'][train_size:]
    forecast, rmse = arima_forecast(train, test)


if __name__ == '__main__':
    main()
