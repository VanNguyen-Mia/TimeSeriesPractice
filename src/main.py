import argparse
from module_1 import (
    create_plot_directory, read_data, preprocess_data, seasonality_analysis,
    remove_outliers, stationarity_check, auto_arima_forecast
)

def main():
    parser = argparse.ArgumentParser(description="Time Series Analysis and Forecasting")
    parser.add_argument('file_path', type=str, help="Path to the CSV file containing time series data.")
    parser.add_argument('save_path', type=str, help="Path to save analysis and forecast results.")
    parser.add_argument('--save_name', type=str, default="forecast", help="Base name for saved plots.")
    args = parser.parse_args()

    # Create the plot directory
    create_plot_directory(args.save_path)

    # 1. Load and preprocess data
    df = read_data(args.file_path)
    df = preprocess_data(df)

    # 2. Seasonality analysis
    seasonality_analysis(df, args.save_path, args.save_name)

    # 3. Outlier removal
    cleaned_df2 = remove_outliers(df)

    # 4. Stationarity check
    stationary_df = stationarity_check(cleaned_df2)

    # 5. Auto-ARIMA forecasting
    m = 12
    auto_arima_forecast(stationary_df, m, args.save_path, args.save_name)


if __name__ == '__main__':
    main()
