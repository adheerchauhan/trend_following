# Import all the necessary modules
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import itertools
import yfinance as yf
import seaborn as sn
from IPython.core.display import display, HTML


def jupyter_interactive_mode():
    display(HTML(
    '<style>'
    '#notebook { padding-top:0px !important; }'
    '.container { width:100% !important; }'
    '.end_space { min-height:0px !important}'
    '</style'
    ))

# Function to pull financial data for a ticker using Yahoo Finance's API
def load_financial_data(start_date, end_date, ticker):
    output_file = f'data_folder/{ticker}-pickle-{start_date}-{end_date}'
    try:
        df = pd.read_pickle(output_file)
        print(f'File data found...reading {ticker} data')
    except FileNotFoundError:
        print(f'File not found...downloading the {ticker} data')
        df = yf.download(ticker, start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def get_close_prices(start_date, end_date, ticker, close_col='Adj Close'):
    if isinstance(ticker, str):
        ticker = [ticker]
    df = load_financial_data(start_date, end_date, ticker)
    if len(ticker) > 1:
        df = df[close_col]
    else:
        df = df[[close_col]]
        df.columns = ticker

    return df


# Function to run a linear regression analysis using in-sample and out-of sample datasets and return the regression
# model and the predicted target variable using the in sample and out of sample data
def linear_regression(x_train, y_train, x_test, y_test):
    # Create a model to fit independent variables to a target variable using ordinary least squares
    regr = linear_model.LinearRegression()

    # Train the model using training sets
    regr.fit(x_train, y_train)

    # Make predictions based on the model using the training and testing dataset
    y_pred_out = regr.predict(x_test)
    y_pred_in = regr.predict(x_train)

    # Print the coefficients of the model
    print('Coefficients: \n', regr.coef_[0])

    # Print the Mean Squared Error
    print('Mean Squared Error for in sample data: %.4f' % mean_squared_error(y_train, y_pred_in))
    print('Mean Squared Error for out of sample data: %.4f' % mean_squared_error(y_test, y_pred_out))

    # Print the Variance Score
    print('R2 Variance Score for in sample data: %.4f' % r2_score(y_train, y_pred_in))
    print('R2 Variance Score for out of sample data: %.4f' % r2_score(y_test, y_pred_out))

    # Plot the outputs for in sample and out of sample data
    fig = plt.figure(figsize=(12, 6))
    plt.style.use('bmh')
    layout = (1, 2)
    in_sample_ax = plt.subplot2grid(layout, (0, 0))
    out_sample_ax = plt.subplot2grid(layout, (0, 1))
    in_sample_ax.scatter(y_pred_in, y_train)
    in_sample_ax.plot(y_train, y_train, linewidth=3)
    in_sample_ax.set_xlabel('Y (actual)')
    in_sample_ax.set_ylabel('Y (predicted)')
    in_sample_ax.set_title('In Sample Predicted vs Actual Scatter Plot')

    out_sample_ax.scatter(y_pred_out, y_test)
    out_sample_ax.plot(y_test, y_test, linewidth=3)
    out_sample_ax.set_xlabel('Y (actual)')
    out_sample_ax.set_ylabel('Y (predicted)')
    out_sample_ax.set_title('Out of Sample Predicted vs Actual Scatter Plot')
    plt.tight_layout()

    return regr, y_pred_in, y_pred_out


# Function to run a LASSO or Ridge regression analysis with a tuning parameter specified and return the model with
# coefficients, RMSE and R^2 values
def regularization_regression(x_train, y_train, x_test, y_test, alpha, regression_type='Ridge'):
    # Create a regression model based on specified input (it will default to Ridge if nothing is specified)
    if regression_type == 'Lasso':
        # Create a model to fit independent variables to a target variable using ordinary least squares
        regr = linear_model.Lasso(alpha)
    elif regression_type == 'Ridge':
        regr = linear_model.Ridge(alpha)
    else:  # Defaults to using Ridge
        regr = linear_model.Ridge(alpha)

    # Train the model using training sets
    regr.fit(x_train, y_train)

    # Make predictions based on the model using the training and testing dataset
    y_pred_train = regr.predict(x_train)
    y_pred_test = regr.predict(x_test)

    # Calculate the Mean Squared Error
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Calculate the Variance Score
    r_squared_train = r2_score(y_train, y_pred_train)
    r_squared_test = r2_score(y_test, y_pred_test)

    return regr, regr.coef_, r_squared_train, r_squared_test, mse_train, mse_test


# Function to build a feature for momentum based on user defined time period
def momentum(df, period):
    return df.sub(df.shift(period), fill_value=0)


def get_returns_volatility(df, vol_range_list=[10], close_px_col='BTC-USD'):
    df[f'{close_px_col}_pct_returns'] = df[close_px_col].pct_change()
    for vol_range in vol_range_list:
        df[f'{close_px_col}_volatility_{vol_range}'] = df[f'{close_px_col}_pct_returns'].rolling(vol_range).std()

    return df


def calculate_slope(df, column, periods):
    return (df[column] - df[column].shift(int(periods))) / periods


def trend_signal(row):
    if all(row[i] <= row[i+1] for i in range(len(row) - 1)):
        return -1
    elif all(row[i] >= row[i+1] for i in range(len(row) - 1)):
        return 1
    else:
        return 0

def slope_signal(row):
    if all(row[i] <= row[i+1] for i in range(len(row) - 1)):
        return -1
    elif all(row[i] >= row[i+1] for i in range(len(row) - 1)):
        return 1
    else:
        return 0


def sharpe_ratio(df, strategy_daily_return_col, annual_trading_days=252, annual_rf=0.01):

    daily_rf = (1 + annual_rf) ** (1/annual_trading_days) - 1
    average_daily_return = df[strategy_daily_return_col].mean()
    std_dev_daily_returns = df[strategy_daily_return_col].std()
    daily_sharpe_ratio = (average_daily_return - daily_rf)/std_dev_daily_returns
    annualized_sharpe_ratio = daily_sharpe_ratio * np.sqrt(annual_trading_days)

    return annualized_sharpe_ratio


def create_trend_strategy(df, ticker, mavg_start, mavg_end, mavg_stepsize, vol_range_list=[10, 20, 30, 60, 90],
                          moving_avg_type='simple'):
    for window in np.linspace(mavg_start, mavg_end, mavg_stepsize):
        if moving_avg_type == 'simple':
            df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}'].rolling(int(window)).mean()
        else:
            df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}'].ewm(span=window).mean()
        df[f'{ticker}_{int(window)}_mavg_slope'] = calculate_slope(df, column=f'{ticker}_{int(window)}_mavg',
                                                                   periods=window)

    df[f'{ticker}_ribbon_thickness'] = df[f'{ticker}_{int(mavg_start)}_mavg'] - df[f'{ticker}_{int(mavg_end)}_mavg']
    df = get_returns_volatility(df, vol_range_list=vol_range_list, close_px_col=ticker)

    ## Ticker Trend Signal and Trade
    mavg_col_list = [f'{ticker}_{int(mavg)}_mavg' for mavg in np.linspace(mavg_start, mavg_end, mavg_stepsize).tolist()]
    mavg_slope_col_list = [f'{ticker}_{int(mavg)}_mavg_slope' for mavg in
                           np.linspace(mavg_start, mavg_end, mavg_stepsize).tolist()]
    df[f'{ticker}_trend_signal'] = df[mavg_col_list].apply(trend_signal, axis=1)
    df[f'{ticker}_trend_signal_diff'] = df[f'{ticker}_trend_signal'].diff().shift(1)
    df[f'{ticker}_trend_trade'] = np.where(df[f'{ticker}_trend_signal_diff'] != 0, df[f'{ticker}'], np.nan)
    df[f'{ticker}_trend_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_trend_signal_diff']

    ## Ticker Trend Slope Signal and Trade
    df[f'{ticker}_trend_slope_signal'] = df[mavg_slope_col_list].apply(slope_signal, axis=1)
    df[f'{ticker}_trend_slope_signal_diff'] = df[f'{ticker}_trend_slope_signal'].diff().shift(1)
    df[f'{ticker}_trend_slope_trade'] = np.where(df[f'{ticker}_trend_slope_signal_diff'] != 0, df[f'{ticker}'], np.nan)
    df[f'{ticker}_trend_slope_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_trend_slope_signal_diff']

    ## Drop all null values
    df = df[df[f'{ticker}_{mavg_end}_mavg_slope'].notnull()]

    ## Calculate P&L
    df[f'{ticker}_mavg_trend_PnL'] = df[f'{ticker}_trend_signal_diff'] * df[f'{ticker}_trend_trade'] * -1
    df[f'{ticker}_mavg_slope_PnL'] = df[f'{ticker}_trend_slope_signal_diff'] * df[f'{ticker}_trend_slope_trade'] * -1

    ## Calculate Cumulative P&L
    df[f'{ticker}_mavg_trend_PnL_cum'] = df[f'{ticker}_mavg_trend_PnL'].cumsum()
    df[f'{ticker}_mavg_slope_PnL_cum'] = df[f'{ticker}_mavg_slope_PnL'].cumsum()

    return df

