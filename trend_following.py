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
from IPython.display import display, HTML


def jupyter_interactive_mode():
    display(HTML(
    '<style>'
    '#notebook { padding-top:0px !important; }'
    '.container { width:100% !important; }'
    '.end_space { min-height:0px !important}'
    '</style'
    ))

# Function to pull financial data for a ticker using Yahoo Finance's API
def load_financial_data(start_date, end_date, ticker, print_status=True):
    output_file = f'data_folder/{ticker}-pickle-{start_date}-{end_date}'
    try:
        df = pd.read_pickle(output_file)
        if print_status:
            print(f'File data found...reading {ticker} data')
    except FileNotFoundError:
        if print_status:
            print(f'File not found...downloading the {ticker} data')
        df = yf.download(ticker, start=start_date, end=end_date)
        df.to_pickle(output_file)
    return df


def get_close_prices(start_date, end_date, ticker, close_col='Adj Close', print_status=True):
    if isinstance(ticker, str):
        ticker = [ticker]
    df = load_financial_data(start_date, end_date, ticker, print_status)
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


def create_trend_strategy(df, ticker, mavg_start, mavg_end, mavg_stepsize, slope_window,
                          vol_range_list=[10, 20, 30, 60, 90], moving_avg_type='simple',
                          price_or_returns_calc='price'):

    df[f'{ticker}_pct_returns'] = df[ticker].pct_change()

    for window in np.linspace(mavg_start, mavg_end, mavg_stepsize):
        if moving_avg_type == 'simple':
            if price_or_returns_calc == 'price':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}'].rolling(int(window)).mean()
            elif price_or_returns_calc == 'returns':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_pct_returns'].rolling(int(window)).mean()
        elif moving_avg_type == 'exponential':
            if price_or_returns_calc == 'price':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}'].ewm(span=window).mean()
            elif price_or_returns_calc == 'returns':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_pct_returns'].ewm(span=window).mean()
        df[f'{ticker}_{int(window)}_mavg_slope'] = calculate_slope(df, column=f'{ticker}_{int(window)}_mavg',
                                                                   periods=slope_window)

    df[f'{ticker}_ribbon_thickness'] = (df[f'{ticker}_{int(mavg_start)}_mavg'] -
                                        df[f'{ticker}_{int(mavg_end)}_mavg']).shift(1)
    df = get_returns_volatility(df, vol_range_list=vol_range_list, close_px_col=ticker)

    ## Ticker Trend Signal and Trade
    mavg_col_list = [f'{ticker}_{int(mavg)}_mavg' for mavg in np.linspace(mavg_start, mavg_end, mavg_stepsize).tolist()]
    mavg_slope_col_list = [f'{ticker}_{int(mavg)}_mavg_slope' for mavg in
                           np.linspace(mavg_start, mavg_end, mavg_stepsize).tolist()]
    df[f'{ticker}_trend_signal'] = df[mavg_col_list].apply(trend_signal, axis=1).shift(1)
    df[f'{ticker}_trend_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_trend_signal']
    df[f'{ticker}_trend_strategy_trades'] = df[f'{ticker}_trend_signal'].diff()

    ## Ticker Trend Slope Signal and Trade
    df[f'{ticker}_trend_slope_signal'] = df[mavg_slope_col_list].apply(slope_signal, axis=1).shift(1)
    df[f'{ticker}_trend_slope_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_trend_slope_signal']
    df[f'{ticker}_trend_slope_strategy_trades'] = df[f'{ticker}_trend_slope_signal'].diff()

    ## Drop all null values
    df = df[df[f'{ticker}_{mavg_end}_mavg_slope'].notnull()]

    return df


def calculate_keltner_channels(start_date, end_date, ticker, price_or_returns_calc='price', rolling_atr_window=20,
                               upper_atr_multiplier=2, lower_atr_multiplier=2):
    df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
    df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    if price_or_returns_calc == 'price':
        # Calculate the Exponential Moving Average (EMA)
        df[f'{ticker}_{rolling_atr_window}_ema_price'] = df['adjclose'].ewm(span=rolling_atr_window,
                                                                            adjust=False).mean()

        # Calculate the True Range (TR) and Average True Range (ATR)
        df[f'{ticker}_high-low'] = df['high'] - df['low']
        df[f'{ticker}_high-close'] = np.abs(df['high'] - df['adjclose'].shift(1))
        df[f'{ticker}_low-close'] = np.abs(df['low'] - df['adjclose'].shift(1))
        df[f'{ticker}_true_range_price'] = df[
            [f'{ticker}_high-low', f'{ticker}_high-close', f'{ticker}_low-close']].max(axis=1)
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[f'{ticker}_true_range_price'].ewm(
            span=rolling_atr_window, adjust=False).mean()

    elif price_or_returns_calc == 'returns':
        # Calculate Percent Returns
        df[f'{ticker}_pct_returns'] = df[f'adjclose'].pct_change()

        # Calculate Middle Line as the EMA of returns
        df[f'{ticker}_{rolling_atr_window}_ema_returns'] = df[f'{ticker}_pct_returns'].ewm(span=rolling_atr_window,
                                                                                           adjust=False).mean()

        # Calculate True Range based on absolute returns
        df[f'{ticker}_true_range_returns'] = df[f'{ticker}_{rolling_atr_window}_ema_returns'].abs()

        # Calculate ATR using the EMA of the True Range
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_returns'] = df[f'{ticker}_true_range_returns'].ewm(
            span=rolling_atr_window, adjust=False).mean()

    # Calculate the Upper, Lower and Middle Bands
    df[f'{ticker}_{rolling_atr_window}_atr_middle_band_{price_or_returns_calc}'] = df[
        f'{ticker}_{rolling_atr_window}_ema_{price_or_returns_calc}']
    df[f'{ticker}_{rolling_atr_window}_atr_upper_band_{price_or_returns_calc}'] = (
                df[f'{ticker}_{rolling_atr_window}_ema_{price_or_returns_calc}'] +
                upper_atr_multiplier * df[f'{ticker}_{rolling_atr_window}_avg_true_range_{price_or_returns_calc}'])
    df[f'{ticker}_{rolling_atr_window}_atr_lower_band_{price_or_returns_calc}'] = (
                df[f'{ticker}_{rolling_atr_window}_ema_{price_or_returns_calc}'] -
                lower_atr_multiplier * df[f'{ticker}_{rolling_atr_window}_avg_true_range_{price_or_returns_calc}'])

    # Shift only the Keltner channel metrics to avoid look-ahead bias
    df[[f'{ticker}_{rolling_atr_window}_atr_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_lower_band_{price_or_returns_calc}']] = df[[
        f'{ticker}_{rolling_atr_window}_atr_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_lower_band_{price_or_returns_calc}']].shift(1)

    return df


def calculate_donchian_channels(start_date, end_date, ticker, price_or_returns_calc='price',
                                rolling_donchian_window=20):
    df = load_financial_data(start_date, end_date, ticker, print_status=False)
    df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    if price_or_returns_calc == 'price':
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}'] = (
            df[f'adjclose'].rolling(window=rolling_donchian_window).max())

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'] = (
            df[f'adjclose'].rolling(window=rolling_donchian_window).min())

    elif price_or_returns_calc == 'returns':
        # Calculate Percent Returns
        df[f'{ticker}_pct_returns'] = df[f'adjclose'].pct_change()

        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=rolling_donchian_window).max()

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=rolling_donchian_window).min()

    # Middle of the channel (optional, could be just average of upper and lower)
    df[f'{ticker}_{rolling_donchian_window}_donchian_middle_band_{price_or_returns_calc}'] = (
            (df[f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}'] +
             df[f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}']) / 2)

    # Shift only the Keltner channel metrics to avoid look-ahead bias
    df[[f'{ticker}_{rolling_donchian_window}_donchian_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}']] = df[[
        f'{ticker}_{rolling_donchian_window}_donchian_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}']].shift(1)

    return df


def generate_rsi_signal(start_date, end_date, ticker, rolling_rsi_period=14, rsi_overbought_threshold=70,
                        rsi_oversold_threshold=30,
                        rsi_mavg_type='exponential', price_or_returns_calc='price'):
    # Get Close Prices
    df = get_close_prices(start_date, end_date, ticker, print_status=False)
    df[f'{ticker}_pct_returns'] = df[f'{ticker}'].pct_change()

    if price_or_returns_calc == 'price':
        # Calculate price differences (delta)
        df[f'{ticker}_price_delta'] = df[f'{ticker}'].diff(1)

        # Calculate gains (positive delta) and losses (negative delta)
        df[f'{ticker}_rsi_gain_{price_or_returns_calc}'] = np.where(df[f'{ticker}_price_delta'] > 0,
                                                                    df[f'{ticker}_price_delta'], 0)
        df[f'{ticker}_rsi_loss_{price_or_returns_calc}'] = np.where(df[f'{ticker}_price_delta'] < 0,
                                                                    -df[f'{ticker}_price_delta'], 0)
    elif price_or_returns_calc == 'returns':
        # Calculate gains (positive delta) and losses (negative delta)
        df[f'{ticker}_rsi_gain_{price_or_returns_calc}'] = np.where(df[f'{ticker}_pct_returns'] > 0,
                                                                    df[f'{ticker}_pct_returns'], 0)
        df[f'{ticker}_rsi_loss_{price_or_returns_calc}'] = np.where(df[f'{ticker}_pct_returns'] < 0,
                                                                    -df[f'{ticker}_pct_returns'], 0)

    # Calculate rolling average of gains and losses
    if rsi_mavg_type == 'simple':
        df[f'{ticker}_rsi_avg_gain_{price_or_returns_calc}'] = df[f'{ticker}_rsi_gain_{price_or_returns_calc}'].rolling(
            window=rolling_rsi_period, min_periods=1).mean()
        df[f'{ticker}_rsi_avg_loss_{price_or_returns_calc}'] = df[f'{ticker}_rsi_loss_{price_or_returns_calc}'].rolling(
            window=rolling_rsi_period, min_periods=1).mean()
    elif rsi_mavg_type == 'exponential':
        df[f'{ticker}_rsi_avg_gain_{price_or_returns_calc}'] = df[f'{ticker}_rsi_gain_{price_or_returns_calc}'].ewm(
            alpha=1 / rolling_rsi_period, min_periods=1, adjust=False).mean()
        df[f'{ticker}_rsi_avg_loss_{price_or_returns_calc}'] = df[f'{ticker}_rsi_loss_{price_or_returns_calc}'].ewm(
            alpha=1 / rolling_rsi_period, min_periods=1, adjust=False).mean()

    # Calculate Relative Strength (RS)
    df[f'{ticker}_rs_{rolling_rsi_period}'] = (df[f'{ticker}_rsi_avg_gain_{price_or_returns_calc}'] / df[
        f'{ticker}_rsi_avg_loss_{price_or_returns_calc}'])

    # Calculate RSI and shift by 1 to avoid look-ahead bias
    df[f'{ticker}_rsi_{rolling_rsi_period}'] = (100 - (100 / (1 + df[f'{ticker}_rs_{rolling_rsi_period}']))).shift(1)

    # Generate buy and sell signals based on RSI
    buy_signal = (df[f'{ticker}_rsi_{rolling_rsi_period}'] < rsi_oversold_threshold)
    sell_signal = (df[f'{ticker}_rsi_{rolling_rsi_period}'] > rsi_overbought_threshold)
    df[f'{ticker}_rsi_{rolling_rsi_period}_{rsi_overbought_threshold}_{rsi_oversold_threshold}_signal'] = np.where(
        buy_signal, 1,
        np.where(sell_signal, -1, 0))

    # Calculate RSI Strategy Performance
    df[f'{ticker}_rsi_{rolling_rsi_period}_{rsi_overbought_threshold}_{rsi_oversold_threshold}_strategy_returns'] = (
            df[f'{ticker}_rsi_{rolling_rsi_period}_{rsi_overbought_threshold}_{rsi_oversold_threshold}_signal'] * df[
        f'{ticker}_pct_returns'].fillna(0))
    df[f'{ticker}_rsi_{rolling_rsi_period}_{rsi_overbought_threshold}_{rsi_oversold_threshold}_strategy_trades'] = (
        df[f'{ticker}_rsi_{rolling_rsi_period}_{rsi_overbought_threshold}_{rsi_oversold_threshold}_signal'].diff())

    return df


def generate_volume_oscillator_signal(start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize, moving_avg_type='simple'):

    df = load_financial_data(start_date, end_date, ticker, print_status=False)
    df.columns = ['open','high','low','close','adjclose','volume']
    df.columns = [f'{ticker}_{x}' for x in df.columns]
    df[f'{ticker}_pct_returns'] = df[f'{ticker}_adjclose'].pct_change()

    for window in np.linspace(fast_mavg, slow_mavg, mavg_stepsize):
        if moving_avg_type == 'simple':
            df[f'{ticker}_volume_{int(window)}_mavg'] = df[f'{ticker}_volume'].rolling(int(window)).mean()
        elif moving_avg_type == 'exponential':
            df[f'{ticker}_volume_{int(window)}_mavg'] = df[f'{ticker}_volume'].ewm(span=window).mean()

    ## Ticker Trend Signal and Trade
    mavg_col_list = [f'{ticker}_volume_{int(mavg)}_mavg'
                     for mavg in np.linspace(fast_mavg, slow_mavg, mavg_stepsize).tolist()]
    df[f'{ticker}_volume_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_trend_signal'] = (
        df[mavg_col_list].apply(trend_signal, axis=1).shift(1))
    df[f'{ticker}_volume_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_trend_strategy_returns'] = (
            df[f'{ticker}_pct_returns'] * df[f'{ticker}_volume_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_trend_signal'])
    df[f'{ticker}_volume_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_trend_strategy_trades'] = (
        df[f'{ticker}_volume_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_trend_signal'].diff())

    ## Drop all null values
    df = df[df[f'{ticker}_volume_{slow_mavg}_mavg'].notnull()]

    return df


def calculate_on_balance_volume(start_date, end_date, ticker):
    df = load_financial_data(start_date, end_date, ticker, print_status=False)
    df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    df.columns = [f'{ticker}_{x}' for x in df.columns]

    obv_list = [0]
    # Loop through the data from the second row onwards
    for i in range(1, len(df)):
        if df[f'{ticker}_adjclose'].iloc[i] > df[f'{ticker}_adjclose'].iloc[i - 1]:
            obv_list.append(obv_list[-1] + df[f'{ticker}_volume'].iloc[i])  # Add today's volume
        elif df[f'{ticker}_adjclose'].iloc[i] < df[f'{ticker}_adjclose'].iloc[i - 1]:
            obv_list.append(obv_list[-1] - df[f'{ticker}_volume'].iloc[i])  # Subtract today's volume
        else:
            obv_list.append(obv_list[-1])  # No change in OBV if price remains the same

    df[f'{ticker}_obv'] = obv_list

    obv_buy_signal = (df[f'{ticker}_obv'] > df[f'{ticker}_obv'].shift(1))
    obv_sell_signal = (df[f'{ticker}_obv'] < df[f'{ticker}_obv'].shift(1))
    df[f'{ticker}_obv_signal'] = np.where(obv_buy_signal, 1,
                                          np.where(obv_sell_signal, -1, 0))
    df[f'{ticker}_obv_signal'] = df[f'{ticker}_obv_signal'].shift(1)

    ## Calculate Returns
    df[f'{ticker}_pct_returns'] = df[f'{ticker}_adjclose'].pct_change()
    df[f'{ticker}_obv_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_obv_signal']
    df[f'{ticker}_obv_strategy_trades'] = df[f'{ticker}_obv_signal'].diff()

    return df

