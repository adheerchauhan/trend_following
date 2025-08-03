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
import coinbase_utils as cn
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


def apply_jupyter_fullscreen_css():
    display(HTML('''<style>:root {
    --jp-notebook-max-width: 100% !important; }</style>'''))


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


def get_returns_volatility(df, vol_range_list=[10], close_px_col='BTC-USD_close'):
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


def create_trend_strategy(df, ticker, mavg_start, mavg_end, mavg_stepsize, slope_window, moving_avg_type='simple',
                          price_or_returns_calc='price'):

    df[f'{ticker}_pct_returns'] = df[f'{ticker}_close'].pct_change()

    for window in np.linspace(mavg_start, mavg_end, mavg_stepsize):
        if moving_avg_type == 'simple':
            if price_or_returns_calc == 'price':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_close'].rolling(int(window)).mean()
            elif price_or_returns_calc == 'returns':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_pct_returns'].rolling(int(window)).mean()
        elif moving_avg_type == 'exponential':
            if price_or_returns_calc == 'price':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_close'].ewm(span=window).mean()
            elif price_or_returns_calc == 'returns':
                df[f'{ticker}_{int(window)}_mavg'] = df[f'{ticker}_pct_returns'].ewm(span=window).mean()
        df[f'{ticker}_{int(window)}_mavg_slope'] = calculate_slope(df, column=f'{ticker}_{int(window)}_mavg',
                                                                   periods=slope_window)

    df[f'{ticker}_ribbon_thickness'] = (df[f'{ticker}_{int(mavg_start)}_mavg'] -
                                        df[f'{ticker}_{int(mavg_end)}_mavg']).shift(1)

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
                               upper_atr_multiplier=2, lower_atr_multiplier=2, use_coinbase_data=True):
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    if price_or_returns_calc == 'price':
        # Calculate the Exponential Moving Average (EMA)
        df[f'{ticker}_{rolling_atr_window}_ema_price'] = df['close'].ewm(span=rolling_atr_window,
                                                                         adjust=False).mean()

        # Calculate the True Range (TR) and Average True Range (ATR)
        df[f'{ticker}_high-low'] = df['high'] - df['low']
        df[f'{ticker}_high-close'] = np.abs(df['high'] - df['close'].shift(1))
        df[f'{ticker}_low-close'] = np.abs(df['low'] - df['close'].shift(1))
        df[f'{ticker}_true_range_price'] = df[
            [f'{ticker}_high-low', f'{ticker}_high-close', f'{ticker}_low-close']].max(axis=1)
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[f'{ticker}_true_range_price'].ewm(
            span=rolling_atr_window, adjust=False).mean()

    elif price_or_returns_calc == 'returns':
        # Calculate Percent Returns
        df[f'{ticker}_pct_returns'] = df[f'close'].pct_change()

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
    shift_columns = [
        f'{ticker}_{rolling_atr_window}_atr_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_atr_window}_atr_lower_band_{price_or_returns_calc}'
    ]
    df[shift_columns] = df[shift_columns].shift(1)

    return df


def calculate_donchian_channels(start_date, end_date, ticker, price_or_returns_calc='price',
                                rolling_donchian_window=20, use_coinbase_data=True):
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    if price_or_returns_calc == 'price':
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}'] = (
            df[f'close'].rolling(window=rolling_donchian_window).max())

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'] = (
            df[f'close'].rolling(window=rolling_donchian_window).min())

    elif price_or_returns_calc == 'returns':
        # Calculate Percent Returns
        df[f'{ticker}_pct_returns'] = df[f'close'].pct_change()

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
    shift_columns = [
        f'{ticker}_{rolling_donchian_window}_donchian_middle_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}',
        f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'
    ]
    df[shift_columns] = df[shift_columns].shift(1)

    return df


def generate_rsi_signal(start_date, end_date, ticker, rolling_rsi_period=14, rsi_overbought_threshold=70,
                        rsi_oversold_threshold=30, rsi_mavg_type='exponential', price_or_returns_calc='price',
                        use_coinbase_data=True):
    # Get Close Prices
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
        df = df[['close']].rename(columns={'close': f'{ticker}'})
    else:
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


def generate_volume_oscillator_signal(start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize,
                                      moving_avg_type='simple', use_coinbase_data=True):

    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    df.columns = [f'{ticker}_{x}' for x in df.columns]
    df[f'{ticker}_pct_returns'] = df[f'{ticker}_close'].pct_change()

    for window in np.linspace(fast_mavg, slow_mavg, mavg_stepsize):
        if moving_avg_type == 'simple':
            df[f'{ticker}_volume_{int(window)}_mavg'] = df[f'{ticker}_volume'].rolling(int(window)).mean()
        elif moving_avg_type == 'exponential':
            df[f'{ticker}_volume_{int(window)}_mavg'] = df[f'{ticker}_volume'].ewm(span=window).mean()

    ## Ticker Trend Signal and Trades
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


def calculate_on_balance_volume(start_date, end_date, ticker, use_coinbase_data=True):
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)
        df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    df.columns = [f'{ticker}_{x}' for x in df.columns]

    obv_list = [0]
    # Loop through the data from the second row onwards
    for i in range(1, len(df)):
        if df[f'{ticker}_close'].iloc[i] > df[f'{ticker}_close'].iloc[i - 1]:
            obv_list.append(obv_list[-1] + df[f'{ticker}_volume'].iloc[i])  # Add today's volume
        elif df[f'{ticker}_close'].iloc[i] < df[f'{ticker}_close'].iloc[i - 1]:
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
    df[f'{ticker}_pct_returns'] = df[f'{ticker}_close'].pct_change()
    df[f'{ticker}_obv_strategy_returns'] = df[f'{ticker}_pct_returns'] * df[f'{ticker}_obv_signal']
    df[f'{ticker}_obv_strategy_trades'] = df[f'{ticker}_obv_signal'].diff()

    return df


def calculate_average_directional_index(start_date, end_date, ticker, adx_period, use_coinbase_data):
    ## Convert number of bars to days. ## alpha = 2/(span + 1) for the Exponentially Weighted Average
    ## If alpha = 1/n, span = 2*n - 1
    adx_atr_window = 2 * adx_period - 1

    ## Pull Market Data
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
        df.columns = [f'{ticker}_{x}' for x in df.columns]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = [f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_adjclose',
                      f'{ticker}_volume']

    ## Calculate Directional Move
    df[f'{ticker}_up_move'] = df[f'{ticker}_high'].diff()
    df[f'{ticker}_down_move'] = -df[f'{ticker}_low'].diff()

    plus_dir_move_cond = (df[f'{ticker}_up_move'] > df[f'{ticker}_down_move']) & (df[f'{ticker}_up_move'] > 0)
    minus_dir_move_cond = (df[f'{ticker}_down_move'] > df[f'{ticker}_up_move']) & (df[f'{ticker}_down_move'] > 0)
    df[f'{ticker}_plus_dir_move'] = np.where(plus_dir_move_cond, df[f'{ticker}_up_move'], 0)
    df[f'{ticker}_minus_dir_move'] = np.where(minus_dir_move_cond, df[f'{ticker}_down_move'], 0)

    ## Calculate the True Range (TR) and Average True Range (ATR)
    df[f'{ticker}_high-low'] = df[f'{ticker}_high'] - df[f'{ticker}_low']
    df[f'{ticker}_high-close'] = np.abs(df[f'{ticker}_high'] - df[f'{ticker}_close'].shift(1))
    df[f'{ticker}_low-close'] = np.abs(df[f'{ticker}_low'] - df[f'{ticker}_close'].shift(1))
    df[f'{ticker}_true_range_price'] = df[[f'{ticker}_high-low', f'{ticker}_high-close', f'{ticker}_low-close']].max(
        axis=1)
    df[f'{ticker}_{adx_atr_window}_avg_true_range'] = df[f'{ticker}_true_range_price'].ewm(span=adx_atr_window,
                                                                                           adjust=False).mean()

    ## Calculate the exponentially weighted directional moves
    df[f'{ticker}_plus_dir_move_exp'] = df[f'{ticker}_plus_dir_move'].ewm(span=adx_atr_window, adjust=False).mean()
    df[f'{ticker}_minus_dir_move_exp'] = df[f'{ticker}_minus_dir_move'].ewm(span=adx_atr_window, adjust=False).mean()

    ## Calculate the directional indicator
    df[f'{ticker}_plus_dir_ind'] = 100 * (
                df[f'{ticker}_plus_dir_move_exp'] / df[f'{ticker}_{adx_atr_window}_avg_true_range'])
    df[f'{ticker}_minus_dir_ind'] = 100 * (
                df[f'{ticker}_minus_dir_move_exp'] / df[f'{ticker}_{adx_atr_window}_avg_true_range'])
    df[f'{ticker}_dir_ind'] = 100 * np.abs((df[f'{ticker}_plus_dir_ind'] - df[f'{ticker}_minus_dir_ind'])) / (
                df[f'{ticker}_plus_dir_ind'] + df[f'{ticker}_minus_dir_ind'])
    df[f'{ticker}_avg_dir_ind'] = df[f'{ticker}_dir_ind'].ewm(span=adx_atr_window, adjust=False).mean()

    ## Shift by a day to avoid look-ahead bias
    df[f'{ticker}_avg_dir_ind'] = df[f'{ticker}_avg_dir_ind'].shift(1)

    return df[[f'{ticker}_avg_dir_ind']]


def generate_trend_signal_with_donchian_channel(start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize,
                                                moving_avg_type='exponential', price_or_returns_calc='price',
                                                rolling_donchian_window=20, long_only=False, use_coinbase_data=True):
    # Create Column Names
    donchian_signal_col = f'{ticker}_{rolling_donchian_window}_donchian_signal'
    trend_signal_col = f'{ticker}_trend_signal'
    trend_donchian_signal_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_signal'
    strategy_returns_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_returns'
    strategy_trades_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_trades'

    # Pull Close Prices from Coinbase
    df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                        end_date=end_date, save_to_file=False)
    df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
    df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]

    # Generate Trend Signal
    df_trend = (create_trend_strategy(df, ticker, mavg_start=fast_mavg, mavg_end=slow_mavg, mavg_stepsize=mavg_stepsize,
                                      slope_window=10, moving_avg_type=moving_avg_type,
                                      price_or_returns_calc=price_or_returns_calc)
                .rename(columns={
        f'{ticker}_trend_strategy_returns': f'{ticker}_trend_strategy_returns_{fast_mavg}_{mavg_stepsize}_{slow_mavg}',
        f'{ticker}_trend_strategy_trades': f'{ticker}_trend_strategy_trades_{fast_mavg}_{mavg_stepsize}_{slow_mavg}'}))

    # Generate Donchian Channels
    df_donchian = calculate_donchian_channels(start_date=start_date, end_date=end_date, ticker=ticker,
                                              price_or_returns_calc=price_or_returns_calc,
                                              rolling_donchian_window=rolling_donchian_window,
                                              use_coinbase_data=use_coinbase_data)

    # Donchian Buy signal: Price crosses above upper band
    # Donchian Sell signal: Price crosses below lower band
    df_donchian[f't_1_close'] = df_donchian[f'close'].shift(1)
    t_1_close_col = f't_1_close'
    donchian_upper_band_col = f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}'
    donchian_lower_band_col = f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'
    donchian_middle_band_col = f'{ticker}_{rolling_donchian_window}_donchian_middle_band_{price_or_returns_calc}'
    df_donchian[f'{donchian_upper_band_col}_t_2'] = df_donchian[donchian_upper_band_col].shift(1)
    df_donchian[f'{donchian_lower_band_col}_t_2'] = df_donchian[donchian_lower_band_col].shift(1)
    df_donchian[f'{donchian_middle_band_col}_t_2'] = df_donchian[donchian_middle_band_col].shift(1)
    df_donchian[donchian_signal_col] = np.where(
        (df_donchian[t_1_close_col] > df_donchian[f'{donchian_upper_band_col}_t_2']), 1,
        np.where((df_donchian[t_1_close_col] < df_donchian[f'{donchian_lower_band_col}_t_2']),
                 -1, 0))

    # Merging the Trend and Donchian Dataframes
    donchian_cols = [f'{donchian_upper_band_col}_t_2', f'{donchian_lower_band_col}_t_2',
                     f'{donchian_middle_band_col}_t_2', donchian_signal_col]
    df_trend = pd.merge(df_trend, df_donchian[donchian_cols], left_index=True, right_index=True, how='left')

    # Trend and Donchian Channel Signal
    buy_signal = ((df_trend[donchian_signal_col] == 1) &
                  (df_trend[trend_signal_col] == 1))
    sell_signal = ((df_trend[donchian_signal_col] == -1) &
                   (df_trend[trend_signal_col] == -1))

    # Generate Long Only Signal
    if long_only:
        df_trend[trend_donchian_signal_col] = np.where(buy_signal, 1, 0)
    # Generate Long & Short Signal
    else:
        df_trend[trend_donchian_signal_col] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

    df_trend[strategy_returns_col] = df_trend[trend_donchian_signal_col] * df_trend[f'{ticker}_pct_returns']
    df_trend[strategy_trades_col] = df_trend[trend_donchian_signal_col].diff()

    return df_trend


def get_trend_donchian_signal_for_portfolio(start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                            rolling_donchian_window, long_only=False, price_or_returns_calc='price',
                                            use_coinbase_data=True):

    ## Generate trend signal for all tickers
    trend_list = []
    date_list = cn.coinbase_start_date_by_ticker_dict
    for ticker in ticker_list:
        close_price_col = f'{ticker}_close'
        open_price_col = f'{ticker}_open'
        signal_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_signal'
        # returns_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_returns'
        # trades_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_trades'
        lower_donchian_col = f'{ticker}_{rolling_donchian_window}_donchian_upper_band_{price_or_returns_calc}_t_2'
        upper_donchian_col = f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}_t_2'
        if pd.to_datetime(date_list[ticker]).date() > start_date:
            df_trend = generate_trend_signal_with_donchian_channel(
                start_date=pd.to_datetime(date_list[ticker]).date(), end_date=end_date, ticker=ticker,
                fast_mavg=fast_mavg, slow_mavg=slow_mavg, mavg_stepsize=mavg_stepsize,
                rolling_donchian_window=rolling_donchian_window, price_or_returns_calc=price_or_returns_calc,
                long_only=long_only, use_coinbase_data=use_coinbase_data)
        else:
            df_trend = generate_trend_signal_with_donchian_channel(
                start_date=start_date, end_date=end_date, ticker=ticker, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
                mavg_stepsize=mavg_stepsize, rolling_donchian_window=rolling_donchian_window,
                price_or_returns_calc=price_or_returns_calc, long_only=long_only, use_coinbase_data=use_coinbase_data)
        trend_cols = [close_price_col, open_price_col, signal_col, lower_donchian_col, upper_donchian_col]#returns_col, trades_col]
        df_trend = df_trend[trend_cols]
        trend_list.append(df_trend)

    df_trend = pd.concat(trend_list, axis=1)

    return df_trend

