# Import all the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from utils import coinbase_utils as cn
import scipy
from pathlib import Path
from scipy.stats import linregress
from sizing import position_sizing_binary_utils as size_bin, position_sizing_continuous_utils as size_cont
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
            df[f'high'].rolling(window=rolling_donchian_window).max())

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{rolling_donchian_window}_donchian_lower_band_{price_or_returns_calc}'] = (
            df[f'low'].rolling(window=rolling_donchian_window).min())

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


def calc_ribbon_slope(row, ticker, fast_mavg, slow_mavg, mavg_stepsize):
    x = np.linspace(slow_mavg, fast_mavg, mavg_stepsize)
    y = row.values
    slope, _, _, _, _ = linregress(x, y)
    return slope


def pct_rank(x, window=250):
    return x.rank(pct=True)


def create_trend_strategy_log_space(df, ticker, mavg_start, mavg_end, mavg_stepsize, mavg_z_score_window=252):
    # ---- constants ----
    windows = np.geomspace(mavg_start, mavg_end, mavg_stepsize).round().astype(int)
    windows = np.unique(windows)
    x = np.log(windows[::-1])
    xm = x - x.mean()
    varx = (xm ** 2).sum()

    # ---- compute MAs (vectorised) ----
    df[f'{ticker}_close_log'] = np.log(df[f'{ticker}_close'])
    for w in windows:
        df[f'{ticker}_{w}_ema'] = df[f'{ticker}_close_log'].ewm(span=w, adjust=False).mean()

    mavg_mat = df[[f'{ticker}_{w}_ema' for w in windows]].to_numpy()

    # ---- slope (vectorised) ----
    slope = mavg_mat.dot(xm) / varx  # ndarray (T,)
    slope = pd.Series(slope, index=df.index).shift(1)  # lag to avoid look-ahead

    # ---- z-score & rank ----
    z = ((slope - slope.rolling(mavg_z_score_window, min_periods=mavg_z_score_window).mean()) /
         slope.rolling(mavg_z_score_window, min_periods=mavg_z_score_window).std())

    # Optional Tail Cap
    z = z.clip(-4, 4)

    # Calculate the Percentile Rank based on CDF
    rank = scipy.stats.norm.cdf(z) - 0.5  # centered 0 ↔ ±0.5

    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    df[trend_continuous_signal_col] = slope
    df[trend_continuous_signal_rank_col] = rank

    return df


def calculate_donchian_channel_dual_window(start_date, end_date, ticker, price_or_returns_calc='price',
                                           entry_rolling_donchian_window=20, exit_rolling_donchian_window=20,
                                           use_coinbase_data=True, use_saved_files=True,
                                           saved_file_end_date='2025-07-31'):
    REPO_ROOT = Path.cwd().parents[1]
    PKL_DIR = REPO_ROOT / "data_folder" / "coinbase_historical_price_folder"
    if use_coinbase_data:
        if use_saved_files:
            date_list = cn.coinbase_start_date_by_ticker_dict
            file_end_date = pd.Timestamp(saved_file_end_date).date()
            filename = f"{ticker}-pickle-{pd.Timestamp(date_list[ticker]).strftime('%Y-%m-%d')}-{file_end_date.strftime('%Y-%m-%d')}"
            # output_file = f'coinbase_historical_price_folder/{filename}'
            output_file = PKL_DIR / filename
            df = pd.read_pickle(output_file)
            date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
            df = df[date_cond]
        else:
            # df = cn.get_coinbase_ohlc_data(ticker=ticker)
            df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True,
                                                                start_date=start_date,
                                                                end_date=end_date, save_to_file=False)
            # df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
            # 1) Ensure DatetimeIndex (tz-naive) and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_localize(None)
            else:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            df.index.name = "date"
            df = df.sort_index()

            # 2) Promote bounds to Timestamps (normalized) to match the index
            start_ts = pd.Timestamp(start_date).normalize()
            end_ts = pd.Timestamp(end_date).normalize()

            # 3) Slice by label using .loc (don’t use get_level_values unless MultiIndex)
            df = df.loc[start_ts:end_ts]
    else:
        df = load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']

    if price_or_returns_calc == 'price':
        ## Entry Channel
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'] = (
            df[f'high'].rolling(window=entry_rolling_donchian_window).max())

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}'] = (
            df[f'low'].rolling(window=entry_rolling_donchian_window).min())

        ## Exit Channel
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'] = (
            df[f'high'].rolling(window=exit_rolling_donchian_window).max())

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'] = (
            df[f'low'].rolling(window=exit_rolling_donchian_window).min())

    elif price_or_returns_calc == 'returns':
        # Calculate Percent Returns
        df[f'{ticker}_pct_returns'] = df[f'close'].pct_change()

        ## Entry Channel
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=entry_rolling_donchian_window).max()

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=entry_rolling_donchian_window).min()

        ## Exit Channel
        # Rolling maximum of returns (upper channel)
        df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=exit_rolling_donchian_window).max()

        # Rolling minimum of returns (lower channel)
        df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'] = df[
            f'{ticker}_pct_returns'].rolling(window=exit_rolling_donchian_window).min()

    # Middle of the channel (optional, could be just average of upper and lower)
    # Entry Middle Band
    df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band_{price_or_returns_calc}'] = (
        (df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'] +
         df[f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}']) / 2)

    # Exit Middle Band
    df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band_{price_or_returns_calc}'] = (
        (df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'] +
         df[f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}']) / 2)

    # Shift only the Keltner channel metrics to avoid look-ahead bias
    shift_columns = [
        f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band_{price_or_returns_calc}',
        f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}',
        f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}',
        f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band_{price_or_returns_calc}',
        f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}',
        f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'
    ]
    df[shift_columns] = df[shift_columns].shift(1)

    return df


def calculate_rolling_r2(df, ticker, t_1_close_price_col, rolling_r2_window=30, lower_r_sqr_limit=0.2,
                         upper_r_sqr_limit=0.8, r2_smooth_window=3):
    log_price_col = f'{ticker}_t_1_close_price_log'
    df[log_price_col] = np.log(df[t_1_close_price_col])

    ## Define the variables
    y = df[log_price_col]
    x = np.arange(len(y), dtype=float)  # Time

    ## Compute rolling sums for rolling R2 calculation
    x_sum = pd.Series(x, y.index).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    y_sum = y.rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    x_sqr = pd.Series(x ** 2, y.index).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    y_sqr = (y ** 2).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()
    xy_sum = pd.Series(x, y.index).mul(y).rolling(rolling_r2_window, min_periods=rolling_r2_window).sum()

    ## Calculate the R squared
    n = rolling_r2_window
    numerator = n * xy_sum - x_sum * y_sum
    denominator = np.sqrt((n * x_sqr) - (x_sum ** 2)) * np.sqrt((n * y_sqr) - (y_sum ** 2))
    df[f'{ticker}_rolling_r_sqr'] = (numerator / denominator) ** 2

    ## Normalize the R Squared centered around 0.5 where values below the lower limit are
    ## clipped to 0 and values above the upper limit are clipped to 1
    df[f'{ticker}_rolling_r_sqr'] = np.clip(
        (df[f'{ticker}_rolling_r_sqr'] - lower_r_sqr_limit) / (upper_r_sqr_limit - lower_r_sqr_limit),
        0, 1)

    ## Smoothing the Rolling R Squared Signal
    if r2_smooth_window >= 1:
        df[f'{ticker}_rolling_r_sqr'] = df[f'{ticker}_rolling_r_sqr'].ewm(span=r2_smooth_window, adjust=False).mean()

    ## Adding Convex Scaling to further reduce the signal during low trends and amplify during high trends
    # df[f'{ticker}_rolling_r_sqr'] = df[f'{ticker}_rolling_r_sqr'].clip(0, 1) ** 1.5

    return df


def generate_vol_of_vol_signal_log_space(df, ticker, t_1_close_price_col, log_std_window=14,
                                         coef_of_variation_window=30, vol_of_vol_z_score_window=252,
                                         vol_of_vol_p_min=0.6):

    log_returns_col = f'{ticker}_t_1_log_returns'
    realized_log_returns_vol = f'{ticker}_ann_log_volatility'
    df[log_returns_col] = np.log(df[t_1_close_price_col] / df[t_1_close_price_col].shift(1))
    eps = 1e-12

    ## Realized Volatility of Log Returns
    df[realized_log_returns_vol] = df[log_returns_col].ewm(span=log_std_window, adjust=False,
                                                           min_periods=log_std_window).std() * np.sqrt(365)

    ## Coefficient of Variation in Volatility
    df[f'{ticker}_coef_variation_vol'] = (
            df[realized_log_returns_vol].rolling(coef_of_variation_window,
                                                 min_periods=coef_of_variation_window).std() /
            df[realized_log_returns_vol].rolling(coef_of_variation_window,
                                                 min_periods=coef_of_variation_window).mean().clip(lower=eps))

    ## Calculate Robust Z-Score of the Coefficient of Variation
    cov_rolling_median = df[f'{ticker}_coef_variation_vol'].rolling(vol_of_vol_z_score_window,
                                                                    min_periods=vol_of_vol_z_score_window).median()
    df[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median'] = cov_rolling_median
    cov_rolling_mad = ((df[f'{ticker}_coef_variation_vol'] -
                        df[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median']).abs()
                       .rolling(vol_of_vol_z_score_window, min_periods=vol_of_vol_z_score_window).median())
    df[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median_abs_dev'] = cov_rolling_mad
    df[f'{ticker}_vol_of_vol_robust_z_score'] = (
            (df[f'{ticker}_coef_variation_vol'] - df[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median']) /
            (1.4826 * df[f'{ticker}_cov_vol_rolling_{vol_of_vol_z_score_window}_median_abs_dev']).clip(lower=eps)
    )
    df[f'{ticker}_vol_of_vol_robust_z_score'] = (df[f'{ticker}_vol_of_vol_robust_z_score']
                                                 .replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=-3, upper=3))

    ## Create Vol of Vol Thresholds
    ## z0 represents low volatility and z1 represents high volatility
    ## The vol of vol penalty will go from 1 to p_min where 1 represents no penalty
    z0, z1 = 0.5, 1.5                # z_vov below 0.5 → no penalty; above 1.5 → max raw penalty
    p_min = vol_of_vol_p_min         # even at max raw penalty, keep at least 60% exposure

    ## Compute a 0..1 raw penalty that rises from 0→1 as z_vov goes z0→z1
    df[f'{ticker}_vol_of_vol_signal_raw'] = (df[f'{ticker}_vol_of_vol_robust_z_score'] - z0) / max((z1 - z0), eps)

    ## Clip the signal to [0, 1]
    df[f'{ticker}_vol_of_vol_signal_raw'] = df[f'{ticker}_vol_of_vol_signal_raw'].clip(0, 1)

    ## Invert so that the raw penalty goes from 1 to 0 instead of 0 to 1
    df[f'{ticker}_vol_of_vol_penalty'] = 1 - df[f'{ticker}_vol_of_vol_signal_raw']

    ## Floor the penalty at p_min
    df[f'{ticker}_vol_of_vol_penalty'] = df[f'{ticker}_vol_of_vol_penalty'].clip(lower=p_min, upper=1)

    return df


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


## Original Signal
def generate_trend_signal_with_donchian_channel_continuous(start_date, end_date, ticker, fast_mavg, slow_mavg,
                                                           mavg_stepsize, mavg_z_score_window,
                                                           entry_rolling_donchian_window,
                                                           exit_rolling_donchian_window, use_donchian_exit_gate,
                                                           ma_crossover_signal_weight, donchian_signal_weight,
                                                           weighted_signal_ewm_window,
                                                           use_activation=True, tanh_activation_constant_dict=None,
                                                           moving_avg_type='exponential', price_or_returns_calc='price',
                                                           long_only=False, use_coinbase_data=True,
                                                           use_saved_files=True, saved_file_end_date='2025-07-31'):
    # Pull Close Prices from Coinbase
    REPO_ROOT = Path.cwd().parents[1]
    PKL_DIR = REPO_ROOT / "data_folder" / "coinbase_historical_price_folder"
    date_list = cn.coinbase_start_date_by_ticker_dict
    if use_saved_files:
        file_end_date = pd.Timestamp(saved_file_end_date).date()
        filename = f"{ticker}-pickle-{pd.Timestamp(date_list[ticker]).strftime('%Y-%m-%d')}-{file_end_date.strftime('%Y-%m-%d')}"
        # output_file = f'coinbase_historical_price_folder/{filename}'
        output_file = PKL_DIR / filename
        df = pd.read_pickle(output_file)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
        df = df[date_cond]
    else:
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
        df = df[date_cond]

    # Create Column Names
    donchian_binary_signal_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_binary_signal'
    donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
    donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
    trend_binary_signal_col = f'{ticker}_trend_signal'
    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    final_binary_signal_col = f'{ticker}_final_binary_signal'
    final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
    final_signal_col = f'{ticker}_final_signal'

    ## Generate Trend Signal in Log Space
    df_trend = create_trend_strategy_log_space(df, ticker, mavg_start=fast_mavg, mavg_end=slow_mavg,
                                               mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window)

    ## Generate Donchian Channels
    # Donchian Buy signal: Price crosses above upper band
    # Donchian Sell signal: Price crosses below lower band
    df_donchian = calculate_donchian_channel_dual_window(start_date=start_date, end_date=end_date, ticker=ticker,
                                                         price_or_returns_calc=price_or_returns_calc,
                                                         entry_rolling_donchian_window=entry_rolling_donchian_window,
                                                         exit_rolling_donchian_window=exit_rolling_donchian_window,
                                                         use_coinbase_data=use_coinbase_data,
                                                         use_saved_files=use_saved_files,
                                                         saved_file_end_date=saved_file_end_date)

    t_1_close_col = f't_1_close'
    df_donchian[t_1_close_col] = df_donchian[f'close'].shift(1)
    donchian_entry_upper_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'
    donchian_entry_lower_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}'
    donchian_entry_middle_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band_{price_or_returns_calc}'
    donchian_exit_upper_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'
    donchian_exit_lower_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'
    donchian_exit_middle_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band_{price_or_returns_calc}'
    shift_cols = [donchian_entry_upper_band_col, donchian_entry_lower_band_col, donchian_entry_middle_band_col,
                  donchian_exit_upper_band_col, donchian_exit_lower_band_col, donchian_exit_middle_band_col]
    for col in shift_cols:
        df_donchian[f'{col}_t_2'] = df_donchian[col].shift(1)

    # Donchian Continuous Signal
    df_donchian[donchian_continuous_signal_col] = (
                (df_donchian[t_1_close_col] - df_donchian[f'{donchian_entry_middle_band_col}_t_2']) /
                (df_donchian[f'{donchian_entry_upper_band_col}_t_2'] - df_donchian[
                    f'{donchian_entry_lower_band_col}_t_2']))

    ## Calculate Donchian Channel Rank
    ## Adjust the percentage ranks by 0.5 as without, the ranks go from 0 to 1. Recentering the function by giving it a steeper
    ## slope near the origin takes into account even little information
    df_donchian[donchian_continuous_signal_rank_col] = pct_rank(df_donchian[donchian_continuous_signal_col]) - 0.5

    # Donchian Binary Signal
    gate_long_condition = df_donchian[t_1_close_col] >= df_donchian[f'{donchian_exit_lower_band_col}_t_2']
    gate_short_condition = df_donchian[t_1_close_col] <= df_donchian[f'{donchian_exit_upper_band_col}_t_2']
    # sign of *entry* score decides direction
    entry_sign = np.sign(df_donchian[donchian_continuous_signal_col])
    # treat exact zero as "flat but allowed" (gate=1) so ranking not wiped out
    entry_sign = np.where(entry_sign == 0, 1, entry_sign)  # default to long-side keep
    df_donchian[donchian_binary_signal_col] = np.where(
        entry_sign > 0, gate_long_condition, gate_short_condition).astype(float)

    # Merging the Trend and Donchian Dataframes
    donchian_cols = [f'{donchian_entry_upper_band_col}_t_2', f'{donchian_entry_lower_band_col}_t_2',
                     f'{donchian_entry_middle_band_col}_t_2',
                     f'{donchian_exit_upper_band_col}_t_2', f'{donchian_exit_lower_band_col}_t_2',
                     f'{donchian_exit_middle_band_col}_t_2',
                     donchian_binary_signal_col, donchian_continuous_signal_col, donchian_continuous_signal_rank_col]
    df_trend = pd.merge(df_trend, df_donchian[donchian_cols], left_index=True, right_index=True, how='left')

    ## Trend and Donchian Channel Signal
    # Calculate the exponential weighted average of the ranked signals to remove short-term flip-flops (whiplash)
    df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]] = (
        df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]].ewm(
            span=weighted_signal_ewm_window, adjust=False).mean())

    # Weighted Sum of Rank Columns
    df_trend[final_weighted_additive_signal_col] = (
                ma_crossover_signal_weight * df_trend[trend_continuous_signal_rank_col] +
                donchian_signal_weight * df_trend[donchian_continuous_signal_rank_col])

    # Activation Scaled Signal
    if use_activation:
        final_signal_unscaled_95th_percentile = np.abs(df_trend[final_weighted_additive_signal_col]).quantile(0.95)
        if tanh_activation_constant_dict:
            k = tanh_activation_constant_dict[ticker]
            df_trend[f'{ticker}_activation'] = np.tanh(df_trend[final_weighted_additive_signal_col] * k)
        else:
            if (final_signal_unscaled_95th_percentile == 0):  # | (final_signal_unscaled_95th_percentile.isnan()):
                k = 1.0
            else:
                k = np.arctanh(0.9) / final_signal_unscaled_95th_percentile
            df_trend[f'{ticker}_activation'] = np.tanh(df_trend[final_weighted_additive_signal_col] * k)
    else:
        df_trend[f'{ticker}_activation'] = df_trend[final_weighted_additive_signal_col]

    # Apply Binary Gate
    if use_donchian_exit_gate:
        df_trend[f'{ticker}_activation'] = df_trend[f'{ticker}_activation'] * df_trend[donchian_binary_signal_col]

    ## Long-Only Filter
    df_trend[final_signal_col] = np.where(long_only, np.maximum(0, df_trend[f'{ticker}_activation']),
                                          df_trend[f'{ticker}_activation'])

    return df_trend


def get_trend_donchian_signal_for_portfolio_continuous(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
        mavg_z_score_window, entry_rolling_donchian_window,
        exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight,
        weighted_signal_ewm_window,
        use_activation=True, tanh_activation_constant_dict=None,
        moving_avg_type='exponential', long_only=False,
        price_or_returns_calc='price',
        use_coinbase_data=True, use_saved_files=True,
        saved_file_end_date='2025-07-31'):

    ## Generate trend signal for all tickers
    trend_list = []
    date_list = cn.coinbase_start_date_by_ticker_dict

    for ticker in ticker_list:
        # Create Column Names
        donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
        donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
        trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
        trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
        final_signal_col = f'{ticker}_final_signal'
        close_price_col = f'{ticker}_close'
        open_price_col = f'{ticker}_open'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'

        if pd.to_datetime(date_list[ticker]).date() > start_date:
            run_date = pd.to_datetime(date_list[ticker]).date()
        else:
            run_date = start_date

        df_trend = generate_trend_signal_with_donchian_channel_continuous(
            start_date=run_date, end_date=end_date, ticker=ticker,
            fast_mavg=fast_mavg, slow_mavg=slow_mavg, mavg_stepsize=mavg_stepsize,
            mavg_z_score_window=mavg_z_score_window,
            entry_rolling_donchian_window=entry_rolling_donchian_window,
            exit_rolling_donchian_window=exit_rolling_donchian_window, use_donchian_exit_gate=use_donchian_exit_gate,
            ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
            weighted_signal_ewm_window=weighted_signal_ewm_window,
            use_activation=use_activation, tanh_activation_constant_dict=tanh_activation_constant_dict,
            moving_avg_type=moving_avg_type, price_or_returns_calc=price_or_returns_calc, long_only=long_only,
            use_coinbase_data=use_coinbase_data, use_saved_files=use_saved_files,
            saved_file_end_date=saved_file_end_date)

        trend_cols = [close_price_col, open_price_col, trend_continuous_signal_col, trend_continuous_signal_rank_col,
                      final_weighted_additive_signal_col, final_signal_col]
        df_trend = df_trend[trend_cols]
        trend_list.append(df_trend)

    df_trend = pd.concat(trend_list, axis=1)

    return df_trend


def apply_target_volatility_position_sizing_continuous_strategy(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window=4,
        use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential',
        long_only=False, price_or_returns_calc='price', initial_capital=15000, rolling_cov_window=20,
        volatility_window=20, stop_loss_strategy='Chandelier', rolling_atr_window=20, atr_multiplier=0.5,
        highest_high_window=56, transaction_cost_est=0.001, passive_trade_rate=0.05,
        notional_threshold_pct=0.05, min_trade_notional_abs=10,
        cooldown_counter_threshold=3, use_coinbase_data=True, use_saved_files=True,
        saved_file_end_date='2025-07-31', rolling_sharpe_window=50, cash_buffer_percentage=0.10,
        annualized_target_volatility=0.20, annual_trading_days=365, use_specific_start_date=False,
        signal_start_date=None):

    ## Check if data is available for all the tickers
    date_list = cn.coinbase_start_date_by_ticker_dict
    ticker_list = [ticker for ticker in ticker_list if pd.Timestamp(date_list[ticker]).date() < end_date]

    print('Generating Moving Average Ribbon Signal!!')
    ## Generate Trend Signal for all tickers
    df_trend = get_trend_donchian_signal_for_portfolio_continuous(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
        mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
        entry_rolling_donchian_window=entry_rolling_donchian_window,
        exit_rolling_donchian_window=exit_rolling_donchian_window,  use_donchian_exit_gate=use_donchian_exit_gate,
        ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
        weighted_signal_ewm_window=weighted_signal_ewm_window, use_activation=use_activation,
        tanh_activation_constant_dict=tanh_activation_constant_dict, moving_avg_type=moving_avg_type,
        long_only=long_only, price_or_returns_calc=price_or_returns_calc, use_coinbase_data=use_coinbase_data,
        use_saved_files=use_saved_files, saved_file_end_date=saved_file_end_date)

    print('Generating Volatility Adjusted Trend Signal!!')
    ## Get Volatility Adjusted Trend Signal
    df_signal = size_cont.get_volatility_adjusted_trend_signal_continuous(df_trend, ticker_list, volatility_window,
                                                                          annual_trading_days)

    print('Getting Average True Range for Stop Loss Calculation!!')
    ## Get Average True Range for Stop Loss Calculation
    df_atr = size_cont.get_average_true_range_portfolio(start_date=start_date, end_date=end_date,
                                                        ticker_list=ticker_list, rolling_atr_window=rolling_atr_window,
                                                        highest_high_window=highest_high_window,
                                                        price_or_returns_calc='price',
                                                        use_coinbase_data=use_coinbase_data,
                                                        use_saved_files=use_saved_files,
                                                        saved_file_end_date=saved_file_end_date)
    df_signal = pd.merge(df_signal, df_atr, left_index=True, right_index=True, how='left')

    print('Calculating Volatility Targeted Position Size and Cash Management!!')
    ## Get Target Volatility Position Sizing and Run Cash Management
    df = size_cont.get_target_volatility_daily_portfolio_positions(
        df_signal, ticker_list=ticker_list, initial_capital=initial_capital, rolling_cov_window=rolling_cov_window,
        stop_loss_strategy=stop_loss_strategy, rolling_atr_window=rolling_atr_window, atr_multiplier=atr_multiplier,
        highest_high_window=highest_high_window, cash_buffer_percentage=cash_buffer_percentage,
        annualized_target_volatility=annualized_target_volatility,
        transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate,
        notional_threshold_pct=notional_threshold_pct, min_trade_notional_abs=min_trade_notional_abs,
        cooldown_counter_threshold=cooldown_counter_threshold,
        annual_trading_days=annual_trading_days, use_specific_start_date=use_specific_start_date,
        signal_start_date=signal_start_date)

    print('Calculating Portfolio Performance!!')
    ## Calculate Portfolio Performance
    df = size_bin.calculate_portfolio_returns(df, rolling_sharpe_window)

    return df


## Original Signal
def generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr(
        start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential',
        price_or_returns_calc='price', long_only=False, use_coinbase_data=True, use_saved_files=True,
        saved_file_end_date='2025-07-31'):

    # Pull Close Prices from Coinbase
    REPO_ROOT = Path.cwd().parents[1]
    PKL_DIR = REPO_ROOT / "data_folder" / "coinbase_historical_price_folder"
    date_list = cn.coinbase_start_date_by_ticker_dict
    if use_saved_files:
        file_end_date = pd.Timestamp(saved_file_end_date).date()
        filename = f"{ticker}-pickle-{pd.Timestamp(date_list[ticker]).strftime('%Y-%m-%d')}-{file_end_date.strftime('%Y-%m-%d')}"
        # output_file = f'coinbase_historical_price_folder/{filename}'
        output_file = PKL_DIR / filename
        df = pd.read_pickle(output_file)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
        df = df[date_cond]
    else:
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
        df = df[date_cond]

    # Create Column Names
    donchian_binary_signal_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_binary_signal'
    donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
    donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
    trend_binary_signal_col = f'{ticker}_trend_signal'
    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    final_binary_signal_col = f'{ticker}_final_binary_signal'
    final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
    final_signal_col = f'{ticker}_final_signal'

    ## Generate Trend Signal in Log Space
    df_trend = create_trend_strategy_log_space(df, ticker, mavg_start=fast_mavg, mavg_end=slow_mavg,
                                               mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window)

    ## Generate Donchian Channels
    # Donchian Buy signal: Price crosses above upper band
    # Donchian Sell signal: Price crosses below lower band
    df_donchian = calculate_donchian_channel_dual_window(start_date=start_date, end_date=end_date, ticker=ticker,
                                                         price_or_returns_calc=price_or_returns_calc,
                                                         entry_rolling_donchian_window=entry_rolling_donchian_window,
                                                         exit_rolling_donchian_window=exit_rolling_donchian_window,
                                                         use_coinbase_data=use_coinbase_data,
                                                         use_saved_files=use_saved_files,
                                                         saved_file_end_date=saved_file_end_date)

    t_1_close_col = f't_1_close'
    df_donchian[t_1_close_col] = df_donchian[f'close'].shift(1)
    donchian_entry_upper_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'
    donchian_entry_lower_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}'
    donchian_entry_middle_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band_{price_or_returns_calc}'
    donchian_exit_upper_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'
    donchian_exit_lower_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'
    donchian_exit_middle_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band_{price_or_returns_calc}'
    shift_cols = [donchian_entry_upper_band_col, donchian_entry_lower_band_col, donchian_entry_middle_band_col,
                  donchian_exit_upper_band_col, donchian_exit_lower_band_col, donchian_exit_middle_band_col]
    for col in shift_cols:
        df_donchian[f'{col}_t_2'] = df_donchian[col].shift(1)

    # Donchian Continuous Signal
    df_donchian[donchian_continuous_signal_col] = (
                (df_donchian[t_1_close_col] - df_donchian[f'{donchian_entry_middle_band_col}_t_2']) /
                (df_donchian[f'{donchian_entry_upper_band_col}_t_2'] - df_donchian[
                    f'{donchian_entry_lower_band_col}_t_2']))

    ## Calculate Donchian Channel Rank
    ## Adjust the percentage ranks by 0.5 as without, the ranks go from 0 to 1. Recentering the function
    ## by giving it a steeper slope near the origin takes into account even little information
    df_donchian[donchian_continuous_signal_rank_col] = pct_rank(df_donchian[donchian_continuous_signal_col]) - 0.5

    # Donchian Binary Signal
    gate_long_condition = df_donchian[t_1_close_col] >= df_donchian[f'{donchian_exit_lower_band_col}_t_2']
    gate_short_condition = df_donchian[t_1_close_col] <= df_donchian[f'{donchian_exit_upper_band_col}_t_2']
    # sign of *entry* score decides direction
    entry_sign = np.sign(df_donchian[donchian_continuous_signal_col])
    # treat exact zero as "flat but allowed" (gate=1) so ranking not wiped out
    entry_sign = np.where(entry_sign == 0, 1, entry_sign)  # default to long-side keep
    df_donchian[donchian_binary_signal_col] = np.where(
        entry_sign > 0, gate_long_condition, gate_short_condition).astype(float)

    # Merging the Trend and Donchian Dataframes
    donchian_cols = [t_1_close_col, f'{donchian_entry_upper_band_col}_t_2', f'{donchian_entry_lower_band_col}_t_2',
                     f'{donchian_entry_middle_band_col}_t_2',
                     f'{donchian_exit_upper_band_col}_t_2', f'{donchian_exit_lower_band_col}_t_2',
                     f'{donchian_exit_middle_band_col}_t_2',
                     donchian_binary_signal_col, donchian_continuous_signal_col, donchian_continuous_signal_rank_col]
    df_trend = pd.merge(df_trend, df_donchian[donchian_cols], left_index=True, right_index=True, how='left')

    ## Trend and Donchian Channel Signal
    # Calculate the exponential weighted average of the ranked signals to remove short-term flip flops (whiplash)
    df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]] = (
        df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]].ewm(
            span=weighted_signal_ewm_window, adjust=False).mean())

    # Weighted Sum of Rank Columns
    df_trend[final_weighted_additive_signal_col] = (
                ma_crossover_signal_weight * df_trend[trend_continuous_signal_rank_col] +
                donchian_signal_weight * df_trend[donchian_continuous_signal_rank_col])

    # Apply Binary Gate
    if use_donchian_exit_gate:
        df_trend[final_weighted_additive_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[
            donchian_binary_signal_col]

    ## Calculate Rolling R Squared Signal
    df_trend = calculate_rolling_r2(df_trend, ticker=ticker, t_1_close_price_col=t_1_close_col,
                                    rolling_r2_window=rolling_r2_window,
                                    lower_r_sqr_limit=lower_r_sqr_limit, upper_r_sqr_limit=upper_r_sqr_limit,
                                    r2_smooth_window=r2_smooth_window)

    ## Apply Regime Filters
    # Introduce a 3-day confirmation period for Rolling R Squared Signal
    if r2_confirm_days >= 1:
        df_trend[f'{ticker}_r2_enable'] = ((df_trend[f'{ticker}_rolling_r_sqr'] > 0.5)
                                           .rolling(r2_confirm_days, min_periods=r2_confirm_days).min().fillna(
            0.0).astype(float))
        df_trend[final_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[
            f'{ticker}_rolling_r_sqr'] * df_trend[f'{ticker}_r2_enable']
    else:
        df_trend[final_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[f'{ticker}_rolling_r_sqr']

    ## Long-Only Filter
    df_trend[final_signal_col] = np.where(long_only, np.maximum(0, df_trend[final_signal_col]),
                                          df_trend[final_signal_col])

    return df_trend


def get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential', long_only=False,
        price_or_returns_calc='price', use_coinbase_data=True, use_saved_files=True, saved_file_end_date='2025-07-31'):

    ## Generate trend signal for all tickers
    trend_list = []
    date_list = cn.coinbase_start_date_by_ticker_dict

    for ticker in ticker_list:
        # Create Column Names
        donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
        donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
        trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
        trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
        final_signal_col = f'{ticker}_final_signal'
        close_price_col = f'{ticker}_close'
        open_price_col = f'{ticker}_open'
        rolling_r2_col = f'{ticker}_rolling_r_sqr'
        # rolling_r2_enable_col = f'{ticker}_r2_enable'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'

        if pd.to_datetime(date_list[ticker]).date() > start_date:
            run_date = pd.to_datetime(date_list[ticker]).date()
        else:
            run_date = start_date

        df_trend = generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr(
            start_date=start_date, end_date=end_date, ticker=ticker, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
            mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
            entry_rolling_donchian_window=entry_rolling_donchian_window,
            exit_rolling_donchian_window=exit_rolling_donchian_window, use_donchian_exit_gate=use_donchian_exit_gate,
            ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
            weighted_signal_ewm_window=weighted_signal_ewm_window,
            rolling_r2_window=rolling_r2_window, lower_r_sqr_limit=lower_r_sqr_limit,
            upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window, r2_confirm_days=r2_confirm_days,
            use_activation=use_activation, tanh_activation_constant_dict=tanh_activation_constant_dict,
            moving_avg_type=moving_avg_type, price_or_returns_calc=price_or_returns_calc, long_only=long_only,
            use_coinbase_data=use_coinbase_data, use_saved_files=use_saved_files,
            saved_file_end_date=saved_file_end_date)

        trend_cols = [close_price_col, open_price_col, trend_continuous_signal_col, trend_continuous_signal_rank_col,
                      final_weighted_additive_signal_col,
                      rolling_r2_col, final_signal_col]
        df_trend = df_trend[trend_cols]
        trend_list.append(df_trend)

    df_trend = pd.concat(trend_list, axis=1)

    return df_trend


def apply_target_volatility_position_sizing_continuous_strategy_with_rolling_r_sqr(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window=4,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential', long_only=False,
        price_or_returns_calc='price', initial_capital=15000, rolling_cov_window=20, volatility_window=20,
        stop_loss_strategy='Chandelier', rolling_atr_window=20, atr_multiplier=0.5, highest_high_window=56,
        transaction_cost_est=0.001, passive_trade_rate=0.05,
        notional_threshold_pct=0.05, min_trade_notional_abs=10, cooldown_counter_threshold=3, use_coinbase_data=True,
        use_saved_files=True, saved_file_end_date='2025-07-31', rolling_sharpe_window=50, cash_buffer_percentage=0.10,
        annualized_target_volatility=0.20, annual_trading_days=365,
        use_specific_start_date=False, signal_start_date=None):

    ## Check if data is available for all the tickers
    date_list = cn.coinbase_start_date_by_ticker_dict
    ticker_list = [ticker for ticker in ticker_list if pd.Timestamp(date_list[ticker]).date() < end_date]

    print('Generating Moving Average Ribbon Signal!!')
    ## Generate Trend Signal for all tickers
    df_trend = get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
        mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
        entry_rolling_donchian_window=entry_rolling_donchian_window,
        exit_rolling_donchian_window=exit_rolling_donchian_window, use_donchian_exit_gate=use_donchian_exit_gate,
        ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
        weighted_signal_ewm_window=weighted_signal_ewm_window, rolling_r2_window=rolling_r2_window,
        lower_r_sqr_limit=lower_r_sqr_limit, upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window,
        r2_confirm_days=r2_confirm_days, use_activation=use_activation,
        tanh_activation_constant_dict=tanh_activation_constant_dict, moving_avg_type=moving_avg_type,
        long_only=long_only, price_or_returns_calc=price_or_returns_calc, use_coinbase_data=use_coinbase_data,
        use_saved_files=use_saved_files, saved_file_end_date=saved_file_end_date)

    print('Generating Volatility Adjusted Trend Signal!!')
    ## Get Volatility Adjusted Trend Signal
    df_signal = size_cont.get_volatility_adjusted_trend_signal_continuous(df_trend, ticker_list, volatility_window,
                                                                          annual_trading_days)

    print('Getting Average True Range for Stop Loss Calculation!!')
    ## Get Average True Range for Stop Loss Calculation
    df_atr = size_cont.get_average_true_range_portfolio(start_date=start_date, end_date=end_date,
                                                        ticker_list=ticker_list, rolling_atr_window=rolling_atr_window,
                                                        highest_high_window=highest_high_window,
                                                        price_or_returns_calc='price',
                                                        use_coinbase_data=use_coinbase_data,
                                                        use_saved_files=use_saved_files,
                                                        saved_file_end_date=saved_file_end_date)
    df_signal = pd.merge(df_signal, df_atr, left_index=True, right_index=True, how='left')

    print('Calculating Volatility Targeted Position Size and Cash Management!!')
    ## Get Target Volatility Position Sizing and Run Cash Management
    df = size_cont.get_target_volatility_daily_portfolio_positions(
        df_signal, ticker_list=ticker_list, initial_capital=initial_capital, rolling_cov_window=rolling_cov_window,
        stop_loss_strategy=stop_loss_strategy, rolling_atr_window=rolling_atr_window, atr_multiplier=atr_multiplier,
        highest_high_window=highest_high_window,
        cash_buffer_percentage=cash_buffer_percentage, annualized_target_volatility=annualized_target_volatility,
        transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate,
        notional_threshold_pct=notional_threshold_pct, min_trade_notional_abs=min_trade_notional_abs,
        cooldown_counter_threshold=cooldown_counter_threshold, annual_trading_days=annual_trading_days,
        use_specific_start_date=use_specific_start_date, signal_start_date=signal_start_date)

    print('Calculating Portfolio Performance!!')
    ## Calculate Portfolio Performance
    df = size_bin.calculate_portfolio_returns(df, rolling_sharpe_window)

    return df


## Original Signal
def generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr_vol_of_vol(
        start_date, end_date, ticker, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        log_std_window=14, coef_of_variation_window=30, vol_of_vol_z_score_window=252, vol_of_vol_p_min=0.6,
        r2_strong_threshold=0.8, use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential',
        price_or_returns_calc='price', long_only=False, use_coinbase_data=True, use_saved_files=True,
        saved_file_end_date='2025-07-31'):

    # Pull Close Prices from Coinbase
    REPO_ROOT = Path.cwd().parents[1]
    PKL_DIR = REPO_ROOT / "data_folder" / "coinbase_historical_price_folder"
    date_list = cn.coinbase_start_date_by_ticker_dict
    if use_saved_files:
        file_end_date = pd.Timestamp(saved_file_end_date).date()
        filename = f"{ticker}-pickle-{pd.Timestamp(date_list[ticker]).strftime('%Y-%m-%d')}-{file_end_date.strftime('%Y-%m-%d')}"
        # output_file = f'data_folder/coinbase_historical_price_folder/{filename}'
        output_file = PKL_DIR / filename
        df = pd.read_pickle(output_file)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        date_cond = (df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)
        df = df[date_cond]
    else:
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True, start_date=start_date,
                                                            end_date=end_date, save_to_file=False)
        df = (df[['close', 'open']].rename(columns={'close': f'{ticker}_close', 'open': f'{ticker}_open'}))
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
        date_cond = ((df.index.get_level_values('date') >= pd.Timestamp(start_date)) &
                     (df.index.get_level_values('date') <= pd.Timestamp(end_date)))
        df = df[date_cond]

    # Create Column Names
    donchian_binary_signal_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_binary_signal'
    donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
    donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
    trend_binary_signal_col = f'{ticker}_trend_signal'
    trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
    trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
    final_binary_signal_col = f'{ticker}_final_binary_signal'
    final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
    final_signal_col = f'{ticker}_final_signal'

    ## Generate Trend Signal in Log Space
    df_trend = create_trend_strategy_log_space(df, ticker, mavg_start=fast_mavg, mavg_end=slow_mavg,
                                               mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window)

    ## Generate Donchian Channels
    # Donchian Buy signal: Price crosses above upper band
    # Donchian Sell signal: Price crosses below lower band
    df_donchian = calculate_donchian_channel_dual_window(start_date=start_date, end_date=end_date, ticker=ticker,
                                                         price_or_returns_calc=price_or_returns_calc,
                                                         entry_rolling_donchian_window=entry_rolling_donchian_window,
                                                         exit_rolling_donchian_window=exit_rolling_donchian_window,
                                                         use_coinbase_data=use_coinbase_data,
                                                         use_saved_files=use_saved_files,
                                                         saved_file_end_date=saved_file_end_date)

    t_1_close_col = f't_1_close'
    df_donchian[t_1_close_col] = df_donchian[f'close'].shift(1)
    donchian_entry_upper_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_upper_band_{price_or_returns_calc}'
    donchian_entry_lower_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_lower_band_{price_or_returns_calc}'
    donchian_entry_middle_band_col = f'{ticker}_{entry_rolling_donchian_window}_donchian_entry_middle_band_{price_or_returns_calc}'
    donchian_exit_upper_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_upper_band_{price_or_returns_calc}'
    donchian_exit_lower_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_lower_band_{price_or_returns_calc}'
    donchian_exit_middle_band_col = f'{ticker}_{exit_rolling_donchian_window}_donchian_exit_middle_band_{price_or_returns_calc}'
    shift_cols = [donchian_entry_upper_band_col, donchian_entry_lower_band_col, donchian_entry_middle_band_col,
                  donchian_exit_upper_band_col, donchian_exit_lower_band_col, donchian_exit_middle_band_col]
    for col in shift_cols:
        df_donchian[f'{col}_t_2'] = df_donchian[col].shift(1)

    # Donchian Continuous Signal
    df_donchian[donchian_continuous_signal_col] = (
                (df_donchian[t_1_close_col] - df_donchian[f'{donchian_entry_middle_band_col}_t_2']) /
                (df_donchian[f'{donchian_entry_upper_band_col}_t_2'] - df_donchian[
                    f'{donchian_entry_lower_band_col}_t_2']))

    ## Calculate Donchian Channel Rank
    ## Adjust the percentage ranks by 0.5 as without, the ranks go from 0 to 1. Recentering the function by giving it a steeper
    ## slope near the origin takes into account even little information
    df_donchian[donchian_continuous_signal_rank_col] = pct_rank(df_donchian[donchian_continuous_signal_col]) - 0.5

    # Donchian Binary Signal
    gate_long_condition = df_donchian[t_1_close_col] >= df_donchian[f'{donchian_exit_lower_band_col}_t_2']
    gate_short_condition = df_donchian[t_1_close_col] <= df_donchian[f'{donchian_exit_upper_band_col}_t_2']
    # sign of *entry* score decides direction
    entry_sign = np.sign(df_donchian[donchian_continuous_signal_col])
    # treat exact zero as "flat but allowed" (gate=1) so ranking not wiped out
    entry_sign = np.where(entry_sign == 0, 1, entry_sign)  # default to long-side keep
    df_donchian[donchian_binary_signal_col] = np.where(
        entry_sign > 0, gate_long_condition, gate_short_condition).astype(float)

    # Merging the Trend and Donchian Dataframes
    donchian_cols = [t_1_close_col, f'{donchian_entry_upper_band_col}_t_2', f'{donchian_entry_lower_band_col}_t_2',
                     f'{donchian_entry_middle_band_col}_t_2',
                     f'{donchian_exit_upper_band_col}_t_2', f'{donchian_exit_lower_band_col}_t_2',
                     f'{donchian_exit_middle_band_col}_t_2',
                     donchian_binary_signal_col, donchian_continuous_signal_col, donchian_continuous_signal_rank_col]
    df_trend = pd.merge(df_trend, df_donchian[donchian_cols], left_index=True, right_index=True, how='left')

    ## Trend and Donchian Channel Signal
    # Calculate the exponential weighted average of the ranked signals to remove short-term flip-flops (whiplash)
    df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]] = (
        df_trend[[trend_continuous_signal_rank_col, donchian_continuous_signal_rank_col]].ewm(
            span=weighted_signal_ewm_window, adjust=False).mean())

    # Weighted Sum of Rank Columns
    df_trend[final_weighted_additive_signal_col] = (
                ma_crossover_signal_weight * df_trend[trend_continuous_signal_rank_col] +
                donchian_signal_weight * df_trend[donchian_continuous_signal_rank_col])

    # Apply Binary Gate
    if use_donchian_exit_gate:
        df_trend[final_weighted_additive_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[
            donchian_binary_signal_col]

    ## Calculate Rolling R Squared Signal
    df_trend = calculate_rolling_r2(df_trend, ticker=ticker, t_1_close_price_col=t_1_close_col,
                                    rolling_r2_window=rolling_r2_window,
                                    lower_r_sqr_limit=lower_r_sqr_limit, upper_r_sqr_limit=upper_r_sqr_limit,
                                    r2_smooth_window=r2_smooth_window)

    ## Calculate Vol of Vol Signal
    df_trend = generate_vol_of_vol_signal_log_space(df_trend, ticker=ticker, t_1_close_price_col=t_1_close_col,
                                                    log_std_window=log_std_window,
                                                    coef_of_variation_window=coef_of_variation_window,
                                                    vol_of_vol_z_score_window=vol_of_vol_z_score_window,
                                                    vol_of_vol_p_min=vol_of_vol_p_min)

    ## Apply Regime Filters
    strong_rolling_r_sqr_cond = (df_trend[f'{ticker}_rolling_r_sqr'] >= r2_strong_threshold)
    df_trend[f'{ticker}_regime_filter'] = (np.where(strong_rolling_r_sqr_cond, df_trend[f'{ticker}_rolling_r_sqr'],
                                                    df_trend[f'{ticker}_rolling_r_sqr'] * df_trend[
                                                        f'{ticker}_vol_of_vol_penalty']).astype(float))
    df_trend[final_signal_col] = df_trend[final_weighted_additive_signal_col] * df_trend[f'{ticker}_regime_filter']

    # Introduce a Confirmation period for Rolling R Squared Signal
    if r2_confirm_days >= 1:
        df_trend[f'{ticker}_r2_enable'] = ((df_trend[f'{ticker}_rolling_r_sqr'] > 0.5)
                                           .rolling(r2_confirm_days, min_periods=r2_confirm_days).min().fillna(
            0.0).astype(float))
        df_trend[final_signal_col] = df_trend[final_signal_col] * df_trend[f'{ticker}_r2_enable']
    else:
        df_trend[final_signal_col] = df_trend[final_signal_col]

    ## Long-Only Filter
    df_trend[final_signal_col] = np.where(long_only, np.maximum(0, df_trend[final_signal_col]),
                                          df_trend[final_signal_col])

    return df_trend


def get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        log_std_window=14, coef_of_variation_window=30, vol_of_vol_z_score_window=252, vol_of_vol_p_min=0.6,
        r2_strong_threshold=0.8, use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential',
        long_only=False, price_or_returns_calc='price', use_coinbase_data=True, use_saved_files=True,
        saved_file_end_date='2025-07-31'):

    ## Generate trend signal for all tickers
    trend_list = []
    date_list = cn.coinbase_start_date_by_ticker_dict

    for ticker in ticker_list:
        # Create Column Names
        close_price_col = f'{ticker}_close'
        open_price_col = f'{ticker}_open'
        trend_continuous_signal_col = f'{ticker}_mavg_ribbon_slope'
        trend_continuous_signal_rank_col = f'{ticker}_mavg_ribbon_rank'
        donchian_continuous_signal_col = f'{ticker}_donchian_continuous_signal'
        donchian_continuous_signal_rank_col = f'{ticker}_donchian_continuous_signal_rank'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
        rolling_r2_col = f'{ticker}_rolling_r_sqr'
        vol_of_vol_penalty_col = f'{ticker}_vol_of_vol_penalty'
        regime_filter_col = f'{ticker}_regime_filter'
        # rolling_r2_enable_col = f'{ticker}_r2_enable'
        final_signal_col = f'{ticker}_final_signal'

        if pd.to_datetime(date_list[ticker]).date() > start_date:
            run_date = pd.to_datetime(date_list[ticker]).date()
        else:
            run_date = start_date

        df_trend = generate_trend_signal_with_donchian_channel_continuous_with_rolling_r_sqr_vol_of_vol(
            start_date=run_date, end_date=end_date, ticker=ticker, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
            mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
            entry_rolling_donchian_window=entry_rolling_donchian_window,
            exit_rolling_donchian_window=exit_rolling_donchian_window, use_donchian_exit_gate=use_donchian_exit_gate,
            ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
            weighted_signal_ewm_window=weighted_signal_ewm_window,
            rolling_r2_window=rolling_r2_window, lower_r_sqr_limit=lower_r_sqr_limit,
            upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window, r2_confirm_days=r2_confirm_days,
            log_std_window=log_std_window, coef_of_variation_window=coef_of_variation_window,
            vol_of_vol_z_score_window=vol_of_vol_z_score_window, vol_of_vol_p_min=vol_of_vol_p_min,
            r2_strong_threshold=r2_strong_threshold,
            use_activation=use_activation, tanh_activation_constant_dict=tanh_activation_constant_dict,
            moving_avg_type=moving_avg_type, price_or_returns_calc=price_or_returns_calc, long_only=long_only,
            use_coinbase_data=use_coinbase_data, use_saved_files=use_saved_files,
            saved_file_end_date=saved_file_end_date)

        trend_cols = [close_price_col, open_price_col, trend_continuous_signal_col, trend_continuous_signal_rank_col,
                      donchian_continuous_signal_col, donchian_continuous_signal_rank_col,
                      final_weighted_additive_signal_col, rolling_r2_col, vol_of_vol_penalty_col,
                      regime_filter_col, final_signal_col]
        df_trend = df_trend[trend_cols]
        trend_list.append(df_trend)

    df_trend = pd.concat(trend_list, axis=1)

    return df_trend


def apply_target_volatility_position_sizing_continuous_strategy_with_rolling_r_sqr_vol_of_vol(
        start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, mavg_z_score_window,
        entry_rolling_donchian_window, exit_rolling_donchian_window, use_donchian_exit_gate,
        ma_crossover_signal_weight, donchian_signal_weight, weighted_signal_ewm_window,
        rolling_r2_window=30, lower_r_sqr_limit=0.2, upper_r_sqr_limit=0.8, r2_smooth_window=3, r2_confirm_days=0,
        log_std_window=14, coef_of_variation_window=30, vol_of_vol_z_score_window=252, vol_of_vol_p_min=0.6,
        r2_strong_threshold=0.8, use_activation=True, tanh_activation_constant_dict=None, moving_avg_type='exponential',
        long_only=False, price_or_returns_calc='price', initial_capital=15000, rolling_cov_window=20,
        volatility_window=20, stop_loss_strategy='Chandelier', rolling_atr_window=20, atr_multiplier=0.5,
        highest_high_window=56, transaction_cost_est=0.001,
        passive_trade_rate=0.05, notional_threshold_pct=0.05, min_trade_notional_abs=10, cooldown_counter_threshold=3,
        use_coinbase_data=True, use_saved_files=True, saved_file_end_date='2025-07-31', rolling_sharpe_window=50,
        cash_buffer_percentage=0.10, annualized_target_volatility=0.20, annual_trading_days=365,
        use_specific_start_date=False, signal_start_date=None):

    ## Check if data is available for all the tickers
    date_list = cn.coinbase_start_date_by_ticker_dict
    ticker_list = [ticker for ticker in ticker_list if pd.Timestamp(date_list[ticker]).date() < end_date]

    print('Generating Moving Average Ribbon Signal!!')
    ## Generate Trend Signal for all tickers

    df_trend = get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list, fast_mavg=fast_mavg, slow_mavg=slow_mavg,
        mavg_stepsize=mavg_stepsize, mavg_z_score_window=mavg_z_score_window,
        entry_rolling_donchian_window=entry_rolling_donchian_window,
        exit_rolling_donchian_window=exit_rolling_donchian_window, use_donchian_exit_gate=use_donchian_exit_gate,
        ma_crossover_signal_weight=ma_crossover_signal_weight, donchian_signal_weight=donchian_signal_weight,
        weighted_signal_ewm_window=weighted_signal_ewm_window, rolling_r2_window=rolling_r2_window,
        lower_r_sqr_limit=lower_r_sqr_limit, upper_r_sqr_limit=upper_r_sqr_limit, r2_smooth_window=r2_smooth_window,
        r2_confirm_days=r2_confirm_days, log_std_window=log_std_window, coef_of_variation_window=coef_of_variation_window,
        vol_of_vol_z_score_window=vol_of_vol_z_score_window, vol_of_vol_p_min=vol_of_vol_p_min,
        r2_strong_threshold=r2_strong_threshold, use_activation=use_activation,
        tanh_activation_constant_dict=tanh_activation_constant_dict, moving_avg_type=moving_avg_type,
        long_only=long_only, price_or_returns_calc=price_or_returns_calc, use_coinbase_data=use_coinbase_data,
        use_saved_files=use_saved_files, saved_file_end_date=saved_file_end_date)

    print('Generating Volatility Adjusted Trend Signal!!')
    ## Get Volatility Adjusted Trend Signal
    df_signal = size_cont.get_volatility_adjusted_trend_signal_continuous(df_trend, ticker_list, volatility_window,
                                                                          annual_trading_days)

    print('Getting Average True Range for Stop Loss Calculation!!')
    ## Get Average True Range for Stop Loss Calculation
    df_atr = size_cont.get_average_true_range_portfolio(start_date=start_date, end_date=end_date,
                                                        ticker_list=ticker_list, rolling_atr_window=rolling_atr_window,
                                                        highest_high_window=highest_high_window,
                                                        price_or_returns_calc='price',
                                                        use_coinbase_data=use_coinbase_data,
                                                        use_saved_files=use_saved_files,
                                                        saved_file_end_date=saved_file_end_date)
    df_signal = pd.merge(df_signal, df_atr, left_index=True, right_index=True, how='left')

    print('Calculating Volatility Targeted Position Size and Cash Management!!')
    ## Get Target Volatility Position Sizing and Run Cash Management
    df = size_cont.get_target_volatility_daily_portfolio_positions(
        df_signal, ticker_list=ticker_list, initial_capital=initial_capital, rolling_cov_window=rolling_cov_window,
        stop_loss_strategy=stop_loss_strategy, rolling_atr_window=rolling_atr_window, atr_multiplier=atr_multiplier,
        highest_high_window=highest_high_window,
        cash_buffer_percentage=cash_buffer_percentage, annualized_target_volatility=annualized_target_volatility,
        transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate,
        notional_threshold_pct=notional_threshold_pct, min_trade_notional_abs=min_trade_notional_abs,
        cooldown_counter_threshold=cooldown_counter_threshold, annual_trading_days=annual_trading_days,
        use_specific_start_date=use_specific_start_date, signal_start_date=signal_start_date)

    print('Calculating Portfolio Performance!!')
    ## Calculate Portfolio Performance
    df = size_bin.calculate_portfolio_returns(df, rolling_sharpe_window)

    return df

