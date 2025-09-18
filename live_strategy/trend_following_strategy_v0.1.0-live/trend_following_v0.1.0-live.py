# Import all the necessary modules
import os
import sys
import os, sys
# from .../research/notebooks -> go up two levels to repo root
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas_datareader as pdr
import math
import datetime
from datetime import datetime, timezone
import itertools
import ast
import yfinance as yf
import seaborn as sn
from IPython.display import display, HTML
from strategy_signal.trend_following_signal import (
    apply_jupyter_fullscreen_css, get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol
)
from portfolio.strategy_performance import (calculate_sharpe_ratio, calculate_calmar_ratio, calculate_CAGR, calculate_risk_and_performance_metrics,
                                          calculate_compounded_cumulative_returns, estimate_fee_per_trade, rolling_sharpe_ratio)
from utils import coinbase_utils as cn
from portfolio import strategy_performance as perf
from sizing import position_sizing_binary_utils as size_bin
from sizing import position_sizing_continuous_utils as size_cont
from strategy_signal import trend_following_signal as tf
from pathlib import Path
import yaml


def load_prod_strategy_config(strategy_version='v0.1.0'):

    nb_cwd = Path.cwd()  # git/trend_following/research/notebooks
    config_path = (
            nb_cwd.parents[1]                    # -> git/trend_following
            / "live_strategy"
            / f"trend_following_strategy_{strategy_version}-live"
            / "config"
            / f"trend_strategy_config_{strategy_version}.yaml"
    )

    print(config_path)            # sanity check
    print(config_path.exists())   # should be True

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_live_portfolio_equity_and_cash(client, portfolio_name='Default'):

    ## Get Portfolio UUID and Positions
    portfolio_uuid = cn.get_portfolio_uuid(client, portfolio_name)
    df_portfolio_positions = cn.get_portfolio_breakdown(client, portfolio_uuid)

    ## Get Portfolio Value and Available Cash
    cash_cond = (df_portfolio_positions.is_cash == True)
    portfolio_equity = float(df_portfolio_positions[~cash_cond]['total_balance_fiat'].sum())
    available_cash = float(df_portfolio_positions[cash_cond]['available_to_trade_fiat'].sum())

    return portfolio_equity, available_cash


def get_price_map(client, ticker_list):
    result_dict = {}
    for ticker in ticker_list:
        ## Get Best Bid and Offer Data
        book = client.get_product_book(ticker, limit=1)['pricebook']

        ## Build Price Dictionary
        price_dict = {'best_bid_price': float(book['bids'][0]['price']),
                      'best_bid_size': float(book['bids'][0]['size']),
                      'best_ask_price': float(book['asks'][0]['price']),
                      'best_ask_size': float(book['asks'][0]['size']),
                      'best_mid_price': (float(book['bids'][0]['price']) + float(book['asks'][0]['price'])) / 2}

        result_dict[ticker] = price_dict

    return result_dict


def get_current_positions_from_portfolio(client, ticker_list, portfolio_name='Default'):

    df_portfolio = cn.get_portfolio_breakdown(client, portfolio_uuid=cn.get_portfolio_uuid(client, portfolio_name))
    price_map = get_price_map(client, ticker_list)

    ticker_result = {}
    for ticker in ticker_list:
        ticker_cond = (df_portfolio['asset'] == ticker[:-4])
        if df_portfolio[ticker_cond].shape[0] > 0:
            ticker_qty = float(df_portfolio[ticker_cond]['total_balance_crypto'])
        else:
            ticker_qty = 0
        ticker_mid_price = float(price_map[ticker]['best_mid_price'])
        ticker_result[ticker] = {'ticker_qty': ticker_qty,
                                 'ticker_mid_price': ticker_mid_price,
                                 'ticker_current_notional': ticker_qty * ticker_mid_price}

    return ticker_result


def get_strategy_trend_signal(cfg):

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - pd.Timedelta(days=cfg['run']['warmup_days'])

    # Build kwargs directly from cfg sections
    sig_kwargs = {
        # Dates
        "start_date": start_date,
        "end_date": end_date,

        # Universe
        "ticker_list": cfg["universe"]["tickers"],

        # Moving Average Signal
        "fast_mavg": cfg["signals"]["moving_average"]["fast_mavg"],
        "slow_mavg": cfg["signals"]["moving_average"]["slow_mavg"],
        "mavg_stepsize": cfg["signals"]["moving_average"]["mavg_stepsize"],
        "mavg_z_score_window": cfg["signals"]["moving_average"]["mavg_z_score_window"],

        # Donchain Channel Signal
        "entry_rolling_donchian_window": cfg["signals"]["donchian"]["entry_rolling_donchian_window"],
        "exit_rolling_donchian_window": cfg["signals"]["donchian"]["exit_rolling_donchian_window"],
        "use_donchian_exit_gate": cfg["signals"]["donchian"]["use_donchian_exit_gate"],

        # Signal Weights
        "ma_crossover_signal_weight": cfg["signals"]["weighting"]["ma_crossover_signal_weight"],
        "donchian_signal_weight": cfg["signals"]["weighting"]["donchian_signal_weight"],
        "weighted_signal_ewm_window": cfg["signals"]["weighting"]["weighted_signal_ewm_window"],
        "rolling_r2_window": cfg["signals"]["filters"]["rolling_r2"]["rolling_r2_window"],

        # Rolling R Squared Filter
        "lower_r_sqr_limit": cfg["signals"]["filters"]["rolling_r2"]["lower_r_sqr_limit"],
        "upper_r_sqr_limit": cfg["signals"]["filters"]["rolling_r2"]["upper_r_sqr_limit"],
        "r2_smooth_window": cfg["signals"]["filters"]["rolling_r2"]["r2_smooth_window"],
        "r2_confirm_days": cfg["signals"]["filters"]["rolling_r2"]["r2_confirm_days"],

        # Vol of Vol Filter
        "log_std_window": cfg["signals"]["filters"]["vol_of_vol"]["log_std_window"],
        "coef_of_variation_window": cfg["signals"]["filters"]["vol_of_vol"]["coef_of_variation_window"],
        "vol_of_vol_z_score_window": cfg["signals"]["filters"]["vol_of_vol"]["vol_of_vol_z_score_window"],
        "vol_of_vol_p_min": cfg["signals"]["filters"]["vol_of_vol"]["vol_of_vol_p_min"],
        "r2_strong_threshold": cfg["signals"]["filters"]["rolling_r2"]["r2_strong_threshold"],

        # Signal & Data Parameters
        "use_activation": cfg["signals"]["activation"]["use_activation"],
        "tanh_activation_constant_dict": cfg["signals"]["activation"]["tanh_activation_constant_dict"],
        "moving_avg_type": cfg["data"]["moving_avg_type"],
        "long_only": cfg["run"]["long_only"],
        "price_or_returns_calc": cfg["data"]["price_or_returns_calc"],
        "use_coinbase_data": cfg["data"]["use_coinbase_data"],
        "use_saved_files": False,
        "saved_file_end_date": cfg["data"]["saved_file_end_date"]
    }

    df_trend = get_trend_donchian_signal_for_portfolio_with_rolling_r_sqr_vol_of_vol(**sig_kwargs)

    print('Generating Volatility Adjusted Trend Signal!!')
    ## Get Volatility Adjusted Trend Signal
    df_signal = size_cont.get_volatility_adjusted_trend_signal_continuous(df_trend,
                                                                          ticker_list=cfg['universe']['tickers'],
                                                                          volatility_window=cfg['risk_and_sizing'][
                                                                              'volatility_window'],
                                                                          annual_trading_days=cfg['run'][
                                                                              'annual_trading_days'])

    print('Getting Average True Range for Stop Loss Calculation!!')
    ## Get Average True Range for Stop Loss Calculation
    atr_kwargs = {
        # dates
        "start_date": start_date,
        "end_date": end_date,

        # run / universe / data
        "ticker_list": cfg["universe"]["tickers"],

        # risk and sizing
        "rolling_atr_window": cfg['risk_and_sizing']['rolling_atr_window'],

        # data
        "price_or_returns_calc": cfg['data']['price_or_returns_calc'],
        "use_coinbase_data": cfg['data']['use_coinbase_data'],
        "use_saved_files": False,
        "saved_file_end_date": cfg['data']['saved_file_end_date']
    }
    df_atr = size_cont.get_average_true_range_portfolio(**atr_kwargs)
    df_signal = pd.merge(df_signal, df_atr, left_index=True, right_index=True, how='left')

    return df_signal


def get_target_notional_by_ticker(df, date, ticker_list, initial_capital, rolling_cov_window,
                                  rolling_atr_window, atr_multiplier, cash_buffer_percentage,
                                  annualized_target_volatility, transaction_cost_est=0.001,
                                  passive_trade_rate=0.05, notional_threshold_pct=0.02,
                                  min_trade_notional_abs=10, cooldown_counter_threshold=3,
                                  annual_trading_days=365, use_specific_start_date=False,
                                  signal_start_date=None, portfolio_name='Default'):
    ## Create Coinbase Client & Portfolio UUID
    client = cn.get_coinbase_rest_api_client(portfolio_name=portfolio_name)
    portfolio_uuid = cn.get_portfolio_uuid(client, portfolio_name=portfolio_name)

    ## Get Live Portfolio Equity
    portfolio_equity, available_cash = get_live_portfolio_equity_and_cash(client=client, portfolio_name=portfolio_name)

    ## Get Portfolio Breakdown
    df_portfolio_breakdown = cn.get_portfolio_breakdown(client, portfolio_uuid=portfolio_uuid)

    ## Get Current Positions using Mid-Price
    ## TODO: CHECK TO SEE IF THE MID-PRICE BEING CAPTURED IS ACCURATE FROM COINBASE
    current_positions = get_current_positions_from_portfolio(client, ticker_list=ticker_list,
                                                             portfolio_name=portfolio_name)

    ## Calculate the covariance matrix for tickers in the portfolio
    returns_cols = [f'{ticker}_t_1_close_pct_returns' for ticker in ticker_list]
    cov_matrix = df[returns_cols].rolling(rolling_cov_window).cov(pairwise=True).dropna()

    ## Delete rows prior to the first available date of the covariance matrix
    cov_matrix_start_date = cov_matrix.index.get_level_values(0).min()
    df = df[df.index >= cov_matrix_start_date]

    ## Derive the Daily Target Portfolio Volatility
    daily_target_volatility = annualized_target_volatility / np.sqrt(annual_trading_days)

    ## Reorder dataframe columns
    for ticker in ticker_list:
        df[f'{ticker}_new_position_size'] = 0.0
        df[f'{ticker}_new_position_notional'] = 0.0
        df[f'{ticker}_open_position_size'] = 0.0
        df[f'{ticker}_open_position_notional'] = 0.0
        df[f'{ticker}_actual_position_size'] = 0.0
        df[f'{ticker}_actual_position_notional'] = 0.0
        df[f'{ticker}_short_sale_proceeds'] = 0.0
        df[f'{ticker}_new_position_entry_exit_price'] = 0.0
        df[f'{ticker}_target_vol_normalized_weight'] = 0.0
        df[f'{ticker}_target_notional'] = 0.0
        df[f'{ticker}_target_size'] = 0.0
        df[f'{ticker}_cash_shrink_factor'] = 0.0
        df[f'{ticker}_stop_loss'] = 0.0
        df[f'{ticker}_stopout_flag'] = False
        df[f'{ticker}_cooldown_counter'] = 0.0
        df[f'{ticker}_event'] = np.nan
    ord_cols = size_bin.reorder_columns_by_ticker(df.columns, ticker_list)
    df = df[ord_cols]

    ## Portfolio Level Cash and Positions are all set to 0
    df['daily_portfolio_volatility'] = 0.0
    df['available_cash'] = 0.0
    df['count_of_positions'] = 0.0
    df['total_actual_position_notional'] = 0.0
    df['total_target_notional'] = 0.0
    df['total_portfolio_value'] = 0.0
    df['total_portfolio_value_upper_limit'] = 0.0
    df['target_vol_scaling_factor'] = 1.0
    df['cash_scaling_factor'] = 1.0
    df['cash_shrink_factor'] = 1.0
    df['final_scaling_factor'] = 1.0

    ## Cash and the Total Portfolio Value on Day 1 is the initial capital for the strategy
    if use_specific_start_date:
        start_index_position = df.index.get_loc(signal_start_date)
    else:
        start_index_position = 0
    # df['available_cash'][start_index_position] = initial_capital
    # df['total_portfolio_value'][start_index_position] = initial_capital

    ## Identify Daily Positions starting from day 2
    # for date in df.index[start_index_position + 1:]:
    previous_date = df.index[df.index.get_loc(date) - 1]

    ## Start the day with the available cash from portfolio
    df['available_cash'].loc[date] = available_cash  # df['available_cash'].loc[previous_date]

    ## Calculate Total Portfolio Value from Portfolio Positions
    short_sale_proceeds_cols = [f'{ticker}_short_sale_proceeds' for ticker in ticker_list]
    df['total_actual_position_notional'].loc[date] = portfolio_equity
    total_portfolio_value = (df['available_cash'].loc[date] +
                             df[short_sale_proceeds_cols].loc[date].sum() +
                             df['total_actual_position_notional'].loc[date])
    df['total_portfolio_value'].loc[date] = total_portfolio_value

    ## Update Total Portfolio Value Upper Limit based on the Total Portfolio Value
    total_portfolio_value_upper_limit = (df['total_portfolio_value'].loc[date] *
                                         (1 - cash_buffer_percentage))
    df['total_portfolio_value_upper_limit'].loc[date] = total_portfolio_value_upper_limit

    ## Calculate the target notional by ticker
    df = size_cont.get_target_volatility_position_sizing(df, cov_matrix, date, ticker_list, daily_target_volatility,
                                                         total_portfolio_value_upper_limit)

    ## Adjust Positions for Cash Available
    desired_positions, cash_shrink_factor = size_cont.get_cash_adjusted_desired_positions(
        df, date, previous_date, ticker_list, cash_buffer_percentage, transaction_cost_est, passive_trade_rate,
        total_portfolio_value, notional_threshold_pct, min_trade_notional_abs)

    ## Apply Cash Shrink Factor to Desired Positions
    for ticker in ticker_list:
        desired_positions[ticker]['new_trade_notional'] = desired_positions[ticker][
                                                              'new_trade_notional'] * cash_shrink_factor
        desired_positions[ticker]['trade_fees'] = desired_positions[ticker]['trade_fees'] * cash_shrink_factor

    return df, desired_positions, cash_shrink_factor