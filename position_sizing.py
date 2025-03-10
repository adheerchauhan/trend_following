import pandas as pd
import numpy as np
import strategy_performance as perf
import trend_following_signal as tf
import coinbase_utils as cn


def reorder_columns_by_ticker(columns, tickers):
    ordered_columns = []
    for ticker in tickers:
        # Find all columns for this ticker
        ticker_columns = [col for col in columns if col.startswith(f"{ticker}")]
        # Add them to the ordered list
        ordered_columns.extend(sorted(ticker_columns))  # Sorting ensures a predictable order (e.g., open, close, volume)
    return ordered_columns


# Function to calculate portfolio volatility
def calculate_portfolio_volatility(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


# Function to calculate position sizes based on target volatility
def get_target_volatility_position_sizing(df, cov_matrix, date, ticker_list, daily_target_volatility,
                                          total_portfolio_value_upper_limit):
    ## Scale weights of positions to ensure the portfolio is in line with the target volatility
    unscaled_weight_cols = [f'{ticker}_position_volatility_adjusted_weight' for ticker in ticker_list]
    scaled_weight_cols = [f'{ticker}_target_vol_normalized_weight' for ticker in ticker_list]
    target_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
    t_1_price_cols = [f'{ticker}_t_1_close' for ticker in ticker_list]

    if date not in df.index or date not in cov_matrix.index:
        raise ValueError(f"Date {date} not found in DataFrame or covariance matrix index.")

    ## Iterate through each day and calculate the covariance matrix, portfolio volatility and scaling factors
    daily_weights = np.abs(df.loc[date, unscaled_weight_cols].values)
    daily_cov_matrix = cov_matrix.loc[date].values
    daily_portfolio_volatility = calculate_portfolio_volatility(daily_weights, daily_cov_matrix)
    df.loc[date, 'daily_portfolio_volatility'] = daily_portfolio_volatility
    if daily_portfolio_volatility > 0:
        scaling_factor = daily_target_volatility / daily_portfolio_volatility
    else:
        scaling_factor = 0
    df.loc[date, 'target_vol_scaling_factor'] = scaling_factor

    # Scale the weights to target volatility
    scaled_weights = daily_weights * scaling_factor
    df.loc[date, scaled_weight_cols] = scaled_weights

    ## Calculate the target notional and size
    target_notionals = scaled_weights * total_portfolio_value_upper_limit
    df.loc[date, target_notional_cols] = target_notionals
    target_sizes = target_notionals / df.loc[date, t_1_price_cols].values

    for i, ticker in enumerate(ticker_list):
        df.loc[date, f'{ticker}_target_size'] = target_sizes[i]

    total_target_notional = target_notionals.sum()
    df.loc[date, 'total_target_notional'] = total_target_notional

    ## Check if the Target Notional is greater than the portfolio value
    if total_target_notional > total_portfolio_value_upper_limit:
        target_notional_scaling_factor = total_portfolio_value_upper_limit / total_target_notional
        df.loc[date, 'target_notional_scaling_factor'] = target_notional_scaling_factor
        adjusted_target_notionals = target_notionals * target_notional_scaling_factor
        df.loc[date, target_notional_cols] = adjusted_target_notionals

        adjusted_target_sizes = adjusted_target_notionals / df.loc[date, t_1_price_cols].values
        for i, ticker in enumerate(ticker_list):
            df.loc[date, f'{ticker}_target_size'] = adjusted_target_sizes[i]

        df.loc[date, 'total_target_notional'] = adjusted_target_notionals.sum()

    return df


def get_daily_positions_and_portfolio_cash(df, date, ticker_list, fast_mavg, mavg_stepsize, slow_mavg, rolling_donchian_window,
                                           transaction_cost_est, passive_trade_rate):

    previous_date = df.index[df.index.get_loc(date) - 1]

    ## Estimated Transaction Costs and Fees
    est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))

    for ticker in ticker_list:
        t_1_close_price_col = f'{ticker}_t_1_close'
        signal_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_signal'
        target_position_notional_col = f'{ticker}_target_notional'
        target_position_size_col = f'{ticker}_target_size'
        actual_position_notional_col = f'{ticker}_actual_position_notional'
        actual_position_size_col = f'{ticker}_actual_size'
        actual_position_entry_price_col = f'{ticker}_actual_position_entry_price'
        actual_position_exit_price_col = f'{ticker}_actual_position_exit_price'
        short_sale_proceeds_col = f'{ticker}_short_sale_proceeds'
        event_col = f'{ticker}_event'
        df[actual_position_notional_col].loc[date] = 0.0
        df[actual_position_entry_price_col].loc[date] = 0.0
        df[actual_position_size_col].loc[date] = 0.0
        df[event_col].loc[date] = np.nan

        ## Taking a New Long position
        if (df[signal_col].loc[date] == 1) and (df[signal_col].loc[previous_date] == 0) and (
                df[actual_position_notional_col].loc[previous_date] == 0):
            # Building a cash buffer
            available_cash_buffer = df['available_cash'].loc[date] * (1 - 0.10)
            target_long_notional = df[target_position_notional_col].loc[date]
            if available_cash_buffer - target_long_notional > 0:
                net_long_notional = target_long_notional * (1 - est_fees)
                df[actual_position_notional_col].loc[date] = net_long_notional
                df[actual_position_entry_price_col].loc[date] = df[t_1_close_price_col].loc[date]
                df[actual_position_size_col].loc[date] = (df[actual_position_notional_col].loc[date] /
                                                          df[actual_position_entry_price_col].loc[date])
                df['available_cash'].loc[date] = df['available_cash'].loc[date] - target_long_notional
                df[event_col].loc[date] = 'New Long Position'

        ## Taking a New Short position
        elif (df[signal_col].loc[date] == -1) and (df[signal_col].loc[previous_date] == 0) and (
                df[actual_position_notional_col].loc[previous_date] == 0):
            # Building a cash buffer
            available_cash_buffer = df['available_cash'].loc[date] * (1 - 0.10)
            target_short_notional = df[target_position_notional_col].loc[date]
            if available_cash_buffer - target_short_notional > 0:
                net_short_notional = -(target_short_notional * (1 - est_fees))
                df[actual_position_notional_col].loc[date] = net_short_notional
                df[actual_position_entry_price_col].loc[date] = df[t_1_close_price_col].loc[date]
                df[actual_position_size_col].loc[date] = (df[actual_position_notional_col].loc[date] /
                                                          df[actual_position_entry_price_col].loc[date])
                df[short_sale_proceeds_col].loc[date] = -net_short_notional
                df[event_col].loc[date] = 'New Short Position'

        ## Open Long Position
        elif (df[signal_col].loc[date] == 1) and (df[signal_col].loc[previous_date] == 1) and (
                df[actual_position_notional_col].loc[previous_date] > 0):
            df[actual_position_size_col].loc[date] = df[actual_position_size_col].loc[previous_date]
            df[actual_position_notional_col].loc[date] = (df[actual_position_size_col].loc[date] *
                                                          df[t_1_close_price_col].loc[date])
            df[actual_position_entry_price_col].loc[date] = df[actual_position_entry_price_col].loc[previous_date]
            df[event_col].loc[date] = 'Open Long Position'

        ## Open Short Position
        elif (df[signal_col].loc[date] == -1) and (df[signal_col].loc[previous_date] == -1) and (
            df[actual_position_notional_col].loc[previous_date] < 0):
            df[actual_position_size_col].loc[date] = df[actual_position_size_col].loc[previous_date]
            df[actual_position_notional_col].loc[date] = (df[actual_position_size_col].loc[date] *
                                                          df[t_1_close_price_col].loc[date])
            df[actual_position_entry_price_col].loc[date] = df[actual_position_entry_price_col].loc[previous_date]
            df[short_sale_proceeds_col].loc[date] = df[short_sale_proceeds_col].loc[previous_date]
            df[event_col].loc[date] = 'Open Short Position'

        ## Taking a New Long Position with an Open Short Position
        elif (df[signal_col].loc[date] == 1) and (df[signal_col].loc[previous_date] == -1) and (
                df[actual_position_notional_col].loc[previous_date] < 0):
            target_long_notional = min(df[target_position_notional_col].loc[date], df['available_cash'].loc[date])
            if target_long_notional > 0:
                net_long_notional = target_long_notional * (1 - est_fees)
                df[actual_position_notional_col].loc[date] = net_long_notional
                df[actual_position_entry_price_col].loc[date] = df[t_1_close_price_col].loc[date]
                df[actual_position_size_col].loc[date] = (df[actual_position_notional_col].loc[date] /
                                                          df[actual_position_entry_price_col].loc[date])
                df[short_sale_proceeds_col].loc[date] = 0.0
                df['available_cash'].loc[date] = df['available_cash'].loc[date] - target_long_notional
                df[event_col].loc[date] = 'New Long with Open Short Position'

        ## Taking a New Short Position with an Existing Long Position
        elif (df[signal_col].loc[date] == -1) and (df[signal_col].loc[previous_date] == 1) and (
                df[actual_position_notional_col].loc[previous_date] > 0):
            target_short_notional = min(df[target_position_notional_col].loc[date], df['available_cash'].loc[date])
            if target_short_notional > 0:
                net_short_notional = -(target_short_notional * (1 - est_fees))
                df[actual_position_notional_col].loc[date] = net_short_notional
                df[actual_position_entry_price_col].loc[date] = df[t_1_close_price_col].loc[date]
                df[actual_position_size_col].loc[date] = (df[actual_position_notional_col].loc[date] /
                                                          df[actual_position_entry_price_col].loc[date])
                df[short_sale_proceeds_col].loc[date] = -net_short_notional
                df[event_col].loc[date] = 'New Short with Open Long Position'

        ## Closing a Long Position
        elif (df[signal_col].loc[date] == 0) and (df[signal_col].loc[previous_date] == 1) and (
                df[actual_position_notional_col].loc[previous_date] > 0):
            df[actual_position_notional_col].loc[date] = 0
            df[actual_position_entry_price_col].loc[date] = 0
            df[actual_position_size_col].loc[date] = 0
            df[actual_position_exit_price_col].loc[date] = df[t_1_close_price_col].loc[date]
            closing_long_market_value = (df[actual_position_size_col].loc[previous_date] *
                                         df[actual_position_exit_price_col].loc[date])
            net_closing_long_market_value = closing_long_market_value * (1 - est_fees)
            df['available_cash'].loc[date] = (df['available_cash'].loc[date] + net_closing_long_market_value)
            df[event_col].loc[date] = 'Closing Long Position'

        ## Closing a Short Position
        elif (df[signal_col].loc[date] == 0) and (df[signal_col].loc[previous_date] == -1) and (
                df[actual_position_notional_col].loc[previous_date] < 0):
            df[actual_position_notional_col].loc[date] = 0
            df[actual_position_entry_price_col].loc[date] = 0
            df[actual_position_size_col].loc[date] = 0
            df[actual_position_exit_price_col].loc[date] = df[t_1_close_price_col].loc[date]
            df[short_sale_proceeds_col].loc[date] = 0.0
            closing_short_market_value = (df[actual_position_size_col].loc[previous_date] *
                                          df[actual_position_exit_price_col].loc[date])
            net_closing_short_market_value = closing_short_market_value * (1 - est_fees)
            short_sale_proceeds = df[short_sale_proceeds_col].loc[previous_date]
            df['available_cash'].loc[date] = (df['available_cash'].loc[date] +
                                              (net_closing_short_market_value + short_sale_proceeds))
            df[event_col].loc[date] = 'Closing Short Position'

        ## No Event
        elif (df[signal_col].loc[date] == 0) and (df[signal_col].loc[previous_date] == 0):
            df[actual_position_notional_col].loc[date] = 0
            df[actual_position_entry_price_col].loc[date] = 0
            df[actual_position_size_col].loc[date] = 0
            df[short_sale_proceeds_col].loc[date] = 0
            df[event_col].loc[date] = 'No Event'

    ## Calculate End of Day Portfolio Positions
    actual_position_notional_cols = [f'{ticker}_actual_position_notional' for ticker in ticker_list]
    short_sale_proceeds_cols = [f'{ticker}_short_sale_proceeds' for ticker in ticker_list]
    df['count_of_positions'].loc[date] = df[actual_position_notional_cols].loc[date].ne(0).sum()
    df['total_actual_position_notional'].loc[date] = df[actual_position_notional_cols].loc[date].sum()
    df['total_portfolio_value'].loc[date] = (df['available_cash'].loc[date] +
                                             df[short_sale_proceeds_cols].loc[date].sum() +
                                             df['total_actual_position_notional'].loc[date])

    return df


# def calculate_portfolio_cash_and_market_value_per_day(df, date, ticker_list, transaction_cost_est, passive_trade_rate):
#     ## Create Previous Date Index
#     previous_date = df.index[df.index.get_loc(date) - 1]
#
#     ## Calculate the estimated transaction costs and exchange fees
#     est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))
#
#     event_cols = [f'{ticker}_event' for ticker in ticker_list]
#     actual_position_notional_cols = [f'{ticker}_actual_position_notional' for ticker in ticker_list]
#     target_position_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
#     actual_position_size_cols = [f'{ticker}_actual_size' for ticker in ticker_list]
#     exit_price_cols = [f'{ticker}_actual_position_exit_price' for ticker in ticker_list]
#     short_sale_proceeds_cols = [f'{ticker}_short_sale_proceeds' for ticker in ticker_list]
#
#     ## New Long Positions
#     long_position_identifiers = ['New Long Position', 'New Long with Open Short Position']
#     new_long_mask = df[event_cols].loc[date].isin(long_position_identifiers)
#     new_long_market_value = df[actual_position_notional_cols].loc[date].where(new_long_mask.values).sum()
#     new_target_long_market_value = df[target_position_notional_cols].loc[date].where(new_long_mask.values).sum()
#     # net_long_market_value = new_long_market_value * (1 - est_fees)
#     ## We're subtracting target long market value from available cash because that what is paid prior to transaction costs
#     ## The actual market value of the position is the target long market value minus transaction costs
#     df['available_cash'].loc[date] = (df['available_cash'].loc[date] -
#                                       new_target_long_market_value)
#
#     ## New Short Positions
#     ## Cash is generated from a New Short position and is stored as Short Sale Proceeds
#     short_position_identifiers = ['New Short Position', 'New Short with Open Long Position']
#     new_short_mask = df[event_cols].loc[date].isin(short_position_identifiers)
#     new_short_market_value = -df[actual_position_notional_cols].loc[date].where(new_short_mask.values).sum()
#
#     ## Closed Long Positions
#     closing_long_mask = df[event_cols].loc[date] == 'Closing Long Position'
#     closing_long_masked_sizes = (df[actual_position_size_cols].loc[previous_date]
#                                  .where(closing_long_mask.values).fillna(0).values)
#     closing_long_masked_prices = (df[exit_price_cols].loc[date]
#                                   .where(closing_long_mask.values).fillna(0).values)
#     closing_long_market_value = (closing_long_masked_sizes * closing_long_masked_prices).sum()
#     net_closing_long_market_value = closing_long_market_value * (1 - est_fees)
#     df['available_cash'].loc[date] = (df['available_cash'].loc[date] +
#                                       net_closing_long_market_value)
#
#     ## Closed Short Positions
#     closing_short_mask = df[event_cols].loc[date] == 'Closing Short Position'
#     closing_short_masked_sizes = (df[actual_position_size_cols].loc[previous_date]
#                                   .where(closing_short_mask.values).fillna(0).values)
#     closing_short_masked_prices = df[exit_price_cols].loc[date].where(closing_short_mask.values).fillna(0).values
#     closing_short_market_value = (closing_short_masked_sizes * closing_short_masked_prices).sum()
#     net_closing_short_market_value = closing_short_market_value * (1 - est_fees)
#     short_sale_proceeds_value = df[short_sale_proceeds_cols].loc[previous_date].where(closing_short_mask.values).sum()
#     df['available_cash'].loc[date] = (df['available_cash'].loc[date] +
#                                       (short_sale_proceeds_value + net_closing_short_market_value))
#
#     ## Portfolio Positions
#     df['count_of_positions'].loc[date] = df[actual_position_notional_cols].loc[date].ne(0).sum()
#     df['total_actual_position_notional'].loc[date] = df[actual_position_notional_cols].loc[date].sum()
#     df['total_portfolio_value'].loc[date] = (df['available_cash'].loc[date] +
#                                              df[short_sale_proceeds_cols].loc[date].sum() +
#                                              df['total_actual_position_notional'].loc[date])
#
#     return df


def get_target_volatility_daily_portfolio_positions(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                                    rolling_donchian_window, initial_capital, rolling_cov_window,
                                                    cash_buffer_percentage, annualized_target_volatility,
                                                    transaction_cost_est=0.001, passive_trade_rate=0.05,
                                                    annual_trading_days=365, use_specific_start_date=False,
                                                    signal_start_date=None):

    ## Calculate the covariance matrix for tickers in the portfolio
    returns_cols = [f'{ticker}_pct_returns' for ticker in ticker_list]
    cov_matrix = df[returns_cols].rolling(rolling_cov_window).cov(pairwise=True).dropna()

    ## Delete rows prior to the first available date of the covariance matrix
    cov_matrix_start_date = cov_matrix.index[0][0]
    df = df[df.index >= cov_matrix_start_date]

    ## Derive the Daily Target Portfolio Volatility
    daily_target_volatility = annualized_target_volatility / np.sqrt(annual_trading_days)

    ## Reorder dataframe columns
    col_list = []
    for ticker in ticker_list:
        df[f'{ticker}_actual_size'] = 0.0
        df[f'{ticker}_actual_position_notional'] = 0.0
        df[f'{ticker}_short_sale_proceeds'] = 0.0
        df[f'{ticker}_actual_position_entry_price'] = 0.0
        df[f'{ticker}_actual_position_exit_price'] = 0.0
        df[f'{ticker}_target_vol_normalized_weight'] = 0.0
        df[f'{ticker}_target_notional'] = 0.0
        df[f'{ticker}_target_size'] = 0.0
        df[f'{ticker}_event'] = np.nan
    ord_cols = reorder_columns_by_ticker(df.columns, ticker_list)
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
    df['target_notional_scaling_factor'] = 1.0

    ## Cash and the Total Portfolio Value on Day 1 is the initial capital for the strategy
    if use_specific_start_date:
        start_index_position = df.index.get_loc(signal_start_date)
    else:
        start_index_position = 0
    df['available_cash'][start_index_position] = initial_capital
    df['total_portfolio_value'][start_index_position] = initial_capital

    ## Identify Daily Positions starting from day 2
    for date in df.index[start_index_position + 1:]:
        previous_date = df.index[df.index.get_loc(date) - 1]

        ## Start the day with the available cash from yesterday
        df['available_cash'].loc[date] = df['available_cash'].loc[previous_date]
        total_portfolio_value_upper_limit = (df['total_portfolio_value'].loc[previous_date] *
                                             (1 - cash_buffer_percentage))
        df['total_portfolio_value_upper_limit'].loc[date] = total_portfolio_value_upper_limit

        ## Calculate the target notional by ticker
        df = get_target_volatility_position_sizing(df, cov_matrix, date, ticker_list, daily_target_volatility,
                                                   total_portfolio_value_upper_limit)

        ## Get the daily positions
        df = get_daily_positions_and_portfolio_cash(df, date, ticker_list, fast_mavg, mavg_stepsize, slow_mavg,
                                                    rolling_donchian_window, transaction_cost_est, passive_trade_rate)

        ## Calculate Portfolio Value and Available Cash
        # df = calculate_portfolio_cash_and_market_value_per_day(df, date, ticker_list, transaction_cost_est,
        #                                                        passive_trade_rate)

    return df


# Get Volatility Adjusted Trend Signal for Target Volatility Strategy
def get_volatility_adjusted_trend_signal(df, ticker_list, volatility_window, fast_mavg, mavg_stepsize, slow_mavg,
                                         rolling_donchian_window, annual_trading_days=365):
    ticker_signal_dict = {}
    final_cols = []
    for ticker in ticker_list:
        trend_signal_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_signal'
        trend_returns_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_returns'
        trend_trades_col = f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_trades'
        annualized_volatility_col = f'{ticker}_annualized_volatility_{volatility_window}'
        vol_adj_trend_signal_col = f'{ticker}_vol_adjusted_trend_signal'

        ## Calculate Position Volatility Adjusted Trend Signal
        df = tf.get_returns_volatility(df, vol_range_list=[volatility_window], close_px_col=f'{ticker}')
        df[annualized_volatility_col] = (df[f'{ticker}_volatility_{volatility_window}'] *
                                         np.sqrt(annual_trading_days))
        df[vol_adj_trend_signal_col] = (df[trend_signal_col] / df[annualized_volatility_col])
        df[vol_adj_trend_signal_col] = df[vol_adj_trend_signal_col].fillna(0)
        df[f'{ticker}_t_1_close'] = df[f'{ticker}'].shift(1)
        trend_cols = [f'{ticker}', f'{ticker}_t_1_close', f'{ticker}_pct_returns', trend_signal_col, trend_returns_col,
                      trend_trades_col, annualized_volatility_col, vol_adj_trend_signal_col]
        final_cols.append(trend_cols)
        ticker_signal_dict[ticker] = df[trend_cols]
    df_signal = pd.concat(ticker_signal_dict, axis=1)

    ## Assign new column names to the dataframe
    df_signal.columns = df_signal.columns.to_flat_index()
    final_cols = [item for sublist in final_cols for item in sublist]
    df_signal.columns = final_cols

    ## Normalize the weights of each position by the total weight of the portfolio
    vol_normalized_signal_cols = [f'{ticker}_vol_adjusted_trend_signal' for ticker in ticker_list]
    df_signal[vol_normalized_signal_cols] = df_signal[vol_normalized_signal_cols].fillna(0)
    for ticker in ticker_list:
        df_signal[f'{ticker}_position_volatility_adjusted_weight'] = (df_signal[f'{ticker}_vol_adjusted_trend_signal'] /
                                                                      df_signal[vol_normalized_signal_cols].abs()
                                                                      .sum(axis=1))
        df_signal[f'{ticker}_position_volatility_adjusted_weight'] = df_signal[
            f'{ticker}_position_volatility_adjusted_weight'].fillna(0)

    return df_signal


# Calculate Portfolio Returns and Rolling Sharpe Ratio
def calculate_portfolio_returns(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, rolling_donchian_window,
                                rolling_sharpe_window):
    ## Calculate Portfolio Returns
    df['portfolio_daily_pct_returns'] = df['total_portfolio_value'].pct_change()
    df['portfolio_daily_pct_returns'].replace([np.inf, -np.inf], 0, inplace=True)
    df['portfolio_daily_pct_returns'] = df['portfolio_daily_pct_returns'].fillna(0)
    trade_cols = [
        f'{ticker}_{fast_mavg}_{mavg_stepsize}_{slow_mavg}_mavg_crossover_{rolling_donchian_window}_donchian_strategy_trades'
        for ticker in ticker_list]
    df['portfolio_trade_count'] = np.abs(df[trade_cols]).sum(axis=1)
    df['portfolio_strategy_cumulative_return'] = (1 + df['portfolio_daily_pct_returns']).cumprod() - 1

    ## Calculate Rolling Sharpe Ratio
    df[f'rolling_sharpe_{rolling_sharpe_window}'] = (perf.rolling_sharpe_ratio(
        df, window=rolling_sharpe_window, strategy_daily_return_col='portfolio_daily_pct_returns',
        strategy_trade_count_col='portfolio_trade_count', include_transaction_costs_and_fees=False,
        annual_trading_days=365))
    df = df.rename(
        columns={f'rolling_sharpe_{rolling_sharpe_window}': f'portfolio_rolling_sharpe_{rolling_sharpe_window}'})

    return df


# Target Volatility Position Sizing Strategy for a Trend Following Signal
def apply_target_volatility_position_sizing_strategy(start_date, end_date, ticker_list, fast_mavg, slow_mavg,
                                                     mavg_stepsize, rolling_donchian_window, long_only=False,
                                                     initial_capital=15000, rolling_cov_window=20, volatility_window=20,
                                                     transaction_cost_est=0.001, passive_trade_rate=0.05,
                                                     use_coinbase_data=True, rolling_sharpe_window=50,
                                                     cash_buffer_percentage=0.10, annualized_target_volatility=0.20,
                                                     annual_trading_days=365, use_specific_start_date=False,
                                                     signal_start_date=None):
    ## Generate Trend Signal for all tickers
    df_trend = tf.get_trend_donchian_signal_for_portfolio(start_date=start_date, end_date=end_date,
                                                          ticker_list=ticker_list, fast_mavg=fast_mavg,
                                                          slow_mavg=slow_mavg,
                                                          mavg_stepsize=mavg_stepsize,
                                                          rolling_donchian_window=rolling_donchian_window,
                                                          long_only=long_only, use_coinbase_data=use_coinbase_data)

    ## Get Volatility Adjusted Trend Signal
    df_signal = get_volatility_adjusted_trend_signal(df_trend, ticker_list, volatility_window, fast_mavg, mavg_stepsize,
                                                     slow_mavg, rolling_donchian_window, annual_trading_days)

    ## Get Daily Positions
    df = get_target_volatility_daily_portfolio_positions(df_signal, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                                         rolling_donchian_window, initial_capital, rolling_cov_window,
                                                         cash_buffer_percentage, annualized_target_volatility,
                                                         transaction_cost_est, passive_trade_rate, annual_trading_days,
                                                         use_specific_start_date, signal_start_date)

    ## Calculate Portfolio Performance
    df = calculate_portfolio_returns(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, rolling_donchian_window,
                                     rolling_sharpe_window)

    return df


def calculate_average_true_range(start_date, end_date, ticker, price_or_returns_calc='price', rolling_atr_window=20,
                                 use_coinbase_data=True):
    if use_coinbase_data:
        # df = cn.get_coinbase_ohlc_data(ticker=ticker)
        df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, end_date=end_date, save_to_file=False)
        df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
        df.columns = [f'{ticker}_{x}' for x in df.columns]
    else:
        df = tf.load_financial_data(start_date, end_date, ticker, print_status=False)  # .shift(1)
        df.columns = [f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_adjclose',
                      f'{ticker}_volume']

    ## Get T-1 Close Price
    df[f'{ticker}_t_1_close'] = df[f'{ticker}_close'].shift(1)

    if price_or_returns_calc == 'price':
        # Calculate the True Range (TR) and Average True Range (ATR)
        df[f'{ticker}_high-low'] = df[f'{ticker}_high'] - df[f'{ticker}_low']
        df[f'{ticker}_high-close'] = np.abs(df[f'{ticker}_high'] - df[f'{ticker}_close'].shift(1))
        df[f'{ticker}_low-close'] = np.abs(df[f'{ticker}_low'] - df[f'{ticker}_close'].shift(1))
        df[f'{ticker}_true_range_price'] = df[
            [f'{ticker}_high-low', f'{ticker}_high-close', f'{ticker}_low-close']].max(axis=1)
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[f'{ticker}_true_range_price'].ewm(
            span=rolling_atr_window, adjust=False).mean()

        ## Shift by 1 to avoid look-ahead bias
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_price'] = df[
            f'{ticker}_{rolling_atr_window}_avg_true_range_price'].shift(1)

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

        ## Shift by 1 to avoid look-ahead bias
        df[f'{ticker}_{rolling_atr_window}_avg_true_range_returns'] = df[
            f'{ticker}_{rolling_atr_window}_avg_true_range_returns'].shift(1)

    return df


## Position Sizing Strategy will calculate Account Risk based on risk tolerance per trade and Trade Risk based on ATR or stop loss multiple
## Position Size will be determined by dividing the Account Risk by the Trade Risk
def get_atr_by_ticker(start_date, end_date, ticker_list, rolling_atr_window):

    atr_list = []
    for ticker in ticker_list:
        df_atr = calculate_average_true_range(start_date=start_date, end_date=end_date, ticker=ticker,
                                              price_or_returns_calc='price',
                                              rolling_atr_window=rolling_atr_window, use_coinbase_data=True)

        ## Calculate Target Size and Notional based on the Average True Range
        atr_cols = [f'{ticker}_close', f'{ticker}_t_1_close', f'{ticker}_{rolling_atr_window}_avg_true_range_price']
        atr_list.append(df_atr[atr_cols])

    df_atr_ticker = pd.concat(atr_list, axis=1)

    return df_atr_ticker


def calculate_atr_target_notional(df, date, ticker_list, rolling_atr_window, risk_per_trade, stop_loss_multiple,
                                  t_1_portfolio_value_col='total_portfolio_value_upper_limit'):

    previous_date = df.index[df.index.get_loc(date) - 1]

    ## Calculate the target notional by ticker
    for ticker in ticker_list:
        atr_col = f'{ticker}_{rolling_atr_window}_avg_true_range_price'
        df.loc[date, f'{ticker}_account_risk'] = (df.loc[date, t_1_portfolio_value_col] * risk_per_trade)
        df.loc[date, f'{ticker}_trade_risk'] = (df.loc[previous_date, atr_col] * stop_loss_multiple)
        df.loc[date, f'{ticker}_target_size'] = (df.loc[date, f'{ticker}_account_risk'] /
                                                 df.loc[date, f'{ticker}_trade_risk'])
        df.loc[date, f'{ticker}_target_notional'] = (df.loc[date, f'{ticker}_target_size'] *
                                                     df.loc[date, f'{ticker}_t_1_close'])

    ## Calculate the total target notional for the day
    target_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
    total_target_notional = df.loc[date, target_notional_cols].sum()
    df.loc[date, 'total_target_notional'] = total_target_notional

    ## Check if the Target Notional Positions need to be scaled
    if total_target_notional > df.loc[date, t_1_portfolio_value_col]:
        scaling_factor = df.loc[date, t_1_portfolio_value_col] / total_target_notional
        df.loc[date, 'target_notional_scaling_factor'] = scaling_factor
        daily_target_notionals = df.loc[date, target_notional_cols]
        scaled_target_notionals = daily_target_notionals * scaling_factor
        df.loc[date, target_notional_cols] = scaled_target_notionals
        df.loc[date, 'total_target_notional'] = df.loc[date, target_notional_cols].sum()

        ## Calculate revised target size
        for ticker in ticker_list:
            df.loc[date, f'{ticker}_target_size'] = (df.loc[date, f'{ticker}_target_notional'] /
                                                     df.loc[date, f'{ticker}_t_1_close'])

    return df


def get_atr_daily_portfolio_positions(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                      rolling_donchian_window, rolling_atr_window, risk_per_trade, stop_loss_multiple,
                                      initial_capital, cash_buffer_percentage, transaction_cost_est=0.001,
                                      passive_trade_rate=0.05, annual_trading_days=365, use_specific_start_date=False,
                                      signal_start_date=None):

    ## Reorder dataframe columns
    col_list = []
    for ticker in ticker_list:
        df[f'{ticker}_actual_size'] = 0.0
        df[f'{ticker}_actual_position_notional'] = 0.0
        df[f'{ticker}_short_sale_proceeds'] = 0.0
        df[f'{ticker}_actual_position_entry_price'] = 0.0
        df[f'{ticker}_actual_position_exit_price'] = 0.0
        df[f'{ticker}_account_risk'] = 0.0
        df[f'{ticker}_trade_risk'] = 0.0
        df[f'{ticker}_target_notional'] = 0.0
        df[f'{ticker}_target_size'] = 0.0
        df[f'{ticker}_event'] = np.nan
    ord_cols = reorder_columns_by_ticker(df.columns, ticker_list)
    df = df[ord_cols]

    ## Portfolio Level Cash and Positions are all set to 0
    df['available_cash'] = 0.0
    df['count_of_positions'] = 0.0
    df['total_actual_position_notional'] = 0.0
    df['total_target_notional'] = 0.0
    df['total_portfolio_value'] = 0.0
    df['total_portfolio_value_upper_limit'] = 0.0
    df['target_notional_scaling_factor'] = 1.0

    ## Cash and the Total Portfolio Value on Day 1 is the initial capital for the strategy
    if use_specific_start_date:
        start_index_position = df.index.get_loc(signal_start_date)
    else:
        start_index_position = 0
    df['available_cash'][start_index_position] = initial_capital
    df['total_portfolio_value'][start_index_position] = initial_capital

    ## Identify Daily Positions starting from day 2
    for date in df.index[start_index_position + 1:]:
        previous_date = df.index[df.index.get_loc(date) - 1]

        ## Start the day with the available cash from yesterday
        df['available_cash'].loc[date] = df['available_cash'].loc[previous_date]
        total_portfolio_value_upper_limit = (df['total_portfolio_value'].loc[previous_date] *
                                             (1 - cash_buffer_percentage))
        df['total_portfolio_value_upper_limit'].loc[date] = total_portfolio_value_upper_limit

        ## Calculate the target notional by ticker
        df = calculate_atr_target_notional(df, date, ticker_list, rolling_atr_window, risk_per_trade,
                                           stop_loss_multiple)

        ## Get the daily positions
        df = get_daily_positions_and_portfolio_cash(df, date, ticker_list, fast_mavg, mavg_stepsize, slow_mavg,
                                                    rolling_donchian_window, transaction_cost_est, passive_trade_rate)

        ## Calculate Portfolio Value and Available Cash
        # df = calculate_portfolio_cash_and_market_value_per_day(df, date, ticker_list, transaction_cost_est,
        #                                                        passive_trade_rate)

    return df


# Average True Range Position Sizing Strategy for a Trend Following Signal
def apply_atr_position_sizing_strategy(start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                       rolling_donchian_window, long_only=False, rolling_atr_window=20,
                                       risk_per_trade=0.02, stop_loss_multiple=1, initial_capital=15000,
                                       transaction_cost_est=0.001, passive_trade_rate=0.05, use_coinbase_data=True,
                                       rolling_sharpe_window=50, cash_buffer_percentage=0.10, annual_trading_days=365,
                                       use_specific_start_date=False, signal_start_date=None,
                                       price_or_returns_calc='price'):
    ## Generate Trend Signal for all tickers
    df_trend = tf.get_trend_donchian_signal_for_portfolio(start_date=start_date, end_date=end_date,
                                                          ticker_list=ticker_list, fast_mavg=fast_mavg,
                                                          slow_mavg=slow_mavg,
                                                          mavg_stepsize=mavg_stepsize,
                                                          rolling_donchian_window=rolling_donchian_window,
                                                          long_only=long_only, use_coinbase_data=use_coinbase_data,
                                                          price_or_returns_calc=price_or_returns_calc)

    ## Generate Target Position Size and Notional for all tickers
    df_atr_ticker = get_atr_by_ticker(start_date=start_date, end_date=end_date, ticker_list=ticker_list,
                                      rolling_atr_window=rolling_atr_window)

    ## Merge the Trend and Target Position Dataframes
    df = pd.merge(df_trend, df_atr_ticker, left_index=True, right_index=True, how='left')

    ## Get Daily Positions
    df = get_atr_daily_portfolio_positions(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                           rolling_donchian_window, rolling_atr_window, risk_per_trade,
                                           stop_loss_multiple, initial_capital, cash_buffer_percentage,
                                           transaction_cost_est, passive_trade_rate, annual_trading_days,
                                           use_specific_start_date, signal_start_date)

    ## Calculate Portfolio Performance
    df = calculate_portfolio_returns(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, rolling_donchian_window,
                                     rolling_sharpe_window)

    return df


def calculate_standard_deviation(start_date, end_date, ticker, rolling_std_window):
    ## Get Close Prices from Coinbase
    df = cn.save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=False, end_date=end_date,
                                                        save_to_file=False)
    df = (df[['close']].rename(columns={'close': ticker}))
    df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    df[f'{ticker}_t_1_close'] = df[ticker].shift(1)

    ## Calculate Annualized Standard Deviation
    df[f'{ticker}_pct_returns'] = df[f'{ticker}'].pct_change()
    df[f'{ticker}_{rolling_std_window}_std_dev'] = df[f'{ticker}_pct_returns'].rolling(window=rolling_std_window).std()

    ## Shift by 1 to avoid look-ahead bias
    df[f'{ticker}_{rolling_std_window}_std_dev'] = df[f'{ticker}_{rolling_std_window}_std_dev'].shift(1)

    return df


## Position Sizing Strategy will calculate Account Risk based on risk tolerance per trade and Trade Risk based on Standard Deviation or stop loss multiple
## Position Size will be determined by dividing the Account Risk by the Trade Risk
def get_std_by_ticker(start_date, end_date, ticker_list, rolling_std_window):
    std_list = []
    for ticker in ticker_list:
        df_std = calculate_standard_deviation(start_date=start_date, end_date=end_date, ticker=ticker,
                                              rolling_std_window=rolling_std_window)

        ## Calculate Target Size and Notional based on the Standard Deviation
        std_cols = [f'{ticker}_t_1_close', f'{ticker}_{rolling_std_window}_std_dev']
        std_list.append(df_std[std_cols])

    df_std_ticker = pd.concat(std_list, axis=1)

    return df_std_ticker


def calculate_std_target_notional(df, date, ticker_list, rolling_std_window, risk_per_trade, stop_loss_multiple,
                                  t_1_portfolio_value_col='total_portfolio_value_upper_limit'):

    previous_date = df.index[df.index.get_loc(date) - 1]

    ## Calculate the target notional by ticker
    for ticker in ticker_list:
        std_col = f'{ticker}_{rolling_std_window}_std_dev'
        t_1_close_col = f'{ticker}_t_1_close'
        df.loc[date, f'{ticker}_account_risk'] = (df.loc[date, t_1_portfolio_value_col] * risk_per_trade)
        df.loc[date, f'{ticker}_trade_risk'] = (df.loc[date, t_1_close_col] * df.loc[previous_date, std_col] *
                                                stop_loss_multiple)
        df.loc[date, f'{ticker}_target_size'] = (df.loc[date, f'{ticker}_account_risk'] /
                                                 df.loc[date, f'{ticker}_trade_risk'])
        df.loc[date, f'{ticker}_target_notional'] = (df.loc[date, f'{ticker}_target_size'] *
                                                     df.loc[date, t_1_close_col])

    ## Calculate the total target notional for the day
    target_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
    total_target_notional = df.loc[date, target_notional_cols].sum()
    df.loc[date, 'total_target_notional'] = total_target_notional

    ## Check if the Target Notional Positions need to be scaled
    if total_target_notional > df.loc[date, t_1_portfolio_value_col]:
        scaling_factor = df.loc[date, t_1_portfolio_value_col] / total_target_notional
        df.loc[date, 'target_notional_scaling_factor'] = scaling_factor
        daily_target_notionals = df.loc[date, target_notional_cols]
        scaled_target_notionals = daily_target_notionals * scaling_factor
        df.loc[date, target_notional_cols] = scaled_target_notionals
        df.loc[date, 'total_target_notional'] = df.loc[date, target_notional_cols].sum()

        ## Calculate revised target size
        for ticker in ticker_list:
            df.loc[date, f'{ticker}_target_size'] = (df.loc[date, f'{ticker}_target_notional'] /
                                                     df.loc[date, f'{ticker}_t_1_close'])

    return df


def get_std_daily_portfolio_positions(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, rolling_donchian_window,
                                      rolling_std_window, risk_per_trade, stop_loss_multiple, initial_capital,
                                      cash_buffer_percentage, transaction_cost_est=0.001, passive_trade_rate=0.05,
                                      annual_trading_days=365, use_specific_start_date=False, signal_start_date=None):

    ## Reorder dataframe columns
    col_list = []
    for ticker in ticker_list:
        df[f'{ticker}_actual_size'] = 0.0
        df[f'{ticker}_actual_position_notional'] = 0.0
        df[f'{ticker}_short_sale_proceeds'] = 0.0
        df[f'{ticker}_actual_position_entry_price'] = 0.0
        df[f'{ticker}_actual_position_exit_price'] = 0.0
        df[f'{ticker}_account_risk'] = 0.0
        df[f'{ticker}_trade_risk'] = 0.0
        df[f'{ticker}_target_notional'] = 0.0
        df[f'{ticker}_target_size'] = 0.0
        df[f'{ticker}_event'] = np.nan
    ord_cols = reorder_columns_by_ticker(df.columns, ticker_list)
    df = df[ord_cols]

    ## Portfolio Level Cash and Positions are all set to 0
    df['available_cash'] = 0.0
    df['count_of_positions'] = 0.0
    df['total_actual_position_notional'] = 0.0
    df['total_target_notional'] = 0.0
    df['total_portfolio_value'] = 0.0
    df['total_portfolio_value_upper_limit'] = 0.0
    df['target_notional_scaling_factor'] = 1.0

    ## Cash and the Total Portfolio Value on Day 1 is the initial capital for the strategy
    if use_specific_start_date:
        start_index_position = df.index.get_loc(signal_start_date)
    else:
        start_index_position = 0
    df['available_cash'][start_index_position] = initial_capital
    df['total_portfolio_value'][start_index_position] = initial_capital

    ## Identify Daily Positions starting from day 2
    for date in df.index[start_index_position + 1:]:
        previous_date = df.index[df.index.get_loc(date) - 1]

        ## Start the day with the available cash from yesterday
        df['available_cash'].loc[date] = df['available_cash'].loc[previous_date]
        total_portfolio_value_upper_limit = (df['total_portfolio_value'].loc[previous_date] *
                                             (1 - cash_buffer_percentage))
        df['total_portfolio_value_upper_limit'].loc[date] = total_portfolio_value_upper_limit

        ## Calculate the target notional by ticker
        df = calculate_std_target_notional(df, date, ticker_list, rolling_std_window, risk_per_trade,
                                           stop_loss_multiple)

        ## Get the daily positions
        df = get_daily_positions_and_portfolio_cash(df, date, ticker_list, fast_mavg, mavg_stepsize, slow_mavg,
                                                    rolling_donchian_window, transaction_cost_est, passive_trade_rate)

        ## Calculate Portfolio Value and Available Cash
        # df = calculate_portfolio_cash_and_market_value_per_day(df, date, ticker_list, transaction_cost_est,
        #                                                        passive_trade_rate)

    return df


def apply_std_position_sizing_strategy(start_date, end_date, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                       rolling_donchian_window, long_only=False, rolling_std_window=20,
                                       risk_per_trade=0.02, stop_loss_multiple=1, initial_capital=15000,
                                       transaction_cost_est=0.001, passive_trade_rate=0.05, use_coinbase_data=True,
                                       rolling_sharpe_window=50, cash_buffer_percentage=0.10, annual_trading_days=365,
                                       use_specific_start_date=False, signal_start_date=None,
                                       price_or_returns_calc='price'):
    ## Generate Trend Signal for all tickers
    df_trend = tf.get_trend_donchian_signal_for_portfolio(start_date=start_date, end_date=end_date,
                                                          ticker_list=ticker_list, fast_mavg=fast_mavg,
                                                          slow_mavg=slow_mavg,
                                                          mavg_stepsize=mavg_stepsize,
                                                          rolling_donchian_window=rolling_donchian_window,
                                                          long_only=long_only, use_coinbase_data=use_coinbase_data,
                                                          price_or_returns_calc=price_or_returns_calc)

    ## Generate Target Position Size and Notional for all tickers
    df_std_ticker = get_std_by_ticker(start_date=start_date, end_date=end_date, ticker_list=ticker_list,
                                      rolling_std_window=rolling_std_window)

    ## Merge the Trend and Target Position Dataframes
    df = pd.merge(df_trend, df_std_ticker, left_index=True, right_index=True, how='left')

    ## Get Daily Positions
    df = get_std_daily_portfolio_positions(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize,
                                           rolling_donchian_window, rolling_std_window, risk_per_trade,
                                           stop_loss_multiple, initial_capital, cash_buffer_percentage,
                                           transaction_cost_est, passive_trade_rate, annual_trading_days,
                                           use_specific_start_date, signal_start_date)

    ## Calculate Portfolio Performance
    df = calculate_portfolio_returns(df, ticker_list, fast_mavg, slow_mavg, mavg_stepsize, rolling_donchian_window,
                                     rolling_sharpe_window)

    return df


