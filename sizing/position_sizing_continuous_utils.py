# Import all the necessary modules
import pandas as pd
import numpy as np
from portfolio import strategy_performance as perf
from sizing import position_sizing_binary_utils as size_bin
from strategy_signal import trend_following_signal as tf


def get_average_true_range_portfolio(start_date, end_date, ticker_list, rolling_atr_window=20,
                                     price_or_returns_calc='price', use_coinbase_data=True, use_saved_files=False,
                                     saved_file_end_date='2025-06-30'):

    atr_list = []
    for ticker in ticker_list:
        atr_cols = [f'{ticker}_{rolling_atr_window}_avg_true_range_{price_or_returns_calc}']
        df_atr = size_bin.calculate_average_true_range(start_date=start_date, end_date=end_date, ticker=ticker,
                                                       price_or_returns_calc=price_or_returns_calc,
                                                       rolling_atr_window=rolling_atr_window,
                                                       use_coinbase_data=use_coinbase_data,
                                                       use_saved_files=use_saved_files,
                                                       saved_file_end_date=saved_file_end_date)
        atr_list.append(df_atr[atr_cols])

    df_atr_concat = pd.concat(atr_list, axis=1)

    return df_atr_concat


# Below we calculate the number of risk units deployed per ticker per day
# Get Volatility Adjusted Trend Signal for Target Volatility Strategy
def get_volatility_adjusted_trend_signal_continuous(df, ticker_list, volatility_window, annual_trading_days=365):

    ticker_signal_dict = {}
    final_cols = []
    for ticker in ticker_list:
        trend_signal_col = f'{ticker}_final_signal'
        final_weighted_additive_signal_col = f'{ticker}_final_weighted_additive_signal'
        annualized_volatility_col = f'{ticker}_annualized_volatility_{volatility_window}'
        vol_adj_trend_signal_col = f'{ticker}_vol_adjusted_trend_signal'

        ## Calculate Position Volatility Adjusted Trend Signal
        df[f'{ticker}_t_1_close'] = df[f'{ticker}_close'].shift(1)
        df = tf.get_returns_volatility(df, vol_range_list=[volatility_window], close_px_col=f'{ticker}_t_1_close')
        df[annualized_volatility_col] = (df[f'{ticker}_t_1_close_volatility_{volatility_window}'] *
                                         np.sqrt(annual_trading_days))
        df[vol_adj_trend_signal_col] = (df[trend_signal_col] / df[annualized_volatility_col])
        df[vol_adj_trend_signal_col] = df[vol_adj_trend_signal_col].fillna(0)
        trend_cols = [f'{ticker}_close', f'{ticker}_open', f'{ticker}_t_1_close', f'{ticker}_t_1_close_pct_returns',
                      trend_signal_col, final_weighted_additive_signal_col, annualized_volatility_col,
                      vol_adj_trend_signal_col]
        final_cols.append(trend_cols)
        ticker_signal_dict[ticker] = df[trend_cols]
    df_signal = pd.concat(ticker_signal_dict, axis=1)

    ## Assign new column names to the dataframe
    df_signal.columns = df_signal.columns.to_flat_index()
    final_cols = [item for sublist in final_cols for item in sublist]
    df_signal.columns = final_cols

    return df_signal


def get_target_volatility_position_sizing(df, cov_matrix, date, ticker_list, daily_target_volatility,
                                          total_portfolio_value_upper_limit):

    ## Scale weights of positions to ensure the portfolio is in line with the target volatility
    unscaled_weight_cols = [f'{ticker}_vol_adjusted_trend_signal' for ticker in ticker_list]
    scaled_weight_cols = [f'{ticker}_target_vol_normalized_weight' for ticker in ticker_list]
    target_notional_cols = [f'{ticker}_target_notional' for ticker in ticker_list]
    t_1_price_cols = [f'{ticker}_t_1_close' for ticker in ticker_list]

    if date not in df.index or date not in cov_matrix.index:
        raise ValueError(f"Date {date} not found in DataFrame or covariance matrix index.")

    ## Iterate through each day and calculate the covariance matrix, portfolio volatility and scaling factors
    daily_weights = df.loc[date, unscaled_weight_cols].values
    daily_cov_matrix = cov_matrix.loc[date].values
    daily_portfolio_volatility = size_bin.calculate_portfolio_volatility(daily_weights, daily_cov_matrix)
    df.loc[date, 'daily_portfolio_volatility'] = daily_portfolio_volatility
    if daily_portfolio_volatility > 0:
        vol_scaling_factor = daily_target_volatility / daily_portfolio_volatility
    else:
        vol_scaling_factor = 0

    ## Apply Scaling Factor with No Leverage
    gross_weight_sum = np.sum(np.abs(daily_weights))
    cash_scaling_factor = 1.0 / np.maximum(gross_weight_sum, 1e-12)            # ∑ w ≤ 1  (long‑only)
    final_scaling_factor = min(vol_scaling_factor, cash_scaling_factor)

    df.loc[date, 'target_vol_scaling_factor'] = vol_scaling_factor
    df.loc[date, 'cash_scaling_factor'] = cash_scaling_factor
    df.loc[date, 'final_scaling_factor'] = final_scaling_factor

    # Scale the weights to target volatility
    scaled_weights = daily_weights * final_scaling_factor
    df.loc[date, scaled_weight_cols] = scaled_weights

    ## Calculate the target notional and size
    target_notionals = scaled_weights * total_portfolio_value_upper_limit
    df.loc[date, target_notional_cols] = target_notionals
    target_sizes = target_notionals / df.loc[date, t_1_price_cols].values

    for i, ticker in enumerate(ticker_list):
        df.loc[date, f'{ticker}_target_size'] = target_sizes[i]

    total_target_notional = target_notionals.sum()
    df.loc[date, 'total_target_notional'] = total_target_notional

    return df


def get_cash_adjusted_desired_positions(df, date, previous_date, ticker_list, cash_buffer_percentage,
                                        transaction_cost_est, passive_trade_rate,
                                        notional_threshold_pct=0.10, min_trade_notional_abs=10):
    desired_positions = {}
    cash_debit = 0.0  # buys + fees
    cash_credit = 0.0  # sells - fees
    available_cash = df['available_cash'].loc[date] * (1 - cash_buffer_percentage)

    ## Estimated Transaction Costs and Fees
    est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))

    for ticker in ticker_list:
        ## Calculate the cash need from all new target positions
        target_notional = df[f'{ticker}_target_notional'].loc[date]
        current_notional = df[f'{ticker}_actual_position_size'].loc[previous_date] * df[f'{ticker}_open'].loc[date]
        new_trade_notional = target_notional - current_notional
        trade_fees = abs(new_trade_notional) * est_fees

        ## Calculate notional difference to determine if a trade is warranted
        notional_threshold = notional_threshold_pct * abs(target_notional)
        notional_floors_list = [
            notional_threshold, min_trade_notional_abs
        ]
        notional_floor = max(notional_floors_list)
        if abs(new_trade_notional) > notional_floor:
            desired_positions[ticker] = {'new_trade_notional': new_trade_notional,
                                         'trade_fees': trade_fees}
        else:
            desired_positions[ticker] = {'new_trade_notional': 0,
                                         'trade_fees': 0}

        if new_trade_notional >= 0:
            ## Buys
            cash_debit = cash_debit + new_trade_notional
        else:
            ## Sells
            net_trade_notional = new_trade_notional + trade_fees
            cash_credit = cash_credit + abs(net_trade_notional)

    net_cash_need = cash_debit - cash_credit
    if net_cash_need > available_cash + 1e-6:
        cash_shrink_factor = available_cash / net_cash_need  # 0 < shrink < 1
    else:
        cash_shrink_factor = 1.0

    df[f'cash_shrink_factor'] = cash_shrink_factor

    return desired_positions, cash_shrink_factor


def get_daily_positions_and_portfolio_cash(df, date, previous_date, desired_positions, cash_shrink_factor, ticker_list,
                                           rolling_atr_window, atr_multiplier, transaction_cost_est,
                                           passive_trade_rate, cooldown_counter_threshold):
    epsilon = 1e-6  # ignore micro-dust

    ## Define New, Add, Trim, No Change and Exit positions
    for ticker in ticker_list:
        ## Define Column Names
        open_position_notional_col = f'{ticker}_open_position_notional'
        open_position_size_col = f'{ticker}_open_position_size'
        actual_position_notional_col = f'{ticker}_actual_position_notional'
        actual_position_size_col = f'{ticker}_actual_position_size'
        short_sale_proceeds_col = f'{ticker}_short_sale_proceeds'
        open_price_col = f'{ticker}_open'
        cooldown_counter_col = f'{ticker}_cooldown_counter'
        stopout_flag_col = f'{ticker}_stopout_flag'
        event_col = f'{ticker}_event'

        ## Roll Positions from Previous Day
        df[open_position_size_col].loc[date] = df[actual_position_size_col].loc[previous_date]
        df[open_position_notional_col].loc[date] = df[open_position_size_col].loc[date] * df[open_price_col].loc[date]
        df[short_sale_proceeds_col].loc[date] = df[short_sale_proceeds_col].loc[previous_date]
        df[cooldown_counter_col].loc[date] = df[cooldown_counter_col].loc[previous_date]
        df[stopout_flag_col].loc[date] = df[stopout_flag_col].loc[previous_date]

        ## Define Variables
        new_trade_notional = desired_positions[ticker]['new_trade_notional'] * cash_shrink_factor
        trade_fees = desired_positions[ticker]['trade_fees'] * cash_shrink_factor

        ## Assess New and Open Positions and update Stop Loss, Available Cash and Short Sale Proceeds
        df = update_new_and_open_positions(df, ticker, date, new_trade_notional, trade_fees,
                                           rolling_atr_window,
                                           atr_multiplier, transaction_cost_est, passive_trade_rate,
                                           cooldown_counter_threshold)

    ## Calculate End of Day Portfolio Positions
    new_trade_notional_cols = [f'{ticker}_new_position_notional' for ticker in ticker_list]
    actual_position_notional_cols = [f'{ticker}_actual_position_notional' for ticker in ticker_list]
    short_sale_proceeds_cols = [f'{ticker}_short_sale_proceeds' for ticker in ticker_list]
    df['count_of_positions'].loc[date] = df[new_trade_notional_cols].loc[date].ne(0).sum()
    df['total_actual_position_notional'].loc[date] = df[actual_position_notional_cols].loc[date].sum()
    df['total_portfolio_value'].loc[date] = (df['available_cash'].loc[date] +
                                             df[short_sale_proceeds_cols].loc[date].sum() +
                                             df['total_actual_position_notional'].loc[date])

    return df


def register_daily_event(df, ticker, date, new_trade_notional, open_position_notional):

    event_col = f'{ticker}_event'
    if new_trade_notional > 0:
        if open_position_notional < 0:
            if np.isclose(open_position_notional, -new_trade_notional):
                df.at[date, event_col] = 'Close Short Position'
            else:
                df.at[date, event_col] = 'Trim Short Position'
        elif np.isclose(open_position_notional, 0.0):
            df.at[date, event_col] = 'New Long Position'
        elif open_position_notional > 0:
            df.at[date, event_col] = 'Add Long Position'
    elif new_trade_notional < 0:
        if open_position_notional > 0:
            if np.isclose(open_position_notional, -new_trade_notional):
                df.at[date, event_col] = 'Close Long Position'
            else:
                df.at[date, event_col] = 'Trim Long Position'
        elif np.isclose(open_position_notional, 0.0):
            df.at[date, event_col] = 'New Short Position'
        elif open_position_notional < 0:
            df.at[date, event_col] = 'Add Short Position'
    elif np.isclose(new_trade_notional, 0.0):
        if open_position_notional > 0:
            df.at[date, event_col] = 'Open Long Position'
        elif open_position_notional < 0:
            df.at[date, event_col] = 'Open Short Position'
        elif np.isclose(open_position_notional, 0.0):
            df.at[date, event_col] = 'No Position'

    return df


def handle_explicit_position_close(df, ticker, date, open_position_notional, open_price, est_fees):

    new_position_notional_col = f'{ticker}_new_position_notional'
    new_position_size_col = f'{ticker}_new_position_size'
    actual_position_notional_col = f'{ticker}_actual_position_notional'
    actual_position_size_col = f'{ticker}_actual_position_size'
    new_position_entry_exit_price_col = f'{ticker}_new_position_entry_exit_price'
    stop_loss_col = f'{ticker}_stop_loss'
    available_cash_col = 'available_cash'
    short_sale_proceeds_col = f'{ticker}_short_sale_proceeds'

    # Close full position
    new_trade_notional = -open_position_notional
    trade_fees = abs(new_trade_notional) * est_fees
    if new_trade_notional > 0:
        net_trade_notional = new_trade_notional - trade_fees
    else:
        net_trade_notional = new_trade_notional + trade_fees

    # Bookkeeping
    df.at[date, new_position_notional_col] = net_trade_notional
    df.at[date, new_position_size_col] = net_trade_notional / open_price
    df.at[date, new_position_entry_exit_price_col] = open_price
    df.at[date, actual_position_notional_col] = 0
    df.at[date, actual_position_size_col] = 0
    df.at[date, stop_loss_col] = 0

    # Cash & proceeds adjustment
    if open_position_notional > 0:
        ## Closing Long Position
        df.at[date, available_cash_col] += abs(net_trade_notional)
    else:
        ## Closing Short Position (Only Mark to Market goes to Available Cash)
        df.at[date, available_cash_col] = df.at[date, available_cash_col] + (
                    df.at[date, short_sale_proceeds_col] - abs(net_trade_notional))
        df.at[date, short_sale_proceeds_col] = 0

    # Register the event
    df = register_daily_event(df, ticker, date, new_trade_notional, open_position_notional)

    return df, True  # True = position was closed, skip rest of logic


def update_new_and_open_positions(df, ticker, date, new_trade_notional, trade_fees,
                                  rolling_atr_window, atr_multiplier, transaction_cost_est, passive_trade_rate,
                                  cooldown_counter_threshold):

    new_position_notional_col = f'{ticker}_new_position_notional'
    new_position_size_col = f'{ticker}_new_position_size'
    open_position_notional_col = f'{ticker}_open_position_notional'
    open_position_size_col = f'{ticker}_open_position_size'
    actual_position_notional_col = f'{ticker}_actual_position_notional'
    actual_position_size_col = f'{ticker}_actual_position_size'
    new_position_entry_exit_price_col = f'{ticker}_new_position_entry_exit_price'
    target_position_notional_col = f'{ticker}_target_notional'
    target_position_size_col = f'{ticker}_target_size'
    stop_loss_col = f'{ticker}_stop_loss'
    cooldown_counter_col = f'{ticker}_cooldown_counter'
    stopout_flag_col = f'{ticker}_stopout_flag'
    available_cash_col = 'available_cash'
    short_sale_proceeds_col = f'{ticker}_short_sale_proceeds'
    open_price_col = f'{ticker}_open'
    t_1_close_price_col = f'{ticker}_t_1_close'
    atr_col = f'{ticker}_{rolling_atr_window}_avg_true_range_price'
    event_col = f'{ticker}_event'

    ## Update Daily Positions
    # Open Trade Notional
    open_position_size = df[open_position_size_col].loc[date]
    open_price = df[open_price_col].loc[date]
    open_position_notional = df[open_position_notional_col].loc[date]
    t_1_close_price = df[t_1_close_price_col].loc[date]
    atr_value = df[atr_col].loc[date]
    target_notional = df[target_position_notional_col].loc[date]
    target_size = df[target_position_size_col].loc[date]
    est_fees = (transaction_cost_est + perf.estimate_fee_per_trade(passive_trade_rate))
    cooldown_counter = df[cooldown_counter_col].loc[date]
    stopout_flag = df[stopout_flag_col].loc[date]

    ## Implement a Cooldown Counter for Stop Loss Breaches
    if cooldown_counter > 0:
        # Increment cooldown
        df[cooldown_counter_col].loc[date] = df[cooldown_counter_col].loc[date] + 1

        # Cooldown threshold (e.g., 3 bars)
        if df[cooldown_counter_col].loc[date] <= cooldown_counter_threshold:
            # Still cooling down: block re-entry
            df[event_col].loc[date] = 'Stop-Loss Cooldown'
            return df
        else:
            # Cooldown complete: reset
            df[stopout_flag_col].loc[date] = False
            df[cooldown_counter_col].loc[date] = 0

    ## Check for Stop Loss Breach
    if handle_stop_loss_breach(df, ticker, date, est_fees, stop_loss_col, open_price, available_cash_col,
                               short_sale_proceeds_col):
        return df  # Stop loss breach handled, skip remaining logic

    ## Handle an Explicit Close Position
    if np.isclose(target_notional, 0.0) and not np.isclose(open_position_notional, 0.0):
        df, closed = handle_explicit_position_close(df, ticker, date, open_position_notional, open_price, est_fees)
        if closed:
            return df

    # New Trade Notional
    if new_trade_notional >= 0:
        net_trade_notional = new_trade_notional - trade_fees
    else:
        net_trade_notional = new_trade_notional + trade_fees
    df[new_position_notional_col].loc[date] = net_trade_notional
    df[new_position_size_col].loc[date] = net_trade_notional / open_price
    df[new_position_entry_exit_price_col].loc[date] = open_price

    # Actual Positions
    df[actual_position_notional_col].loc[date] = (df[new_position_notional_col].loc[date] +
                                                  df[open_position_notional_col].loc[date])
    df[actual_position_size_col].loc[date] = (df[new_position_size_col].loc[date] +
                                              df[open_position_size_col].loc[date])

    ## Update Trailing Stop Loss
    if df[actual_position_notional_col].loc[date] > 0:
        df[stop_loss_col].loc[date] = open_price - atr_value * atr_multiplier
    elif df[actual_position_notional_col].loc[date] < 0:
        df[stop_loss_col].loc[date] = open_price + atr_value * atr_multiplier
    else:
        df[stop_loss_col].loc[date] = 0

    ## Update Available Cash and Short Sale Proceeds
    if new_trade_notional > 0:
        if open_position_notional < 0:
            if abs(new_trade_notional) <= abs(open_position_notional):
                ## Account for Trim Shorts and Close Shorts
                df[short_sale_proceeds_col].loc[date] = df[short_sale_proceeds_col].loc[date] - abs(net_trade_notional)
            elif abs(new_trade_notional) > abs(open_position_notional):
                ## Account for Open Shorts to New Long
                # Cash Need to Close Short Position
                df[available_cash_col].loc[date] = df[available_cash_col].loc[date] + (
                            open_position_notional + df[short_sale_proceeds_col].loc[date])
                df[short_sale_proceeds_col].loc[date] = 0
                # Cash Need to Open New Position
                df[available_cash_col].loc[date] = df[available_cash_col].loc[date] - (
                            new_trade_notional - abs(open_position_notional))
        else:
            ## Account for New Longs and Add Longs
            df[available_cash_col].loc[date] = df[available_cash_col].loc[date] - abs(new_trade_notional)

    elif new_trade_notional < 0:
        if open_position_notional > 0:
            if abs(new_trade_notional) <= abs(open_position_notional):
                ## Account for Trim Longs and Close Longs
                df[available_cash_col].loc[date] = df[available_cash_col].loc[date] + abs(net_trade_notional)
            elif abs(new_trade_notional) > abs(open_position_notional):
                ## Account for Open Longs to New Shorts
                # Cash Need to Close Long Position
                df[available_cash_col].loc[date] = df[available_cash_col].loc[date] + open_position_notional * (
                            1 - est_fees)
                # Cash Need to Open New Short Position
                df[short_sale_proceeds_col].loc[date] = df[short_sale_proceeds_col].loc[date] + (
                            abs(net_trade_notional) - abs(open_position_notional))
        else:
            ## Account for New Shorts and Add Shorts
            df[short_sale_proceeds_col].loc[date] = df[short_sale_proceeds_col].loc[date] + abs(net_trade_notional)

    ## Register Daily Event
    df = register_daily_event(df, ticker, date, new_trade_notional, open_position_notional)

    return df


def handle_stop_loss_breach(df, ticker, date, est_fees, stop_loss_col, open_price, available_cash_col,
                            short_sale_proceeds_col):

    actual_position_notional_col = f'{ticker}_actual_position_notional'
    actual_position_size_col = f'{ticker}_actual_position_size'
    open_position_notional_col = f'{ticker}_open_position_notional'
    open_position_size_col = f'{ticker}_open_position_size'
    new_position_notional_col = f'{ticker}_new_position_notional'
    new_position_size_col = f'{ticker}_new_position_size'
    new_position_entry_exit_price_col = f'{ticker}_new_position_entry_exit_price'
    event_col = f'{ticker}_event'

    if date == df.index[0]:
        return False  # no previous date to compare

    previous_date = df.index[df.index.get_loc(date) - 1]
    prev_stop_loss = df.at[previous_date, stop_loss_col]
    open_position_notional = df.at[previous_date, actual_position_notional_col]
    open_position_size = df.at[previous_date, actual_position_size_col]

    if open_position_notional > 0 and open_price < prev_stop_loss:
        # Long position breached stop
        new_trade_notional = -(open_position_size * prev_stop_loss)  # -open_position_notional
        net_trade_notional = new_trade_notional * (1 - est_fees)
        df.at[date, actual_position_notional_col] = 0
        df.at[date, actual_position_size_col] = 0
        df.at[date, open_position_notional_col] = 0
        df.at[date, open_position_size_col] = 0
        df.at[date, new_position_notional_col] = new_trade_notional
        df.at[date, new_position_size_col] = new_trade_notional / prev_stop_loss
        df.at[date, new_position_entry_exit_price_col] = prev_stop_loss
        df.at[date, stop_loss_col] = 0
        df.at[date, f'{ticker}_stopout_flag'] = True
        df.at[date, f'{ticker}_cooldown_counter'] = 1
        df.at[date, available_cash_col] = df.at[date, available_cash_col] + abs(net_trade_notional)
        df.at[date, event_col] = 'Close Long Position'
        return True

    elif open_position_notional < 0 and open_price > prev_stop_loss:
        # Short position breached stop
        new_trade_notional = -(open_position_size * prev_stop_loss)  # -open_position_notional
        net_trade_notional = new_trade_notional * (1 - est_fees)
        df.at[date, actual_position_notional_col] = 0
        df.at[date, actual_position_size_col] = 0
        df.at[date, new_position_notional_col] = new_trade_notional
        df.at[date, new_position_size_col] = new_trade_notional / prev_stop_loss
        df.at[date, new_position_entry_exit_price_col] = prev_stop_loss
        df.at[date, stop_loss_col] = 0
        df.at[date, f'{ticker}_stopout_flag'] = True
        df.at[date, f'{ticker}_cooldown_counter'] = 1
        df.at[date, available_cash_col] = df.at[date, available_cash_col] + (
                    df.at[date, short_sale_proceeds_col] - abs(net_trade_notional))
        df.at[date, short_sale_proceeds_col] = 0
        df.at[date, event_col] = 'Close Short Position'
        return True

    return False  # No breach


def get_target_volatility_daily_portfolio_positions(df, ticker_list, initial_capital, rolling_cov_window,
                                                    rolling_atr_window, atr_multiplier, cash_buffer_percentage,
                                                    annualized_target_volatility, transaction_cost_est=0.001,
                                                    passive_trade_rate=0.05, notional_threshold_pct=0.02,
                                                    min_trade_notional_abs=10, cooldown_counter_threshold=3,
                                                    annual_trading_days=365, use_specific_start_date=False,
                                                    signal_start_date=None):

    ## Calculate the covariance matrix for tickers in the portfolio
    returns_cols = [f'{ticker}_t_1_close_pct_returns' for ticker in ticker_list]
    cov_matrix = df[returns_cols].rolling(rolling_cov_window).cov(pairwise=True).dropna()

    ## Delete rows prior to the first available date of the covariance matrix
    cov_matrix_start_date = cov_matrix.index[0][0]
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
    df['final_scaling_factor'] = 1.0

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

        ## Roll Portfolio Value from the Previous Day
        df['total_portfolio_value'].loc[date] = df['total_portfolio_value'].loc[previous_date]

        ## Update Total Portfolio Value Upper Limit based on the Total Portfolio Value
        total_portfolio_value_upper_limit = (df['total_portfolio_value'].loc[date] *
                                             (1 - cash_buffer_percentage))
        df['total_portfolio_value_upper_limit'].loc[date] = total_portfolio_value_upper_limit

        ## Calculate the target notional by ticker
        df = get_target_volatility_position_sizing(df, cov_matrix, date, ticker_list, daily_target_volatility,
                                                   total_portfolio_value_upper_limit)

        ## Adjust Positions for Cash Available
        desired_positions, cash_shrink_factor = get_cash_adjusted_desired_positions(
            df, date, previous_date, ticker_list, cash_buffer_percentage, transaction_cost_est, passive_trade_rate,
            notional_threshold_pct, min_trade_notional_abs)

        ## Get the daily positions
        df = get_daily_positions_and_portfolio_cash(
            df, date, previous_date, desired_positions, cash_shrink_factor, ticker_list, rolling_atr_window,
            atr_multiplier, transaction_cost_est, passive_trade_rate, cooldown_counter_threshold)

    return df
