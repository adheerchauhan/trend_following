import numpy as np
import pandas as pd
from scipy import stats
from utils import coinbase_utils as cn
import matplotlib.pyplot as plt

def estimate_fee_per_trade(passive_trade_rate=0.5):
    ## Maker/Taker Fee based on lowest tier at Coinbase
    maker_fee = 0.006  # 0.6%
    taker_fee = 0.012  # 1.20%
    proportion_maker = passive_trade_rate
    proportion_taker = (1 - passive_trade_rate)

    average_fee_per_trade = (maker_fee * proportion_maker) + (taker_fee * proportion_taker)

    return average_fee_per_trade


def calculate_compounded_cumulative_returns(df, strategy_daily_return_col, strategy_trade_count_col,
                                            include_transaction_costs_and_fees=True, transaction_cost_est=0.001,
                                            passive_trade_rate=0.5):

    # Calculate cumulative return
    df['strategy_cumulative_return'] = (1 + df[strategy_daily_return_col]).cumprod() - 1

    # Calculate the total cumulative return at the end of the period
    total_cumulative_return = df['strategy_cumulative_return'].iloc[-1]

    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        num_trades = np.abs(df[strategy_trade_count_col]).sum()
        total_transaction_cost = num_trades * transaction_cost_est
        total_fee_cost = num_trades * average_fee_per_trade
        total_cumulative_return = total_cumulative_return - (total_transaction_cost + total_fee_cost)

    return total_cumulative_return


def calculate_CAGR(df, strategy_daily_return_col, strategy_trade_count_col, annual_trading_days=252,
                   include_transaction_costs_and_fees=True, transaction_cost_est=0.001, passive_trade_rate=0.5):

    # Calculate cumulative return
    total_cumulative_return = calculate_compounded_cumulative_returns(
        df, strategy_daily_return_col=strategy_daily_return_col, strategy_trade_count_col=strategy_trade_count_col,
        include_transaction_costs_and_fees=include_transaction_costs_and_fees,
        transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate)

    # Calculate CAGR
    # Calculate the number of periods (days)
    num_periods = len(df)

    # Convert the number of periods to years (assuming daily data, 252 trading days per year)
    trading_days_per_year = annual_trading_days
    num_years = num_periods / trading_days_per_year
    annualized_return = (1 + total_cumulative_return) ** (1 / num_years) - 1

    return annualized_return


def calculate_drawdown(df, strategy_daily_return_col, strategy_trade_count_col, include_transaction_costs_and_fees=True,
                       transaction_cost_est=0.001, passive_trade_rate=0.5):
    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        adjusted_daily_returns = df[strategy_daily_return_col] - (
                    np.abs(df[strategy_trade_count_col]) * (transaction_cost_est + average_fee_per_trade))
        df['equity_curve'] = (1 + adjusted_daily_returns).cumprod()
    else:
        df['equity_curve'] = (1 + df[strategy_daily_return_col]).cumprod()
    df[f'equity_curve_cum_max'] = df['equity_curve'].cummax()
    df[f'drawdown'] = df['equity_curve'] - df[f'equity_curve_cum_max']
    df[f'drawdown_pct'] = df[f'drawdown'] / df[f'equity_curve_cum_max']
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # Calculate maximum drawdown
    max_drawdown = df[f'drawdown_pct'].min()

    # Calculate maximum drawdown duration
    df['End'] = df.index
    df['Start'] = df[f'equity_curve_cum_max'].ne(df[f'equity_curve_cum_max'].shift(1)).cumsum()
    df[f'equity_curve_DDDuration'] = df.groupby('Start')['End'].transform(lambda x: x.max() - x.min())
    max_drawdown_duration = df[f'equity_curve_DDDuration'].max()

    # Drop NaN values for better display
    #     df = df.dropna(inplace=True)
    return df, max_drawdown, max_drawdown_duration


def calculate_calmar_ratio(df, strategy_daily_return_col, strategy_trade_count_col, annual_trading_days=252,
                           include_transaction_costs_and_fees=True,
                           transaction_cost_est=0.001, passive_trade_rate=0.5):
    # Calculate CAGR
    cagr = calculate_CAGR(df, strategy_daily_return_col, strategy_trade_count_col,
                          annual_trading_days=annual_trading_days,
                          include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                          transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate)

    # Calculate Max Drawdown
    _, max_drawdown, _ = calculate_drawdown(df, strategy_daily_return_col, strategy_trade_count_col,
                                            include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                            transaction_cost_est=transaction_cost_est,
                                            passive_trade_rate=passive_trade_rate)

    # Calculate Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown)

    return calmar_ratio


def calculate_hit_rate(df, strategy_daily_return_col, strategy_trade_count_col, include_transaction_costs_and_fees=True,
                       transaction_cost_est=0.001, passive_trade_rate=0.5):
    # Identify profitable trades (daily returns > 0)
    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        df['profitable_trade'] = (df[strategy_daily_return_col] - np.abs(df[strategy_trade_count_col]) * (
                transaction_cost_est + average_fee_per_trade)) > 0
    else:
        df['profitable_trade'] = df[strategy_daily_return_col] > 0

    # Calculate hit rate
    total_trades = df['profitable_trade'].count()
    profitable_trades = df['profitable_trade'].sum()
    hit_rate = profitable_trades / total_trades

    return hit_rate


def calculate_t_stat(df, strategy_daily_return_col, strategy_trade_count_col, include_transaction_costs_and_fees=True,
                     transaction_cost_est=0.001,
                     passive_trade_rate=0.5):
    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        mean_return = (df[strategy_daily_return_col] - np.abs(df[strategy_trade_count_col]) * (
                    transaction_cost_est + average_fee_per_trade)).mean()
        std_dev_return = (df[strategy_daily_return_col] - np.abs(df[strategy_trade_count_col]) * (
                    transaction_cost_est + average_fee_per_trade)).std()
    else:
        mean_return = df[strategy_daily_return_col].mean()
        std_dev_return = df[strategy_daily_return_col].std()

    num_days = len(df[strategy_daily_return_col])
    t_stat = mean_return / (std_dev_return / np.sqrt(num_days))
    p_value = stats.t.sf(np.abs(t_stat), df=num_days - 1) * 2

    return t_stat, p_value


def calculate_annualized_std_dev(df, strategy_daily_return_col, strategy_trade_count_col, annual_trading_days,
                                 include_transaction_costs_and_fees=True, transaction_cost_est=0.001,
                                 passive_trade_rate=0.5):
    # Filter on days with active exposure
    df = df[df[strategy_trade_count_col] != 0]

    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        annualized_std_dev = (df[strategy_daily_return_col] - np.abs(df[strategy_trade_count_col]) *
                              (transaction_cost_est + average_fee_per_trade)).std() * np.sqrt(annual_trading_days)
    else:
        annualized_std_dev = df[strategy_daily_return_col].std() * np.sqrt(annual_trading_days)

    return annualized_std_dev


def calculate_sharpe_ratio(df, strategy_daily_return_col, strategy_trade_count_col, annual_trading_days=252, annual_rf=0.05,
                           include_transaction_costs_and_fees=True, transaction_cost_est=0.001, passive_trade_rate=0.5):

    daily_rf = (1 + annual_rf) ** (1/annual_trading_days) - 1
    if include_transaction_costs_and_fees:
        average_fee_per_trade = estimate_fee_per_trade(passive_trade_rate=passive_trade_rate)
        daily_cost = np.abs(df[strategy_trade_count_col]) * (transaction_cost_est + average_fee_per_trade)
    else:
        daily_cost = 0.0

    excess_return = df[strategy_daily_return_col] - daily_cost - daily_rf
    average_daily_return = excess_return.mean()
    std_dev_daily_return = excess_return.std()

    daily_sharpe_ratio = average_daily_return/std_dev_daily_return
    annualized_sharpe_ratio = daily_sharpe_ratio * np.sqrt(annual_trading_days)

    return annualized_sharpe_ratio


def calculate_risk_and_performance_metrics(df, strategy_daily_return_col, strategy_trade_count_col, annual_rf=0.05,
                                           annual_trading_days=252, include_transaction_costs_and_fees=True,
                                           transaction_cost_est=0.001,
                                           passive_trade_rate=0.05):
    # Calculate CAGR
    annualized_return = calculate_CAGR(df, strategy_daily_return_col=strategy_daily_return_col,
                                       strategy_trade_count_col=strategy_trade_count_col,
                                       annual_trading_days=annual_trading_days,
                                       include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                       transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate)

    # Calculate Annualized Sharpe Ratio
    annualized_sharpe_ratio = calculate_sharpe_ratio(df, strategy_daily_return_col=strategy_daily_return_col,
                                                     strategy_trade_count_col=strategy_trade_count_col,
                                                     annual_trading_days=annual_trading_days, annual_rf=annual_rf,
                                                     include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                                     transaction_cost_est=transaction_cost_est,
                                                     passive_trade_rate=passive_trade_rate)

    # Calculate Calmar Ratio
    calmar_ratio = calculate_calmar_ratio(df, strategy_daily_return_col=strategy_daily_return_col,
                                          strategy_trade_count_col=strategy_trade_count_col,
                                          annual_trading_days=annual_trading_days,
                                          include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                          transaction_cost_est=transaction_cost_est,
                                          passive_trade_rate=passive_trade_rate)

    # Calculate Annualized Standard Deviation
    annualized_std_dev = calculate_annualized_std_dev(
        df, strategy_daily_return_col=strategy_daily_return_col,
        strategy_trade_count_col=strategy_trade_count_col,
        annual_trading_days=annual_trading_days,
        include_transaction_costs_and_fees=include_transaction_costs_and_fees,
        transaction_cost_est=transaction_cost_est,
        passive_trade_rate=passive_trade_rate)

    # Calculate Max Drawdown
    df, max_drawdown, max_drawdown_duration = calculate_drawdown(
        df,
        strategy_daily_return_col=strategy_daily_return_col,
        strategy_trade_count_col=strategy_trade_count_col,
        include_transaction_costs_and_fees=include_transaction_costs_and_fees,
        transaction_cost_est=transaction_cost_est,
        passive_trade_rate=passive_trade_rate)

    # Calculate Hit Rate
    hit_rate = calculate_hit_rate(df, strategy_daily_return_col=strategy_daily_return_col,
                                  strategy_trade_count_col=strategy_trade_count_col,
                                  include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                  transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate)

    # Calculate T-Stat and P-Value
    t_stat, p_value = calculate_t_stat(df, strategy_daily_return_col=strategy_daily_return_col,
                                       strategy_trade_count_col=strategy_trade_count_col,
                                       include_transaction_costs_and_fees=include_transaction_costs_and_fees,
                                       transaction_cost_est=transaction_cost_est, passive_trade_rate=passive_trade_rate)

    # Count of trades
    total_num_trades = np.abs(df[strategy_trade_count_col]).sum()

    performance_metrics = {'annualized_return': annualized_return,
                           'annualized_sharpe_ratio': annualized_sharpe_ratio,
                           'calmar_ratio': calmar_ratio,
                           'annualized_std_dev': annualized_std_dev,
                           'max_drawdown': max_drawdown,
                           'max_drawdown_duration': max_drawdown_duration,
                           'hit_rate': hit_rate,
                           't_statistic': t_stat,
                           'p_value': p_value,
                           'trade_count': total_num_trades}

    return performance_metrics


def rolling_sharpe_ratio(df, window, strategy_daily_return_col, strategy_trade_count_col, min_trade_count=100,
                         **kwargs):
    def sharpe_on_window(window_df):
        # Calculate the Sharpe ratio on the windowed data frame
        if np.abs(window_df[strategy_trade_count_col]).sum() < min_trade_count or window_df[strategy_daily_return_col].std() == 0:
            return 0  # Return 0 Sharpe ratio if there are no trades or no variation in returns
        else:
            return calculate_sharpe_ratio(window_df, strategy_daily_return_col, strategy_trade_count_col, **kwargs)

    # Apply the function over a rolling window and return as a Series (not a full DataFrame)
    rolling_sharpe = df[strategy_daily_return_col].rolling(window=window).apply(
        lambda x: sharpe_on_window(df.loc[x.index]), raw=False
    )

    return rolling_sharpe


def calculate_asset_level_returns(df, end_date, ticker_list):
    ## Check if data is available for all the tickers
    date_list = cn.coinbase_start_date_by_ticker_dict
    ticker_list = [ticker for ticker in ticker_list if pd.Timestamp(date_list[ticker]).date() < end_date]

    for ticker in ticker_list:
        df[f'{ticker}_daily_pnl'] = (df[f'{ticker}_actual_position_size'] * df[f'{ticker}_open'].diff().shift(-1))
        df[f'{ticker}_daily_pct_returns'] = (df[f'{ticker}_daily_pnl'] / df[f'total_portfolio_value'].shift(1)).fillna(
            0)
        df[f'{ticker}_position_count'] = np.where((df[f'{ticker}_actual_position_notional'] != 0), 1,
                                                  0)  ## This is not entirely accurate
    return df


def plot_daily_returns_bubble(df, return_col='daily_return'):

    df['color'] = df[return_col].apply(lambda x: 'green' if x >= 0 else 'red')
    df['size'] = df[return_col].abs() * 1000  # scale for visibility

    plt.figure(figsize=(14, 6))
    plt.scatter(df.index, df[return_col],
                c=df['color'],
                s=df['size'],
                alpha=0.6, edgecolors='k')

    plt.title('Daily Returns Bubble Plot')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return
