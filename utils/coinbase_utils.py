from coinbase.rest import RESTClient
import time
import requests.exceptions
from json import dumps
import pandas as pd
import datetime
import os
from requests.exceptions import HTTPError


key_location = f'{os.environ.get('HOME')}/Documents/git/trend_following/cdp_api_key.json'
coinbase_start_date_by_ticker_dict = {
    'BTC-USD': '2016-01-01',
    'ETH-USD': '2016-06-01',
    'SOL-USD': '2021-06-01',
    'LTC-USD': '2016-09-01',
    'DOGE-USD': '2021-06-01',
    'CRO-USD': '2021-11-01',
    'ADA-USD': '2021-03-01',
    'AVAX-USD': '2021-09-01',
    'XRP-USD': '2023-06-01',
    'SHIB-USD': '2021-08-01',
    'LINK-USD': '2019-06-01',
    'UNI-USD': '2020-09-01',
    'DOT-USD': '2021-06-01',
    'FET-USD': '2021-07-01',
    'ALGO-USD': '2019-08-01',
    'DAI-USD': '2020-04-01',
    'AAVE-USD': '2020-12-01',
    'XLM-USD': '2019-02-01',
    'MATIC-USD': '2021-02-01',
    'ATOM-USD': '2020-01-01',
    'MANA-USD': '2021-04-01',
    'OXT-USD': '2019-12-01',
    'KRL-USD': '2021-10-01',
    'AMP-USD': '2021-05-01',
    'REQ-USD': '2021-07-01',
    'SKL-USD': '2021-02-01',
    'GRT-USD': '2020-12-01',
    'MOBILE-USD': '2024-02-01',
    'AIOZ-USD': '2022-02-01',
    'ZRO-USD': '2024-06-01',
    'HNT-USD': '2023-06-01',
    'HONEY-USD': '2024-01-01'
}

def get_coinbase_rest_api_client(key_location):
    client = RESTClient(key_file=key_location)
    return client


def get_portfolio_uuid(client, portfolio_name='Default'):
    portfolio_list = client.get_portfolios()['portfolios']
    if portfolio_name == 'Default':
        portfolio = next((p for p in portfolio_list
                          if p['name'] == 'Default' and not p['deleted']), None)
    else:
        portfolio = next((p for p in portfolio_list
                          if p['name'] == 'Trend Following' and not p['deleted']), None)

    portfolio_uuid = portfolio['uuid']
    return portfolio_uuid


def get_portfolio_breakdown(client, portfolio_uuid):
    # portfolio_uuid = get_portfolio_uuid(client)
    portfolio_list = client.get_portfolio_breakdown(portfolio_uuid).breakdown.spot_positions
    portfolio_data = []

    # Assuming accounts are available directly in a list (e.g., accounts[0] or accounts.accounts)
    for position in portfolio_list:  # Adjust this line based on the actual
        position_info = {
            'asset': position['asset'],
            'account_uuid': position['account_uuid'],
            'asset_uuid': position['asset_uuid'],
            'total_balance_fiat': position['total_balance_fiat'],
            'available_to_trade_fiat': position['available_to_trade_fiat'],
            'total_balance_crypto': position['total_balance_crypto'],
            'allocation': position['allocation'],
            'cost_basis_value': position['cost_basis']['value'],
            'cost_basis_currency': position['cost_basis']['currency'],
            'is_cash': position['is_cash'],
            'average_entry_price_value': position['average_entry_price']['value'],
            'average_entry_price_currency': position['average_entry_price']['currency'],
            'available_to_trade_crypto': position['available_to_trade_crypto'],
            'unrealized_pnl': position['unrealized_pnl'],
            'available_to_transfer_fiat': position['available_to_transfer_fiat'],
            'available_to_transfer_crpyto': position['available_to_transfer_crypto']
        }
        portfolio_data.append(position_info)
        df_portfolio = pd.DataFrame(portfolio_data)

    return df_portfolio


def determine_coinbase_start_date(ticker_list):
    start_date_dict = {}
    ticker_start_date = '2016-01-01'
    dates = pd.date_range(start=ticker_start_date, periods=12 * 10, freq='MS')
    end_date = datetime.datetime.now().date()
    for ticker in ticker_list:
        for date in dates:
            try:
                # Pass the current date as the end_date to the function
                print(f"Checking data for {ticker}: {date}")
                df = save_historical_crypto_prices_from_coinbase(ticker=ticker, user_start_date=True,
                                                                 start_date=date, end_date=end_date)
                available_date = date  # Store the first available date
                print(f"Data available from: {available_date}")
                break  # Exit the loop when data is found
            except KeyError:
                pass  # Continue checking the next date if data is not available
            except HTTPError:
                print(f"HTTPError encountered for {ticker}. Skipping further checks.")
                break  # Stop checking more dates for this ticker

        if available_date:
            print(f"First available date for {ticker}: {available_date}")
            start_date_dict[ticker] = date.strftime('%Y-%m-%d')
        else:
            print("No data available within the date range.")

    return start_date_dict


def get_coinbase_daily_historical_price_data(client, ticker, start_timestamp, end_timestamp, retries=3, delay=5):
    granularity = 'ONE_DAY'  # Daily granularity
    attempts = 0

    while attempts < retries:
        try:
            # Attempt to fetch the candles
            candle_list = client.get_candles(
                product_id=ticker,
                start=start_timestamp,
                end=end_timestamp,
                granularity=granularity
            ).candles

            # Process candle data
            candle_data = []
            for candles in candle_list:
                candle_info = {
                    'date': candles['start'],
                    'low': float(candles['low']),
                    'high': float(candles['high']),
                    'open': float(candles['open']),
                    'close': float(candles['close']),
                    'volume': float(candles['volume'])
                }
                candle_data.append(candle_info)

            # Convert to DataFrame
            df_candles = pd.DataFrame(candle_data).sort_values('date')
            df_candles['date'] = pd.to_datetime(df_candles['date'], unit='s').dt.date
            df_candles = df_candles.set_index('date')
            # df_candles['ticker'] = ticker

            return df_candles

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}. Retrying in {delay} seconds...")
            attempts += 1
            time.sleep(delay)

    # If all retries fail, raise the error
    raise Exception("Max retries exceeded. Could not connect to Coinbase API.")


def save_historical_crypto_prices_from_coinbase(ticker, user_start_date=False, start_date=None, end_date=None,
                                                save_to_file=False):

    client = get_coinbase_rest_api_client(key_location)
    if user_start_date:
        start_date = pd.Timestamp(start_date)
    else:
        start_date = coinbase_start_date_by_ticker_dict.get(ticker)
        start_date = pd.Timestamp(start_date)
        if not start_date:
            print(f"Start date for {ticker} is not included in the dictionary!")
            return None

    temp_start_date = start_date
    end_date = pd.Timestamp(end_date)
    current_end_date = temp_start_date
    crypto_price_list = []
    while current_end_date < end_date:
        current_end_date = pd.to_datetime(temp_start_date) + datetime.timedelta(weeks=6)
        if current_end_date > end_date:
            current_end_date = end_date
        start_timestamp = int(temp_start_date.timestamp())
        end_timestamp = int(current_end_date.timestamp())
        # print(temp_start_date, current_end_date, end_date)
        crypto_price_list.append(
            get_coinbase_daily_historical_price_data(client, ticker, start_timestamp, end_timestamp))
        temp_start_date = pd.to_datetime(current_end_date) + datetime.timedelta(days=1)

    df = pd.concat(crypto_price_list, axis=0)

    if save_to_file:
        filename = f"{ticker}-pickle-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
        output_file = f'coinbase_historical_price_folder/{filename}'
        df.to_pickle(output_file)

    return df


def get_coinbase_ohlc_data(ticker):
    pickle_file_path = f"{os.environ.get('HOME')}/Documents/git/trend_following/coinbase_historical_price_folder/"
    files = os.listdir(pickle_file_path)
    matching_files = [f for f in files if f.startswith(ticker)]

    if not matching_files:
        raise FileNotFoundError(f"No file for {ticker} is found!!!")

    file_to_load = os.path.join(pickle_file_path, matching_files[0])
    df = pd.read_pickle(file_to_load)

    return df

