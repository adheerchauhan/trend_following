from coinbase.rest import RESTClient
import time
import requests.exceptions
from json import dumps
import pandas as pd
import os


key_location = f'{os.environ.get('HOME')}/Documents/git/trend_following/cdp_api_key.json'


def get_coinbase_rest_api_client(key_location):
    client = RESTClient(key_file=key_location)
    return client


def get_portfolio_uuid(client):
    portfolio_uuid = client.get_portfolios().portfolios[0]['uuid']
    return portfolio_uuid


def get_portfolio_breakdown(client):
    portfolio_uuid = get_portfolio_uuid(client)
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
