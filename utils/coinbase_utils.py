from coinbase.rest import RESTClient
import requests.exceptions
from json import dumps
import json
import ast
import pandas as pd
import numpy as np
import datetime as dt
import os
import math
from decimal import Decimal
import math, time, uuid
from typing import Dict, Any
from requests.exceptions import HTTPError

## Coinbase API Documentation: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/introduction

coinbase_start_date_by_ticker_dict = {
    ## L1 Coins
    'BTC-USD': '2016-01-01',
    'ETH-USD': '2016-06-01',
    'SOL-USD': '2021-06-01',
    'LTC-USD': '2016-09-01',
    'DOGE-USD': '2021-06-01',
    'CRO-USD': '2021-11-01',
    'ADA-USD': '2021-03-01',
    'AVAX-USD': '2021-09-01',
    'XRP-USD': '2023-06-01',
    'DOT-USD': '2021-06-01',
    'ALGO-USD': '2019-08-01',
    'XLM-USD': '2019-02-01',
    'ATOM-USD': '2020-01-01',
    'NEAR-USD': '2022-09-01',
    'APT-USD': '2022-10-19',
    'SUI-USD': '2023-05-18',
    'TON-USD': '2025-11-18',
    'ICP-USD': '2021-05-10',
    'XTZ-USD': '2019-08-06',
    'HBAR-USD': '2022-10-13',
    'EGLD-USD': '2022-12-07',
    'FIL-USD': '2020-12-09',
    'RNDR-USD': '2022-02-03',

    ## L2 Coins
    'MATIC-USD': '2021-02-01',
    'SKL-USD': '2021-02-01',
    'OP-USD': '2022-06-01',
    'ARB-USD': '2023-03-23',
    'POL-USD': '2024-09-04',
    'IMX-USD': '2021-12-09',
    'STRK-USD': '2024-02-21',
    'BLAST-USD': '2024-06-26',
    'ZK-USD': '2024-09-25',
    'LRC-USD': '2020-09-15',
    'ZORA-USD': '2025-04-24',
    'METIS-USD': '2022-06-28',
    'STX-USD': '2022-01-20',

    ## Stable Coins
    'USDT-USD': '2021-05-04',
    'DAI-USD': '2020-04-01',
    'USD1-USD': '2025-08-21',
    'PAX-USD': '2021-07-27',

    ## Defi Coins
    'SHIB-USD': '2021-08-01',
    'LINK-USD': '2019-06-01',
    'UNI-USD': '2020-09-01',
    'FET-USD': '2021-07-01',
    'AAVE-USD': '2020-12-01',
    'MANA-USD': '2021-04-01',
    'OXT-USD': '2019-12-01',
    'KRL-USD': '2021-10-01',
    'AMP-USD': '2021-05-01',
    'REQ-USD': '2021-07-01',
    'GRT-USD': '2020-12-01',
    'MOBILE-USD': '2024-02-01',
    'AIOZ-USD': '2022-02-01',
    'ZRO-USD': '2024-06-01',
    'HNT-USD': '2023-06-01',
    'HONEY-USD': '2024-01-01',
    'COMP-USD': '2020-06-23',
    'LDO-USD': '2022-11-17',
    'MKR-USD': '2020-06-09',
    'SNX-USD': '2020-12-15',
    'INJ-USD': '2022-09-20',
    'SUSHI-USD': '2021-03-11',
    'CRV-USD': '2021-03-25',
    'BAL-USD': '2020-10-06',
    '1INCH-USD': '2021-04-09',
    'RPL-USD': '2022-12-08',
    'RSR-USD': '2025-04-22',
    'DIA-USD': '2022-01-25',
    'ONDO-USD': '2024-01-22',
    'ETHFI-USD': '2025-02-06'
}


def get_portfolio_key(portfolio_name):
    key_location = ''
    if portfolio_name == 'Default':
        key_location = f'{os.environ.get('HOME')}/Documents/git/trend_following/exchange_config/cdp_api_key_default.json'
    elif portfolio_name == 'Trend Following':
        key_location = f'{os.environ.get('HOME')}/Documents/git/trend_following/exchange_config/cdp_api_key_trend_following.json'

    return key_location


def get_coinbase_rest_api_client(portfolio_name):
    key_location = get_portfolio_key(portfolio_name)
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

    # Make end_date tz-naive (date) so it plays nicely with your helper
    end_date = dt.datetime.now().date()

    for ticker in ticker_list:
        available_date = None  # reset per ticker

        for date in dates:
            try:
                print(f"Checking data for {ticker}: {date}")
                df = save_historical_crypto_prices_from_coinbase(
                    ticker=ticker,
                    user_start_date=True,
                    start_date=date,  # tz-naive Timestamp from pd.date_range
                    end_date=end_date,  # tz-naive date
                )

                # If no data returned for this query, move to the next month
                if df is None or df.empty:
                    continue

                # Determine the *actual* first date from the returned data
                if isinstance(df.index, pd.DatetimeIndex):
                    first_ts = df.index.min()
                else:
                    first_ts = pd.to_datetime(df['time']).min()

                available_date = first_ts.date()
                print(f"Data available from: {available_date}")
                break  # Found first non-empty response; stop scanning

            except KeyError:
                # Your helper might still raise KeyError if nothing found
                continue
            except HTTPError:
                print(f"HTTPError encountered for {ticker}. Skipping further checks.")
                break  # Stop checking more dates for this ticker

        if available_date is not None:
            print(f"First available date for {ticker}: {available_date}")
            start_date_dict[ticker] = available_date.strftime('%Y-%m-%d')
        else:
            print(f"No data available within the date range for {ticker}.")
            start_date_dict[ticker] = None

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

            # If empty, return an EMPTY frame with expected schema & a proper index name
            if not candle_list:
                cols = ["open", "high", "low", "close", "volume"]
                return (pd.DataFrame(columns=cols)
                        .rename_axis("date"))

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
            df_candles = pd.DataFrame(candle_data)#.sort_values('date')

            if df_candles.empty or "date" not in df_candles.columns:
                cols = ["open", "high", "low", "close", "volume"]
                return (pd.DataFrame(columns=cols)
                        .rename_axis("date"))

            # df_candles['date'] = (
            #     pd.to_datetime(pd.to_numeric(df_candles['date'], errors='coerce'), unit='s', utc=True)
            #     .dt.tz_convert(None)  # optional: drop tz
            #     .dt.date  # if you truly want Python date objects
            # )
            # # df_candles = df_candles.set_index('date')
            # df_candles = (df_candles.set_index("date")
            #               .sort_index()
            #               .rename_axis("date"))
            # # df_candles['ticker'] = ticker

            # With this (Timestamp index, normalized to midnight):
            # epoch seconds -> tz-aware UTC
            s = pd.to_datetime(pd.to_numeric(df_candles['date'], errors='coerce'), unit='s', utc=True)

            # drop timezone and normalize to midnight
            s = s.dt.tz_localize(None).dt.normalize()

            df_candles['date'] = s
            df_candles = (df_candles
                          .set_index('date')
                          .sort_index()
                          .rename_axis('date'))

            return df_candles

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}. Retrying in {delay} seconds...")
            attempts += 1
            time.sleep(delay)

    # If all retries fail, raise the error
    raise Exception("Max retries exceeded. Could not connect to Coinbase API.")


def save_historical_crypto_prices_from_coinbase(ticker, user_start_date=False, start_date=None, end_date=None,
                                                save_to_file=False, portfolio_name='Default'):

    client = get_coinbase_rest_api_client(portfolio_name=portfolio_name)
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
        current_end_date = pd.to_datetime(temp_start_date) + dt.timedelta(weeks=6)
        if current_end_date > end_date:
            current_end_date = end_date
        start_timestamp = int(temp_start_date.timestamp())
        end_timestamp = int(current_end_date.timestamp())
        # print(temp_start_date, current_end_date, end_date)
        crypto_price_list.append(
            get_coinbase_daily_historical_price_data(client, ticker, start_timestamp, end_timestamp))
        temp_start_date = pd.to_datetime(current_end_date) + dt.timedelta(days=1)

    df = pd.concat(crypto_price_list, axis=0)

    if save_to_file:
        filename = f"{ticker}-pickle-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
        output_file = f'coinbase_historical_price_folder/{filename}'
        df.to_pickle(output_file)

    return df


## Get OHLC data for each coin
def get_coinbase_candle_data(client, product_id, start_date, end_date):
    """Return a daily OHLCV DataFrame indexed by date. Empty DF if no data."""

    start_date = pd.Timestamp(start_date)
    end_date   = pd.Timestamp(end_date)
    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).timestamp())

    resp = client.get_candles(
        product_id=product_id,
        start=start_timestamp,
        end=end_timestamp,
        granularity='ONE_DAY',
    )
    candles = resp.candles or []

    if not candles:
        # return an empty frame with the expected schema
        cols = ['low','high','open','close','volume']
        return pd.DataFrame(columns=cols).astype({c:'float64' for c in cols})

    rows = [{
        'date':   c['start'],
        'low':    float(c['low']),
        'high':   float(c['high']),
        'open':   float(c['open']),
        'close':  float(c['close']),
        'volume': float(c['volume']),
    } for c in candles]

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(pd.to_numeric(df['date'], errors='coerce'), unit='s', utc=True).dt.date
    return df.sort_values('date').set_index('date')


def get_coinbase_ohlc_data(ticker):
    pickle_file_path = f"{os.environ.get('HOME')}/Documents/git/trend_following/data_folder/coinbase_historical_price_folder/"
    files = os.listdir(pickle_file_path)
    matching_files = [f for f in files if f.startswith(ticker)]

    if not matching_files:
        raise FileNotFoundError(f"No file for {ticker} is found!!!")

    file_to_load = os.path.join(pickle_file_path, matching_files[0])
    df = pd.read_pickle(file_to_load)

    return df


def round_to_increment(x, step):
    if step is None or step == 0:
        return float(x)
    return round(round(x / step) * step, int(max(0, -math.log10(step))))


def round_down(x: float, step: float) -> float:
    if step is None or step == 0:
        return float(x)
    return round_to_increment(math.floor(x / step) * step, step)


def round_up(x: float, step: float) -> float:
    if step is None or step == 0:
        return float(x)
    return round_to_increment(math.ceil(x / step) * step, step)


def get_product_meta(client, product_id: str):
    """
    Returns increments & mins for sizing/price rounding.
    """
    p = client.get_product(product_id)  # Advanced Trade: /api/v3/brokerage/products/{product_id}
    # Public/List Public Products also exposes similar fields.
    # Fields include base_increment, quote_increment, base_min_size, quote_min_size, price_increment, etc.
    return {
        "base_increment": float(p.base_increment),                                                    ## Minimum amount base value can be increased or decreased at once.
        "quote_increment": float(p.quote_increment) if getattr(p, "quote_increment", None) else None, ## Minimum amount quote value can be increased or decreased at once
        "base_min_size": float(p.base_min_size),                                                      ## Minimum size that can be represented of base currency
        "quote_min_size": float(p.quote_min_size) if getattr(p, "quote_min_size", None) else None,    ## Minimum size that can be represented of quote currency
        "price_increment": float(p.price_increment) if getattr(p, "price_increment", None) else None, ## Minimum amount price can be increased or decreased at once
    }


def get_price_map(client, ticker_list):
    """
     Pull Live Prices for each ticker in the portfolio
    """
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


## Load Total Portfolio Equity and Available Cash
def get_live_portfolio_equity_and_cash(client, portfolio_name='Default'):

    ## Get Portfolio UUID and Positions
    portfolio_uuid = get_portfolio_uuid(client, portfolio_name)
    df_portfolio_positions = get_portfolio_breakdown(client, portfolio_uuid)

    ## Get Portfolio Value and Available Cash
    cash_cond = (df_portfolio_positions.is_cash == True)
    portfolio_equity = float(df_portfolio_positions[~cash_cond]['total_balance_fiat'].sum())
    available_cash = float(df_portfolio_positions[cash_cond]['available_to_trade_fiat'].sum())

    return portfolio_equity, available_cash


## Live Positions in the portfolio
def get_current_positions_from_portfolio(client, ticker_list, portfolio_name='Default'):

    df_portfolio = get_portfolio_breakdown(client, portfolio_uuid=get_portfolio_uuid(client, portfolio_name))
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


def _to_utc_iso(x) -> str:
    ts = pd.to_datetime(x, utc=True)
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        ts = ts.normalize()
    return ts.isoformat().replace("+00:00", "Z")


def _parse_fill_time(ts_val):
    """
    Parses a timestamp field from the fills payload into a tz-aware UTC datetime.
    """
    return pd.to_datetime(ts_val, utc=True).to_pydatetime()


def list_open_stop_orders(client, product_id):
    """
    Return a normalized list of OPEN stop/stop-limit orders for `product_id`.
    Adjust the underlying client call and field names to match your SDK.
    Output items have: order_id, client_order_id, product_id, side, type, stop_price, created_at
    """
    # 1) Fetch open orders (adapt this block to your SDK)
    try:
        # Path A: some SDKs use list_orders(product_id=..., order_status=[...])
        res = client.list_orders(product_id=product_id, order_status=['OPEN'])
        raw = res['orders']
    except Exception:
        raw = []

    # 2) Normalize & filter only stops
    orders = []
    for o in raw:
        otype = o['order_type'].lower()
        # NEW: get stop_price from order_configuration
        cfg = o['order_configuration']
        stop_cfg = cfg['stop_limit_stop_limit_gtc']
        stop_price = stop_cfg['stop_price']

        if ('stop' in otype) or (stop_price is not None):
            orders.append({
                'order_id': o['order_id'] or o['id'],
                'client_order_id': o['client_order_id'] or o['clientOrderId'],
                'product_id': o['product_id'] or o['productId'] or product_id,
                'side': o['side'],
                'type': o['order_type'] or o['type'],
                'stop_price': float(stop_price) if stop_price is not None else np.nan,
                'created_at': o['created_time'] or o['created_at'],
            })
    return orders


def get_open_stop_price(client, product_id, client_id_prefix="stop-"):
    """
    Return the highest stop_price among your currently-open stops for `product_id`,
    optionally filtered by your client_order_id prefix. NaN if none.
    """
    try:
        open_stops = list_open_stop_orders(client, product_id) or []
    except Exception:
        open_stops = []

    prices = []
    for order in open_stops:
        clordid = str(order['client_order_id'] or '')
        if (client_id_prefix is None) or clordid.startswith(f"{client_id_prefix}{product_id}-"):
            stop_price = order['stop_price']
            if stop_price is not None and np.isfinite(stop_price):
                prices.append(float(stop_price))
    return max(prices) if prices else np.nan


def get_stop_fills(
        client,
        product_id: str,
        start,  # date or datetime
        end,  # date or datetime
        client_id_prefix: str = "stop-",
        page_size: int = 250,
):
    """
    Returns [(fill_timestamp_utc, price_float), ...] for STOP orders only,
    between start and end inclusive. Output is sorted ascending by time.
    """
    cursor = None
    out = []

    start_iso = _to_utc_iso(start)
    end_iso = _to_utc_iso(end)

    while True:
        resp = client.get_fills(
            product_id=product_id,
            start_date=start_iso,
            end_date=end_iso,
            limit=page_size,
            cursor=cursor
        )

        # SDKs vary: support attr or dict access
        # fills = getattr(resp, "fills", None) or getattr(resp, "data", None) or resp.get("fills", [])
        # cursor = getattr(resp, "cursor", None) or resp.get("cursor")
        fills = resp['fills']
        cursor = resp['cursor']

        for f in fills or []:
            # 1) try to get client_order_id directly (not present in fills)
            cid = f.get("client_order_id") or f.get("clientOrderId")

            # 2) if missing, fetch the order by order_id to read its client_order_id
            if not cid:
                try:
                    ord_resp = client.get_order(order_id=f.get("order_id"))['order']
                    # handle dict or object
                    if isinstance(ord_resp, dict):
                        cid = ord_resp.get("client_order_id") or ord_resp.get("clientOrderId")
                    else:
                        cid = getattr(ord_resp, "client_order_id", None)
                except Exception:
                    cid = None

            # 3) keep only fills whose client_order_id matches your prefix (e.g., "stop-")
            if client_id_prefix is not None:
                if not (cid and str(cid).startswith(client_id_prefix)):
                    continue

            ts_raw = f.get("trade_time") or f.get("time") or f.get("created_at")
            price = float(f.get("price"))
            ts = _parse_fill_time(ts_raw)
            out.append((ts, price))

        if not cursor:
            break

    out.sort(key=lambda x: x[0])
    return out


def get_fills_with_client_ids(
    client,
    product_id: str,
    start,
    end,
    client_id_prefix = None,   # e.g. "stop-" to keep only stop orders; None = keep all
    page_size: int = 250,
):
    """
    Returns a list of dicts, one per fill:
    {
      'trade_id', 'order_id', 'client_order_id', 'product_id', 'side',
      'price', 'size', 'commission', 'trade_time'
    }
    """
    cursor = None
    out = []
    cache = {}  # order_id -> client_order_id

    start_iso = _to_utc_iso(start)
    end_iso   = _to_utc_iso(end)

    while True:
        resp = client.get_fills(
            product_id=product_id,
            start_date=start_iso,
            end_date=end_iso,
            limit=page_size,
            cursor=cursor
        )
        fills  = resp.get('fills', []) if isinstance(resp, dict) else resp.fills
        cursor = (resp.get('cursor') if isinstance(resp, dict) else getattr(resp, 'cursor', None)) or None

        for f in fills or []:
            order_id = f.get("order_id")
            cid = cache.get(order_id)

            if cid is None:
                try:
                    ord_resp = client.get_order(order_id=order_id)
                    ord_obj  = ord_resp['order']#ord_resp.get("order") if isinstance(ord_resp, dict) else ord_resp
                    cid = (ord_obj.get("client_order_id") if isinstance(ord_obj, dict)
                           else getattr(ord_obj, "client_order_id", None))
                except Exception:
                    cid = None
                cache[order_id] = cid

            # optional filter by client_order_id prefix (e.g., keep only "stop-" orders)
            if client_id_prefix and not (cid and str(cid).startswith(client_id_prefix)):
                continue

            out.append({
                "trade_id":        f.get("trade_id"),
                "order_id":        order_id,
                "client_order_id": cid,
                "product_id":      f.get("product_id"),
                "side":            f.get("side"),
                "price":           float(f.get("price")) if f.get("price") is not None else None,
                "size":            float(f.get("size"))  if f.get("size")  is not None else None,
                "commission":      float(f.get("commission")) if f.get("commission") is not None else None,
                "trade_time":      _parse_fill_time(f.get("trade_time") or f.get("time") or f.get("created_at")),
            })

        if not cursor:  # '' or None means no next page
            break

    # sort by time
    out.sort(key=lambda r: r["trade_time"])
    return out


def _order_cfg(o):
    if o["type"].lower() == "market":
        return {"market_market_ioc": {"base_size": str(o["size"])}}
    # limit
    lp = o.get("limit_price")
    if lp is None:
        raise ValueError("limit_price required for limit orders")
    tif = (o.get("time_in_force") or "GTC").upper()
    key = {"GTC": "limit_limit_gtc", "GTD": "limit_limit_gtd", "FOK": "limit_limit_fok", "IOC": "limit_limit_ioc"}.get(
        tif, "limit_limit_gtc")
    return {key: {"base_size": str(o["size"]), "limit_price": str(lp), "post_only": bool(o.get("post_only", False))}}


def _preview_payload(o):
    return {"product_id": o["product_id"], "side": str(o["side"]).upper(), "order_configuration": _order_cfg(o)}


def _create_payload(o):
    p = _preview_payload(o)
    p["client_order_id"] = o["client_order_id"]

    return p


def _as_dict(resp):
    # handle dict, pydantic-like, or stringified dict
    if isinstance(resp, dict):
        return resp
    for attr in ("dict", "to_dict", "model_dump"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                d = fn();
                if isinstance(d, dict):
                    return d
            except:
                pass
    s = resp if isinstance(resp, str) else str(resp)
    try:
        return json.loads(s)
    except:
        try:
            d = ast.literal_eval(s);
            return d if isinstance(d, dict) else {"raw": s}
        except:
            return {"raw": s}


def preview_order(client, order):
    payload = _preview_payload(order)
    r = client.preview_order(**payload)
    d = _as_dict(r)
    errs = d.get("errs") or []
    ok = bool(d.get("preview_id")) and (len(errs) == 0)
    return {"ok": ok, "data": d if ok else None, "error": None if ok else d, "request": payload}


def create_order(client, order, preview_first=False):
    if preview_first:
        pv = preview_order(client, order)
        if not pv["ok"]:
            return {"ok": False, "error": {"preview_failed": pv}, "request": order}
    payload = _create_payload(order)
    r = client.create_order(**payload)
    d = _as_dict(r)
    ok = bool(d.get("order_id") or d.get("success") is True)
    return {"ok": ok, "data": d if ok else None, "error": None if ok else d, "request": payload}


def preview_orders(client, orders):
    return [preview_order(client, od) for od in orders]


def create_orders(client, orders, preview_first=False):
    return [create_order(client, od, preview_first=preview_first) for od in orders]


def place_stop_limit_order(
    client,
    product_id,
    side,
    stop_price,
    size,
    client_order_id,
    *,
    buffer_bps=50,
    preview=False,
    price_increment=0.01,
    base_increment=1e-8,
    quote_min_size=None
):
    """
    Stop-LIMIT (emulates stop-market with buffer) using your round_to_increment.
      SELL (long protection): limit = stop * (1 - buffer), stop ↑, limit ↓
      BUY  (short protection): limit = stop * (1 + buffer), stop ↓, limit ↑
    """
    side = side.upper()
    buf = float(buffer_bps) / 10_000.0

    # Directional rounding for prices
    if side == "SELL":
        sp = round_up(float(stop_price), float(price_increment))
        lp = round_down(sp * (1.0 - buf), float(price_increment))
        stop_dir = "STOP_DIRECTION_STOP_DOWN"
    elif side == "BUY":
        sp = round_down(float(stop_price), float(price_increment))
        lp = round_up(sp * (1.0 + buf), float(price_increment))
        stop_dir = "STOP_DIRECTION_STOP_UP"
    else:
        raise ValueError("side must be 'BUY' or 'SELL'")

    # Base size rounding (↓) and optional min notional enforcement
    sz = round_down(float(size), float(base_increment))
    if quote_min_size:
        # ensure price*size >= quote_min_size
        if sp * sz < float(quote_min_size):
            needed = float(quote_min_size) / sp
            # bump up, then round down to base_increment (to avoid exceeding increments)
            sz = round_down(needed, float(base_increment))

    order_configuration = {
        "stop_limit_stop_limit_gtc": {
            "base_size":     f"{sz}",
            "limit_price":   f"{lp}",
            "stop_price":    f"{sp}",
            "stop_direction": stop_dir,
        }
    }

    if preview:
        # NOTE: preview must NOT include client_order_id
        return client.preview_order(
            product_id=product_id,
            side=side,
            order_configuration=order_configuration,
        )

    # Live placement: include client_order_id
    return client.create_order(
        client_order_id=client_order_id,
        product_id=product_id,
        side=side,
        order_configuration=order_configuration,
    )


def _flatten_cfg(order):
    """
    Pull common fields from order.order_configuration without guessing types.
    Looks at the first (and usually only) sub-config present.
    """
    cfg = getattr(order, "order_configuration", None) or {}
    if not isinstance(cfg, dict) or not cfg:
        return {}

    # take the first sub-config dict (e.g., 'limit_limit_gtc', 'stop_limit_stop_limit_gtc', etc.)
    key, sub = next(iter(cfg.items()))
    if not isinstance(sub, dict):
        return {}

    # extract common fields if present
    out = {}
    if "base_size"   in sub: out["base_size"]   = sub["base_size"]
    if "limit_price" in sub: out["limit_price"] = sub["limit_price"]
    if "stop_price"  in sub: out["stop_price"]  = sub["stop_price"]
    if "stop_direction" in sub: out["stop_direction"] = sub["stop_direction"]
    if "post_only"   in sub: out["post_only"]   = sub["post_only"]
    out["config_key"] = key  # helpful to know which config it was
    return out


def norm_order(o):
    """Normalize a coinbase.rest.types.common_types.Order into a plain dict."""
    data = {
        "order_id":        o.order_id,
        "client_order_id": getattr(o, "client_order_id", None),
        "product_id":      getattr(o, "product_id", None),
        "side":            getattr(o, "side", None),
        "type":            getattr(o, "order_type", None),
        "status":          getattr(o, "status", None),
        "time_in_force":   getattr(o, "time_in_force", None),
        "created_time":    getattr(o, "created_time", None),
        "filled_size":     getattr(o, "filled_size", None),
        "average_price":   getattr(o, "average_filled_price", None),
    }
    data.update(_flatten_cfg(o))
    return data


def list_open_orders(client, product_id: str | None = None):
    out, cursor = [], None
    while True:
        kw = {"order_status": ["OPEN"]}
        if product_id: kw["product_id"] = product_id
        if cursor:     kw["cursor"] = cursor

        res = client.list_orders(**kw)           # {'orders': [Order,...], 'cursor': '', 'has_next': False}
        orders = res['orders']
        out.extend({
            "order_id":        o.order_id,
            "client_order_id": getattr(o, "client_order_id", None),
            "product_id":      getattr(o, "product_id", None),
            "side":            getattr(o, "side", None),
            "type":            getattr(o, "order_type", None),
            "status":          getattr(o, "status", None),
            "time_in_force":   getattr(o, "time_in_force", None),
            "created_time":    getattr(o, "created_time", None),
            "filled_size":     getattr(o, "filled_size", None),
            "average_price":   getattr(o, "average_filled_price", None),
        } for o in orders)

        cursor   = res['cursor']#res.get("cursor") or None     # '' -> None
        has_next = bool(res["has_next"])#bool(res.get("has_next", False))
        if not cursor or not has_next:
            break
    return out


def find_open_by_client_id(client, client_order_id: str, product_id: str | None = None):
    """Find one OPEN order by your client_order_id."""
    for o in list_open_orders(client, product_id=product_id):
        if o.get("client_order_id") == client_order_id:
            return o
    return None


def cancel_order_by_id(client, order_id: str):
    # MUST be a list
    return client.cancel_orders(order_ids=[order_id])


def cancel_order_by_client_id(client, client_order_id: str, product_id: str | None = None):
    # find the open order, then cancel by order_id
    for o in list_open_orders(client, product_id=product_id):
        if o["client_order_id"] == client_order_id:
            return cancel_order_by_id(client, o["order_id"])
    return {"ok": False, "error": "not_found", "client_order_id": client_order_id}


def cancel_all_open_orders(client, product_id: str | None = None):
    """Cancel every OPEN order (optionally for a single product)."""
    orders = list_open_orders(client, product_id=product_id)
    results = []
    for o in orders:
        oid = o.get("order_id")
        if oid:
            results.append(cancel_order_by_id(client, oid))
    return results


