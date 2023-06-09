from typing import Dict, Any, List
from datetime import datetime, date
from contextlib import closing
from collections import defaultdict

from pymongo import MongoClient, ReplaceOne
from pymongo.database import Database

import pandas as pd
from tqsdk import TqApi, TqAuth, TqSim, TqBacktest
from tqsdk.objs import Quote, Account, Position
from tqsdk.tafunc import time_to_datetime, time_to_str

from utils.utils import SETTINGS, Interval
from utils.commodity import Commodity

class Mongo:
    def __init__(self) -> None:
        self.db_name: str = SETTINGS['dbs']['db_name']
        self.host: str = SETTINGS['dbs']['host']
        self.port: int = SETTINGS['dbs']['port']
        self.username: str = SETTINGS['dbs']['username']
        self.password: str = SETTINGS['dbs']['password']

        self.client: MongoClient = MongoClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )

        # Initialize database
        self.db: Database = self.client[self.db_name]
        self.INTERVAL = Interval()

        # Initialize collections
        self._init_collection('bar', self.INTERVAL.ONE_SEC)
        self._init_collection('bar', self.INTERVAL.FIVE_SEC)
        self._init_collection('bar', self.INTERVAL.ONE_MIN)
        self._init_collection('bar', self.INTERVAL.FIVE_MIN)
        self._init_collection('bar', self.INTERVAL.FIFTEEN_MIN)
        self._init_collection('bar', self.INTERVAL.THIRTY_MIN)
        self._init_collection('bar', self.INTERVAL.ONE_HOUR)
        # self._init_collection('bar', self.INTERVAL.FOUR_HOUR)
        self._init_collection('bar', self.INTERVAL.ONE_DAY)
        self._init_collection('tick', self.INTERVAL.TICK)

    def _init_collection(self, collection_name: str, interval: str) -> None:
        # Initialize collection
        collection = self.db[collection_name + '_' + interval]
        # Create index
        collection.create_index(
            [
                ("underlying_symbol", 1),
                ("datetime", 1),
            ],
            unique=True,
        )
        # 同一时间可能有两个主连是同一个的合约, 所以不能是unique
        collection.create_index(
            [
                ("instrument_id", 1),
                ("datetime", 1),
            ]
        )
        collection.create_index(
            [
                ("datetime", 1),
            ]
        )
        return collection
    
    def load_tick_data(self, instrument_id: str, start: datetime, end: datetime, limit: int = 0) -> pd.DataFrame:
        collection = self.db['tick_tick']
        cursor = collection.find(
            {
                "instrument_id": instrument_id,
                "datetime": {
                    "$gte": start,
                    "$lte": end,
                },
            },
            {
                "_id": 0,
            },
        ).sort('datetime', 1).limit(limit)
        df = pd.DataFrame(list(cursor))
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def load_bar_data(self, instrument_id: str, start: datetime, end: datetime, interval: str, limit: int = 0) -> pd.DataFrame:
        collection = self.db['bar_' + interval]
        cursor = collection.find(
            {
                "instrument_id": instrument_id,
                "datetime": {
                    "$gte": start,
                    "$lte": end,
                },
            },
            {
                "_id": 0,
            },
        ).sort('datetime', 1).limit(limit)
        df = pd.DataFrame(list(cursor))
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def save_bar_data(self, symbol: str, instrument_id: str, bars: List[pd.Series], interval: str) -> None:
        """
        kline: pd.Series
            * id: 1234 (k线序列号)
            * datetime: 1501080715000000000 (K线起点时间(按北京时间), 自unix epoch(1970-01-01 00:00:00 GMT)以来的纳秒数)
            * open: 51450.0 (K线起始时刻的最新价)
            * high: 51450.0 (K线时间范围内的最高价)
            * low: 51450.0 (K线时间范围内的最低价)
            * close: 51450.0 (K线结束时刻的最新价)
            * volume: 11 (K线时间范围内的成交量)
            * open_oi: 27354 (K线起始时刻的持仓量)
            * close_oi: 27355 (K线结束时刻的持仓量)
        """
        if len(bars) == 0:
            return
        collection = self.db['bar_' + interval]
        requests: List[ReplaceOne] = []
        print("save bar data: ", symbol, instrument_id, interval)
        for bar in bars:
            bar_dict = bar.to_dict()
            bar_dict['datetime'] = time_to_datetime(bar_dict['datetime'])
            bar_dict['underlying_symbol'] = symbol
            bar_dict['instrument_id'] = instrument_id
            requests.append(ReplaceOne(
                {
                    "underlying_symbol": bar_dict['underlying_symbol'],
                    "datetime": bar_dict['datetime'],
                },
                bar_dict,
                upsert=True,
            ))
        result = collection.bulk_write(requests)

    def save_tick_data(self, symbol: str, instrument_id: str, ticks: List[pd.Series]) -> None:
        if len(ticks) == 0:
            return
        collection = self.db['tick_' + self.INTERVAL.TICK]
        requests: List[ReplaceOne] = []
        print("save tick data: ", symbol, instrument_id)
        for tick in ticks:
            tick_dict = tick.to_dict()
            tick_dict['datetime'] = time_to_datetime(tick_dict['datetime'])
            tick_dict['underlying_symbol'] = symbol
            tick_dict['instrument_id'] = instrument_id
            requests.append(ReplaceOne(
                {
                    "underlying_symbol": tick_dict['underlying_symbol'],
                    "datetime": tick_dict['datetime'],
                },
                tick_dict,
                upsert=True,
            ))
        result = collection.bulk_write(requests)

    def update_bar_data_subscribe(self, instrument: str, instrument_quotes: Dict[str, Quote], intervals = None) -> None:
        underlying_symbol = instrument_quotes[instrument].underlying_symbol
        print("updating", underlying_symbol)

        # subscribe data
        if intervals is None or "tick" in intervals:
            self.tick_data[instrument] = self.api.get_tick_serial(
                underlying_symbol, data_length=2)
        if intervals is None or "1s" in intervals:
            self.bar_data_1s[instrument] = self.api.get_kline_serial(
                underlying_symbol, 1, data_length=2)
        if intervals is None or "5s" in intervals:
            self.bar_data_5s[instrument] = self.api.get_kline_serial(
                underlying_symbol, 5, data_length=2)
        if intervals is None or "1m" in intervals:
            self.bar_data_1m[instrument] = self.api.get_kline_serial(
                underlying_symbol, 60, data_length=2)
        if intervals is None or "5m" in intervals:
            self.bar_data_5m[instrument] = self.api.get_kline_serial(
                underlying_symbol, 300, data_length=2)
        if intervals is None or "15m" in intervals:
            self.bar_data_15m[instrument] = self.api.get_kline_serial(
                underlying_symbol, 900, data_length=2)
        if intervals is None or "30m" in intervals:
            self.bar_data_30m[instrument] = self.api.get_kline_serial(
                underlying_symbol, 1800, data_length=2)
        if intervals is None or "1h" in intervals:
            self.bar_data_1h[instrument] = self.api.get_kline_serial(
                underlying_symbol, 3600, data_length=2)
        if intervals is None or "1d" in intervals:
            self.bar_data_1d[instrument] = self.api.get_kline_serial(
                underlying_symbol, 86400, data_length=2)

    def setup(self, auth: TqAuth, start_dt: date = None, end_dt: date = None) -> None:
        if start_dt and end_dt:
            backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
            self.account = TqSim(init_balance=1000000)
            self.api = TqApi(auth=auth, backtest=backtest,
                             account=self.account)
        else:
            self.api = TqApi(auth=auth)

    def download_data(self, auth: TqAuth, instrument_list: list, start_dt: date = None, end_dt: date = None, intervals = None) -> None:
        """
        e.g. 
        dao = MongoDAO()
        dao.download_data(tqAPI.auth, ['cotton', 'methanol', 'rebar', 'soybean', 'sliver', 'copper', 'iron_orb'], date(2016, 1, 1), date(2016, 2, 1))
        """
        print('start download data')
        cmod = Commodity()
        instrument_list = [cmod.get_instrument_name(
            x) for x in instrument_list]

        # Initialize data object
        instrument_quotes: Dict[str, Quote] = dict()
        self.tick_data: Dict[str, pd.DataFrame] = dict()
        self.bar_data_1s: Dict[str, pd.DataFrame] = dict()
        self.bar_data_5s: Dict[str, pd.DataFrame] = dict()
        self.bar_data_1m: Dict[str, pd.DataFrame] = dict()
        self.bar_data_5m: Dict[str, pd.DataFrame] = dict()
        self.bar_data_15m: Dict[str, pd.DataFrame] = dict()
        self.bar_data_30m: Dict[str, pd.DataFrame] = dict()
        self.bar_data_1h: Dict[str, pd.DataFrame] = dict()
        self.bar_data_1d: Dict[str, pd.DataFrame] = dict()

        # Initialize TqApi
        self.setup(auth, start_dt, end_dt)

        self.ticks_data = defaultdict(list)
        self.bars_data_1s = defaultdict(list)
        self.bars_data_5s = defaultdict(list)
        self.bars_data_1m = defaultdict(list)
        self.bars_data_5m = defaultdict(list)
        self.bars_data_15m = defaultdict(list)
        self.bars_data_30m = defaultdict(list)
        self.bars_data_1h = defaultdict(list)
        self.bars_data_1d = defaultdict(list)

        with closing(self.api):
            for instrument in instrument_list:
                instrument_quotes[instrument] = self.api.get_quote(instrument)
                self.update_bar_data_subscribe(instrument, instrument_quotes, intervals)

            print("start loop")
            while True:
                self.api.wait_update()
                for instrument in instrument_list:
                    iqt = instrument_quotes[instrument]
                    underlying_symbol = iqt.underlying_symbol

                    if self.api.is_changing(iqt, "underlying_symbol"):
                        self.update_bar_data_subscribe(
                            instrument, instrument_quotes)
                    if intervals is None or "tick" in intervals:
                        if self.api.is_changing(self.tick_data[instrument].iloc[-1], "datetime"):
                            self.ticks_data[instrument].append(
                                self.tick_data[instrument].iloc[-2])
                    if intervals is None or "1s" in intervals:
                        if self.api.is_changing(self.bar_data_1s[instrument].iloc[-1], "datetime"):
                            self.bars_data_1s[instrument].append(
                                self.bar_data_1s[instrument].iloc[-2])
                    if intervals is None or "5s" in intervals:
                        if self.api.is_changing(self.bar_data_5s[instrument].iloc[-1], "datetime"):
                            self.bars_data_5s[instrument].append(
                                self.bar_data_5s[instrument].iloc[-2])
                    if intervals is None or "1m" in intervals:
                        if self.api.is_changing(self.bar_data_1m[instrument].iloc[-1], "datetime"):
                            self.bars_data_1m[instrument].append(
                                self.bar_data_1m[instrument].iloc[-2])
                    if intervals is None or "5m" in intervals:
                        if self.api.is_changing(self.bar_data_5m[instrument].iloc[-1], "datetime"):
                            self.bars_data_5m[instrument].append(
                                self.bar_data_1m[instrument].iloc[-2])
                    if intervals is None or "15m" in intervals:
                        if self.api.is_changing(self.bar_data_15m[instrument].iloc[-1], "datetime"):
                            self.bars_data_15m[instrument].append(
                                self.bar_data_15m[instrument].iloc[-2])
                    if intervals is None or "30m" in intervals:
                        if self.api.is_changing(self.bar_data_30m[instrument].iloc[-1], "datetime"):
                            self.bars_data_30m[instrument].append(
                                self.bar_data_30m[instrument].iloc[-2])
                    if intervals is None or "1h" in intervals:
                        if self.api.is_changing(self.bar_data_1h[instrument].iloc[-1], "datetime"):
                            self.bars_data_1h[instrument].append(
                                self.bar_data_1h[instrument].iloc[-2])

                    if intervals is None or "1d" in intervals:
                        if self.api.is_changing(self.bar_data_1d[instrument].iloc[-1], "datetime"):
                            self.bars_data_1d[instrument].append(
                                self.bar_data_1d[instrument].iloc[-2])
                        # save data by day to database
                        if intervals is None or "tick" in intervals:
                            self.save_tick_data(underlying_symbol, instrument,
                                                self.ticks_data[instrument])
                            self.ticks_data[instrument] = []
                        if intervals is None or "1s" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_1s[instrument], self.INTERVAL.ONE_SEC)
                            self.bars_data_1s[instrument] = []
                        if intervals is None or "5s" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_5s[instrument], self.INTERVAL.FIVE_SEC)
                            self.bars_data_5s[instrument] = []
                        if intervals is None or "1m" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_1m[instrument], self.INTERVAL.ONE_MIN)
                            self.bars_data_1m[instrument] = []
                        if intervals is None or "5m" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_5m[instrument], self.INTERVAL.FIVE_MIN)
                            self.bars_data_5m[instrument] = []
                        if intervals is None or "15m" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_15m[instrument], self.INTERVAL.FIFTEEN_MIN)
                            self.bars_data_15m[instrument] = []
                        if intervals is None or "30m" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_30m[instrument], self.INTERVAL.THIRTY_MIN)
                            self.bars_data_30m[instrument] = []
                        if intervals is None or "1h" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_1h[instrument], self.INTERVAL.ONE_HOUR)
                            self.bars_data_1h[instrument] = []
                        if intervals is None or "1d" in intervals:
                            self.save_bar_data(underlying_symbol, instrument,
                                            self.bars_data_1d[instrument], self.INTERVAL.ONE_DAY)
                            self.bars_data_1d[instrument] = []
