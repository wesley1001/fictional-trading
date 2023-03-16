
from datetime import date, datetime

from contextlib import closing
import math
import typing
import numpy as np
from tqsdk import tafunc, TqAuth, TqApi, TqSim, TqBacktest, TqAccount, TargetPosTask, BacktestFinished
from tqsdk.objs import Account, Quote

import wandb

from ml_extensions_2 import ml_extensions as ml
from .common import get_lorentzian_distance

class LClassification:
    def __init__(self, auth: TqAuth, commission_fee: float = 4.4, volume: int = 1, is_wandb: bool = True):
        self.auth = auth
        self.commission_fee = commission_fee
        self.is_wandb = is_wandb
        self.volume = volume

        self.set_ml_model()

    def backtest(
        self,
        symbol: str,
        start_dt=date(2022, 12, 1),
        end_dt=date(2022, 12, 25),
        is_live: bool = False,
    ):
        if is_live:
            acc: Account = TqAccount()  # TBD
            self.api = TqApi(account=acc, auth=self.auth, web_gui=False)
            self.account: Account = self.api.get_account()
        else:
            # use simulation account
            sim = TqSim(init_balance=200000)
            self.api = TqApi(account=sim, auth=self.auth, backtest=TqBacktest(
                start_dt=start_dt, end_dt=end_dt), web_gui=False)
            sim.set_commission(symbol=symbol, commission=self.commission_fee)
            self.account: Account = self.api.get_account()

        print("Subscribing quote")
        quote: Quote = self.api.get_quote(symbol)

        klines_1m = self.api.get_kline_serial(
            symbol, duration_seconds=45, data_length=500)
        self.target_pos_task = TargetPosTask(self.api, symbol, price="ACTIVE")
        if self.is_wandb:
            wandb.init(project="backtest-1", config={"symbol": symbol})
        with closing(self.api):
            try:
                while True:
                    self.api.wait_update()
                    if self.api.is_changing(klines_1m.iloc[-1], "datetime"):

                        # calculate signal time
                        start_time = datetime.now()
                        signal = self.get_signal(klines_1m)
                        end_time = datetime.now()

                        if signal == 1:
                            self.target_pos_task.set_target_volume(self.volume)
                        elif signal == -1:
                            self.target_pos_task.set_target_volume(
                                -self.volume)
                        else:
                            # hold or close position
                            # print("Hold position")
                            pass
                        self.api.wait_update()

                    if self.is_wandb:
                        wandb.log({
                            "singal_time": (end_time - start_time).total_seconds(),
                            "signal": signal,
                            "last_price": quote.last_price,
                            "static_balance": self.account.static_balance,
                            "account_balance": self.account.balance,
                            "commission": self.account.commission,
                        })
            except BacktestFinished:
                print("Backtest done")
    
    def get_signal(self, klines_1m: np.ndarray) -> int:
        # calculate signal using ML classification
        signal = 0 # pleaceholder 
        return signal

    #====================#
    #===== ML Model =====#
    #====================#

    def set_ml_model(self):

        #===== ML Settings =====#
        self.source: str = "close"
        # Source of the input data
        self.neighborsCount: int = 8
        # Number of neighbors to consider
        self.maxBarsBack: int = 2000
        # Maximum number of bars to look back
        self.featureCount: int = 5
        # Number of features to use
        self.colorCompression: int = 1
        # Compression factor for adjusting the intensity of the color scale.
        self.showExits: bool = False
        # Default exits occur exactly 4 bars after an entry signal. This corresponds to the predefined length of a trade during the model's training process.
        self.useDynamicExits: bool = False
        # Dynamic exits attempt to let profits ride by dynamically adjusting the exit threshold based on kernel regression logic.
        self.selected_feature: typing.List[str] = ["RSI", "WT", "CCI", "ADX"]    

        #===== Backtest Settings =====#
        '''
        Whether to use the worst case scenario for backtesting. 
        This option can be useful for creating a conservative estimate that is based on close prices only, 
        thus avoiding the effects of intrabar repainting. This option assumes that the user does not enter when 
        the signal first appears and instead waits for the bar to close as confirmation. 
        On larger timeframes, this can mean entering after a large move has already occurred. 
        Leaving this option disabled is generally better for those that use this indicator as a
        source of confluence and prefer estimates that demonstrate discretionary mid-bar entries. 
        Leaving this option enabled may be more consistent with traditional backtesting results.
        '''
        self.showTradeStats: bool = True
        self.useWorstCase: bool = False
        
        #===== Filter =====#
        self.volatility: bool = True
        self.regime: bool = True
        self.adx: bool = True
        #===== Filter Settings =====#
        self.useVolatilityFilter: bool = True
        # Whether to use the volatility filter
        self.useRegimeFilter: bool = True
        # Whether to use the regime filter
        self.useAdxFilter: bool = False
        # Whether to use the ADX filter
        self.regimeThreshold: float = -0.1
        # Whether to use the trend detection filter. Threshold for detecting Trending/Ranging markets
        self.adxThreshold: float = 20
        # Whether to use the ADX filter. Threshold for detecting Trending/Ranging markets.


        #===== ML Model Settings =====#
        self.firstBarIndex: int = 0
        self.trainingLabels: np.ndarray = []
        self.loopSize: int = 1
        self.lastDistance: float = 1.0
        self.distancesArray: np.ndarray = []
        self.predictionsArray: np.ndarray = []
        self.prediction: int = []

        # Feature Variables: User-Defined Inputs for calculating Feature Series.
        self.featureParam: dict = {
            "RSI": {
                "a": 14,
                "b": 1
            },
            "WT": {
                "a": 10,
                "b": 11
            },
            "CCI": {
                "a": 20,
                "b": 1
            },
            "ADX": {
                "a": 20,
                "b": 2
            },
        }
        self.signal = 0 # 1: long, -1: short, 0: neutral

        self.max_bars_back_index = lambda last_bar_index: last_bar_index - self.maxBarsBack if last_bar_index >= self.maxBarsBack else 0

        #===== EMA Settings =====#
        self.use_ma_filter = False
        self.ma_period = 200
        # The period of the EMA used for the EMA Filter.

        #===== Nadaraya-Watson Kernel Regression Settings =====#
        self.use_kernel_filter = True
        self.show_kernel_estimates = False
        self.user_kernel_smoothing = False
        self.kernel_lookback = 3
        # The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars. Recommended range: 3-50
        self.kernel_relative_weight = 0.25
        # Relative weighting of time frames. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel. Recommended range: 0.25-25
        self.kernel_regression_level = 25
        # Bar index on which to start regression. Controls how tightly fit the kernel estimate is to the data. Smaller values are a tighter fit. Larger values are a looser fit. Recommended range: 2-25
        self.kernel_lag = 2
        # Lag for crossover detection. Lower values result in earlier crossovers. Recommended range: 1-2
    
    def ma_filter(self, klines_1m: np.ndarray, ma_type: str = "ema") -> bool:
        if self.use_ma_filter:
            if ma_type == "ema":
                ma = tafunc.ema(klines_1m['close'], self.ma_period)
            elif ma_type == "sma":
                ma = tafunc.sma(klines_1m['close'], self.ma_period)
            ma_uptrend = klines_1m['close'] > ma
            ma_downtrend = klines_1m['close'] < ma

        else:
            ma_uptrend = np.ones(len(klines_1m))
            ma_downtrend = np.ones(len(klines_1m))
        return ma_uptrend, ma_downtrend

    def core_ml_logic(self, ):
        predictions = []
        prediction = 0
        signal = 0
        distances = []
        
        
