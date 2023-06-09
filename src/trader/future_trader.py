from datetime import date


from utils.utils import get_auth
from utils.dataloader import get_symbols_by_names


class FutureTrader:
    def __init__(self, account="a1"):
        self.auth, _ = get_auth(account)
        self.commodity = "iron_orb"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.is_wandb = True
        self.volume = 5
        self.commission_fee = 4.5

    def backtest(self, strategy: str = "dynamic_linear_regression_channels"):
        if strategy == "simple_ema":
            from .strategies.simple_ema import SimpleHFEMA
            symbol = "CZCE.CF305"
            model = SimpleHFEMA(
                auth=self.auth,
                commission_fee=self.commission_fee,
                volume=self.volume,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
                start_dt=date(2022, 11, 10),
                end_dt=date(2022, 12, 14)
            )
        elif strategy == "simple_hf":
            from .strategies.simple_hf import backtest
            symbol = "DCE.i2301"
            tick_price = 1
            close_countdown_seconds = 5
            backtest(
                auth=self.auth,
                symbol=symbol,
                is_wandb=self.is_wandb,
                commission_fee=self.commission_fee,
                volume=self.volume,
                tick_price=tick_price,
                close_countdown_seconds=close_countdown_seconds,
                start_dt=date(2022, 11, 20),
                end_dt=date(2022, 11, 30)
            )
        elif strategy == "simple_arbitrage":
            from .strategies.simple_arbitrage import SimpleArbitrage
            model = SimpleArbitrage(
                auth=self.auth,
            )
            model.backtest()
        elif strategy == "simple_hf_arron":
            from .strategies.simple_hf_aroon import SimpleHFAroon
            symbol = "CZCE.CF305"
            model = SimpleHFAroon(
                auth=self.auth,
                commission_fee=self.commission_fee,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
                start_dt=date(2022, 11, 10),
                end_dt=date(2022, 12, 14)
            )
        elif strategy == "half_hull":
            from .strategies.half_hull import HalfHull
            symbol = "CZCE.CF305"
            model = HalfHull(
                auth=self.auth,
                commission_fee=self.commission_fee,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
                start_dt=date(2022, 11, 10),
                end_dt=date(2022, 12, 14)
            )
        elif strategy == "nadaraya_watson_combine":
            from .strategies.nadaraya_watson_combine import NadarayaWatsonCombine
            symbol = "CZCE.CF305"
            model = NadarayaWatsonCombine(
                auth=self.auth,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
            )
        elif strategy == "dynamic_linear_regression_channels":
            from .strategies.dynamic_linear_regression_channels import DynamicLinearRegressionChannels
            symbol = "CZCE.CF305"
            model = DynamicLinearRegressionChannels(
                auth=self.auth,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
            )
        elif strategy == "lclassification":
            from .strategies.lclassification import LClassification
            symbol = "CZCE.CF305"
            model = LClassification(
                auth=self.auth,
                is_wandb=self.is_wandb
            )
            model.backtest(
                symbol=symbol,
            )

