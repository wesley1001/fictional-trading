from datetime import date

from utils.dataloader import DataLoader, get_symbols_by_names
from utils.constant import INTERVAL

class ModelTrainer:
    def __init__(self, account = "a1", max_sample_size = 1e8):
        print("Initializing Model trainer")
        # auth, _ = get_auth(account)
        self.interval = INTERVAL.ONE_MIN
        self.commodity = "cotton"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.max_sample_size = int(max_sample_size)
    
    def get_training_data(self, start_dt=date(2022, 12, 1), end_dt=date(2023, 3, 1)):
        dataloader = DataLoader(start_dt=start_dt, end_dt=end_dt)
        data = dataloader.get_offline_data(
                    interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size, fixed_dt=True)
        return data

    def run(self, is_train=True, model_name = "ttfp"):
        print("Running Model trainer")
        if model_name == "tcc":
            from .models import TTCModel
            model = TTCModel(interval=self.interval, commodity_name=self.commodity, max_encode_length=200, max_label_length=20)
            if is_train:
                # data = self.get_training_data()
                data = []
                model.set_training_data(data)
                del data
                model.train() 
                # model.tune(search_data_ratio=0.5)
            else:
                predict_data = self.get_training_data(start_dt=date(2022, 1, 1), end_dt=date(2022, 8, 1))
                X_pred, y_pred = model.set_predict_data(predict_data)
                best_model_path = "./tmp/model-best.h5"
                model.predict(best_model_path, X_pred, y_pred)
        elif model_name == "ttfp":
            from .models import TTFPModel
            model = TTFPModel(interval=self.interval, commodity_name=self.commodity)
            if is_train:
                data = self.get_training_data()
                model.set_training_data(data)
                del data
                model.train()
            else:
                pass
        elif model_name == "ttc2":
            from .models import TTCModel2
            data = self.get_training_data()
            model = TTCModel2(
                data = data,
                interval=self.interval, 
                commodity_name=self.commodity, 
                max_encode_length=300, max_label_length=7)
            del data
            if is_train:
                model.train()
            else:
                predict_data = self.get_training_data(start_dt=date(2022, 7, 16), end_dt=date(2022, 8, 1))
                X_pred, y_pred = model.set_predict_data(predict_data)
                # best_model_path = "./tmp/model-best.h5"
                best_model_path = []
                model.predict(best_model_path, X_pred, y_pred)
            
        print("Done")