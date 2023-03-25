import logging 
import os 
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['__MODIN_AUTOIMPORT_PANDAS__'] = '1' # Fix modin warning 

# import modin.pandas as pd
import pandas as pd

import numpy as np

from tqdm import tqdm
# import time

import wandb
from sklearn.model_selection import train_test_split

# from utils.callbacks import get_lr_metric
from utils.preprocess import process_prev_close_spread, set_training_label, process_datatime, add_factors

class TTCModel2:
    # Transformer Time series classification Version 2
    def __init__(self, data: np.ndarray, interval: str, commodity_name: str, max_encode_length: int = 60, max_label_length: int = 5):
        # print('GPU name: ', tf.config.list_physical_devices('GPU'))
        self.project_name = "ts_prediction_5"
        self.commodity_name = commodity_name
        self.interval = interval
        self.max_encode_length = max_encode_length
        self.max_label_length = max_label_length

        self.n_classes = 3
        self.train_col_name = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
        # self.train_col_name = ["close", "volume", "is_daytime"]
        self.factors_name = ["rsi_14", "ma_60", "ma_120", "wt_10_21", "adx_14_6", "cci_14"] 
        self.train_col_name += self.factors_name
        self.fit_config = {
            "batch_size": 512,
            "epochs": 20,
            "validation_split": 0.3,
            "shuffle": True,
        }
        self.datatype_name = "{}{}_{}_{}".format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length)
        self.data_output_path = "./tmp/"+ self.datatype_name+".csv"
        # self.X_output_path = "./tmp/"+ "X_" + self.datatype_name+".npy"
        # self.y_output_path = "./tmp/"+ "y_" + self.datatype_name+".npy"
        self.set_seed(42)
        self.model_path = "./artifacts/"+ self.datatype_name+".h5"
        
        self.set_training_data(data)
    
    def _set_classes(self, y: np.ndarray):
        """
        Set the classes for the model
        """
        c = np.array(np.unique(y, return_counts=True)).T
        print("Class distribution: ", c, c[:, 1] / c[:, 1].sum())
        self.n_classes = len(c)
    
    def _pre_process_data(self, df: pd.DataFrame, is_prev_close_spread: bool = True):
        # start preprocessing by intervals
        print("Start preprocessing data")
        df = pd.DataFrame(df)
        df = process_datatime(df, is_daytime=False)
        # set datatime label and convert them to same format
        if is_prev_close_spread:
            df = process_prev_close_spread(df)
            # fill the spread of different symbols
        df = set_training_label(df, self.max_label_length, self.n_classes, self.interval)
        df = add_factors(self.factors_name)
        print(df.shape)
        df = df[self.train_col_name + ["label"]].dropna()
        print(self.train_col_name + ["label"])
        return df
    
    def _set_train_test(self, data: pd.DataFrame):
        X = np.empty(shape = (data.shape[0] - self.max_encode_length, self.max_encode_length, len(self.train_col_name)), dtype=np.float32)
        y = data.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32)
        data = data[self.train_col_name]
        for i in range(X.shape[0]):
            X[i] = data[i:i+self.max_encode_length]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        self._set_classes(y)
        del X, y, data
        self.input_shape = self.X_train[0].shape
        print("Input shape: ", self.input_shape)
    
    def _set_train_dataset(self, data: pd.DataFrame):
        print("Splitting data")
        train, test = train_test_split(data, test_size=0.25, shuffle=False)
        train, val = train_test_split(train, test_size=0.2, shuffle=False)
        
        self._set_classes(train.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32))
        self.input_shape = (self.max_encode_length, len(self.train_col_name))

    def set_training_data(self, data: pd.DataFrame):
        if len(data) > 0:
            data = self._pre_process_data(data)
            print("Saving data")
            data.to_csv(self.data_output_path, index=False)
        else:
            print("Loading data from", self.data_output_path)
            data = pd.read_csv(self.data_output_path)
            # save 1000 head data for testing
            # data.iloc[20000:20500].to_csv(self.data_output_path + "_test.csv", index=False)

        self._set_train_dataset(data._to_pandas())

    def set_predict_data(self, data: pd.DataFrame):
        print("Set predict data")
        data = self._pre_process_data(data)
        data.to_csv("predict_data.csv", index=False)
        X, y = data[self.train_col_name].to_numpy(), data["label"].to_numpy()
        self._set_classes(y)
        return X, y
    
    def predict(self, model, X_pred, y_pred):
        correct_action_count = 0
        hold_action_count = 0
        for i in tqdm(range(self.max_encode_length, X_pred.shape[0])):
            x = X_pred[i-self.max_encode_length:i]
            predict = model(np.array([x]), training=False)
            close_price = x[-1, 3]

            if predict[0][2] > 0.4:
                # print("Long", predict, y_pred[i])
                if y_pred[i] == 2:
                    correct_action_count += 1
            elif predict[0][0] > 0.4:
                # print("Short", predict, y_pred[i])
                if y_pred[i] == 0:
                    correct_action_count += 1
            else:
                # print("Hold", predict, y_pred[i])
                hold_action_count += 1
        print("Action count: ", X_pred.shape[0] - self.max_encode_length - hold_action_count)
        print("Correct action count: ", correct_action_count)
        print("Hold action count: ", hold_action_count)
    
    def set_seed(self, seed):
        """
        Set seed for reproducibility
        """
        np.random.seed(seed)