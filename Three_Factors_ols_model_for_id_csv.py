import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression

class OlsModel:
    def __init__(self):
        # the folder path for setting sequence data
        self.train_data_path = "/data/wangyifei/First_round_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
    
    def get_all_symbol_list(self):
        # get a list of all file names in the training data directory
        parquet_name_list = os.listdir(self.train_data_path)
         # remove the file extension and keep only the currency code symbol to generate a list of currency codes
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list
    
    def get_single_symbol_kline_data(self, symbol):
        try:
            # read the specified cryptocurrency's Parquet file and obtain its K-line data as a DataFrame
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            # set the DataFrame's index to the "timestamp" column
            df = df.set_index("timestamp")
            # convert the data to 64-bit floating-point type.
            df = df.astype(np.float64)
            # calculate the volume-weighted average price (VWAP), handle infinite values, and fill them with the previous valid value
            df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
        except Exception as e:
            print(f"get_single_symbol_kline_data error: {e}")
            df = pd.DataFrame()
        return df
    
    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
         # create a process pool, using the number of available CPU cores minus 2, for parallel processing
        pool = mp.Pool(mp.cpu_count() - 2)
        # get a list of all currencies
        all_symbol_list = self.get_all_symbol_list()
         # the initialization list is used to store the results returned by each asynchronous read task
        df_list = []
        for i in range(len(all_symbol_list)):
            df_list.append(pool.apply_async(self.get_single_symbol_kline_data, (all_symbol_list[i], )))
        # the process pool is closed and no new tasks will be accepted
        pool.close()
        # wait for all asynchronous tasks to complete
        pool.join()
        # collect the opening price series of all asynchronous results and concatenate them into a DataFrame by columns, then sort the index in ascending order of time
        df_open_price = pd.concat([i.get()['open_price'] for i in df_list], axis=1).sort_index(ascending=True)
        # convert the time index (milliseconds) to a datetime type array
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit = "ms").values
        # get the values from the opening price in the DataFrame and convert them into a NumPy array of float type
        open_price_arr = df_open_price.values.astype(float)
        # get the values from the highest price in the DataFrame and convert them into a NumPy array of float type
        high_price_arr = pd.concat([i.get()['high_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # get the values from the lowest price in the DataFrame and convert them into a NumPy array of float type
        low_price_arr = pd.concat([i.get()['low_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # get the values from the closing price in the DataFrame and convert them into a NumPy array of float type
        close_price_arr = pd.concat([i.get()['close_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # collect the volume-weighted average price series of all currencies and concatenate them into an array by columns
        vwap_arr = pd.concat([i.get()['vwap'] for i in df_list], axis=1).sort_index(ascending=True).values
        # collect the trading amount series of all currencies and concatenate them into an array by columns
        amount_arr = pd.concat([i.get()['amount'] for i in df_list], axis=1).sort_index(ascending=True).values
        print(f"finished get all symbols kline, time escaped {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
    
    def weighted_spearmanr(self, y_true, y_pred):
        """
        Calculate the weighted Spearman correlation coefficient according to the formula in the appendix:
        1) Rank y_true and y_pred in descending order (rank=1 means the maximum value)
        2) Normalize the rank indices to [-1, 1], then square to obtain the weight w_i
        3) Calculate the correlation coefficient using the weighted Pearson formula
        """
        # number of samples
        n = len(y_true)
        # rank the true values in descending order (average method for handling ties)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        # rank the predicted values in descending order (average method for handling ties)
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        
        # normalize the index i = rank - 1, mapped to [-1, 1]
        x = 2 * (r_true - 1) / (n - 1) - 1
        # weight w_i (the weight factor for each sample)
        w = x ** 2  
        
        # weighted mean
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        
        # calculate the weighted covariance
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        # calculate the weighted variance of the true value rankings
        var_true = (w * (r_true - mu_true)**2).sum()
        # calculate the weighted variance of the predicted value rankings
        var_pred = (w * (r_pred - mu_pred)**2).sum()
        
        # return the weighted Spearman correlation coefficient
        return cov / np.sqrt(var_true * var_pred)

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        # pivot factor1 data to long format (multi-index: time and symbol)
        factor1_long = df_factor1.stack()
        # pivot factor2 data to long format
        factor2_long = df_factor2.stack()
        # pivot factor3 data to long format
        factor3_long = df_factor3.stack()
        # pivot target data to long format
        target_long = df_target.stack()
         # set the name of the factor1 series as 'factor1'
        factor1_long.name = 'factor1'
        # set the name of the factor1 series as 'factor2'
        factor2_long.name = 'factor2'
        # set the name of the factor1 series as 'factor3'
        factor3_long.name = 'factor3'
        # set the name of the target series as 'target'
        target_long.name = 'target'
        # merge the four series (factor1, factor2, factor3, target) into a single DataFrame by columns
        data = pd.concat([factor1_long, factor2_long, factor3_long, target_long], axis=1)
        # drop the rows with missing values (NaN)
        data = data.dropna()
        # construct the feature matrix X, including the three factor columns
        X = data[['factor1', 'factor2', 'factor3']]
        # construct the target variable y, which is the target value column
        y = data['target']
        # fit the model using LinearRegression (multiple linear regression)
        model = LinearRegression()
        model.fit(X, y)

        # output the coefficients and intercept of the regression model
        print("Linear regression model coefficients:", model.coef_)
        print("Linear regression model intercept:", model.intercept_)
        
        # add a column in the original data to store the model's predicted value for each sample
        data['y_pred'] = model.predict(X)
        df_submit = data.reset_index(level=0)
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        print(df_submit)

        df_submission_id = pd.read_csv("/kaggle/input/testing-for-crypto-3/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competion['id']))
        new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
        df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
        print(df_submit_competion.shape)
        df_submit_competion.to_csv("submit.csv", index = False)
        
        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        
        print(df_check)

        df_check.to_csv("check.csv", index = False)
        
        # calculate the weighted Spearman correlation coefficient on the entire sample
        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")
        
    
    def run(self):
        # call the get_all_symbol_kline function to get the K-line data and event data for all currencies
        all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr = self.get_all_symbol_kline()
        # convert the vwap array into a DataFrame, with currencies as columns and time as the index (next line sets the index)
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        # convert the amount array into a DataFrame, with currencies as columns and time as the index
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        # calculate the return for the past 24 hours using rolling calculation
        df_24hour_rtn = df_vwap / df_vwap.shift(windows_1d) - 1
        # calculate the return for the past 15 minutes using rolling calculation
        df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
        # calculate the first factor: 7-day volatility factor
        df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
        # calculate the second factor: 7-day momentum factor
        df_7d_momentum = df_vwap / df_vwap.shift(windows_7d) - 1
        # calculate the third factor: 7-day total volume factor
        df_amount_sum = df_amount.rolling(windows_7d).sum()
        # call the train method, using the lagged 7-day 24-hour return as the target value, and the three factors as inputs for model training
        self.train(df_24hour_rtn.shift(-windows_1d), df_7d_volatility, df_7d_momentum, df_amount_sum)   
        
        
        
if __name__ == '__main__':
    model = OlsModel()
    model.run()
