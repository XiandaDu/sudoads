import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class OlsModel:
    def __init__(self):
        # the folder path for setting sequence data
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
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
            df["vwap"] = (
                (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan).ffill()
            )
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
            df_list.append(
                pool.apply_async(
                    self.get_single_symbol_kline_data, (all_symbol_list[i],)
                )
            )
        # the process pool is closed and no new tasks will be accepted
        pool.close()
        # wait for all asynchronous tasks to complete
        pool.join()
        # collect the opening price series of all asynchronous results and concatenate them into a DataFrame by columns, then sort the index in ascending order of time
        df_open_price = pd.concat(
            [i.get()["open_price"] for i in df_list], axis=1
        ).sort_index(ascending=True)
        # convert the time index (milliseconds) to a datetime type array
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        # get the values from the opening price in the DataFrame and convert them into a NumPy array of float type
        open_price_arr = df_open_price.values.astype(float)
        # get the values from the highest price in the DataFrame and convert them into a NumPy array of float type
        high_price_arr = (
            pd.concat([i.get()["high_price"] for i in df_list], axis=1)
            .sort_index(ascending=True)
            .values
        )
        # get the values from the lowest price in the DataFrame and convert them into a NumPy array of float type
        low_price_arr = (
            pd.concat([i.get()["low_price"] for i in df_list], axis=1)
            .sort_index(ascending=True)
            .values
        )
        # get the values from the closing price in the DataFrame and convert them into a NumPy array of float type
        close_price_arr = (
            pd.concat([i.get()["close_price"] for i in df_list], axis=1)
            .sort_index(ascending=True)
            .values
        )
        # collect the volume-weighted average price series of all currencies and concatenate them into an array by columns
        vwap_arr = (
            pd.concat([i.get()["vwap"] for i in df_list], axis=1)
            .sort_index(ascending=True)
            .values
        )
        # collect the trading amount series of all currencies and concatenate them into an array by columns
        amount_arr = (
            pd.concat([i.get()["amount"] for i in df_list], axis=1)
            .sort_index(ascending=True)
            .values
        )
        print(
            f"finished get all symbols kline, time escaped {datetime.datetime.now() - t0}"
        )
        return (
            all_symbol_list,
            time_arr,
            open_price_arr,
            high_price_arr,
            low_price_arr,
            close_price_arr,
            vwap_arr,
            amount_arr,
        )

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
        r_true = pd.Series(y_true).rank(ascending=False, method="average")
        # rank the predicted values in descending order (average method for handling ties)
        r_pred = pd.Series(y_pred).rank(ascending=False, method="average")

        # normalize the index i = rank - 1, mapped to [-1, 1]
        x = 2 * (r_true - 1) / (n - 1) - 1
        # weight w_i (the weight factor for each sample)
        w = x**2

        # weighted mean
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum

        # calculate the weighted covariance
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        # calculate the weighted variance of the true value rankings
        var_true = (w * (r_true - mu_true) ** 2).sum()
        # calculate the weighted variance of the predicted value rankings
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()

        # return the weighted Spearman correlation coefficient
        return cov / np.sqrt(var_true * var_pred)

    def train(self, df_vwap, df_open, df_high, df_low, df_close, df_amount):
        windows_1d = 4 * 24
        windows_7d = windows_1d * 7

        # 基础收益与波动率特征
        ret_15min = df_vwap / df_vwap.shift(1) - 1
        ret_1d = df_vwap / df_vwap.shift(windows_1d) - 1
        ret_7d = df_vwap / df_vwap.shift(windows_7d) - 1
        vol_1d = ret_15min.rolling(windows_1d).std()
        vol_7d = ret_15min.rolling(windows_7d).std()
        trend_diff = ret_1d - ret_7d

        # 形态特征
        intraday_range = df_high - df_low
        norm_close_pos = (df_close - df_low) / (intraday_range + 1e-6)
        closing_strength = (df_close - df_open).abs() / (intraday_range + 1e-6)
        logrange_body_strength = (
            np.log(df_high / df_low + 1e-6) * (df_close - df_open).abs()
        )

        upper_shadow = df_high - df_open.combine(df_close, func=max)
        lower_shadow = df_open.combine(df_close, func=min) - df_low
        body = (df_close - df_open).abs()
        shadow_ratio = (upper_shadow + lower_shadow) / (body + 1e-6)

        # KDJ 动量指标
        low_n = df_low.rolling(windows_7d).min()
        high_n = df_high.rolling(windows_7d).max()
        k_val = 100 * (df_close - low_n) / (high_n - low_n + 1e-6)
        d_val = k_val.rolling(3).mean()

        # 量能相关
        price_range_ratio = intraday_range / df_close
        candle_body_ratio = (df_close - df_open) / (intraday_range + 1e-6)
        vwap_deviation = (df_vwap - df_close) / df_close
        log_amount_diff = np.log1p(df_amount) - np.log1p(df_amount.shift(windows_1d))

        # 目标值：未来一天收益
        target = ret_1d.shift(-windows_1d)

        features = {
            "ret_1d": ret_1d,
            "ret_7d": ret_7d,
            "vol_1d": vol_1d,
            "vol_7d": vol_7d,
            "trend_diff": trend_diff,
            "price_range_ratio": price_range_ratio,
            "candle_body_ratio": candle_body_ratio,
            "vwap_dev": vwap_deviation,
            "log_amount_diff": log_amount_diff,
            "intraday_range": intraday_range,
            "norm_close_pos": norm_close_pos,
            "closing_strength": closing_strength,
            "logrange_body_strength": logrange_body_strength,
            "shadow_ratio": shadow_ratio,
            "kdj_k": k_val,
            "kdj_d": d_val,
        }

        df_list = [f.stack().rename(k) for k, f in features.items()]
        df_all = pd.concat(df_list + [target.stack().rename("target")], axis=1).dropna()
        X = df_all[df_all.columns.difference(["target"])]
        y = df_all["target"]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42,
                        learning_rate_init=0.001,
                        verbose=True,
                    ),
                ),
            ]
        )
        model.fit(X, y)
        df_all["y_pred"] = model.predict(X)

        df_submit = df_all.reset_index().rename(
            columns={"level_0": "datetime", "level_1": "symbol"}
        )
        df_submit = df_submit[df_submit["datetime"] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[["id", "y_pred"]].rename(
            columns={"y_pred": "predict_return"}
        )

        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competion = df_submit[df_submit["id"].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competion["id"]))
        new_rows = pd.DataFrame(
            {"id": missing_elements, "predict_return": [0] * len(missing_elements)}
        )
        df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
        df_submit_competion.to_csv("./result/submit.csv", index=False)

        df_check = df_all.reset_index().rename(
            columns={"level_0": "datetime", "level_1": "symbol"}
        )
        df_check = df_check[df_check["datetime"] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[["id", "target"]].rename(columns={"target": "true_return"})
        df_check.to_csv("./result/check.csv", index=False)

        rho = self.weighted_spearmanr(df_all["target"], df_all["y_pred"])
        print(f"Weighted Spearman correlation coefficient: {rho:.4f}")

    def run(self):
        (
            all_symbol_list,
            time_arr,
            open_price_arr,
            high_price_arr,
            low_price_arr,
            close_price_arr,
            vwap_arr,
            amount_arr,
        ) = self.get_all_symbol_kline()

        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_open = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr)
        df_close = pd.DataFrame(
            close_price_arr, columns=all_symbol_list, index=time_arr
        )
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)

        self.train(df_vwap, df_open, df_high, df_low, df_close, df_amount)


if __name__ == "__main__":
    model = OlsModel()
    model.run()
