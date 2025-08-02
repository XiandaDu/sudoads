import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class OlsModel:
    def __init__(self):
        self.train_data_path = "D:/avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)

    def get_all_symbol_list(self):
        parquet_name_list = os.listdir(self.train_data_path)
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
            df = df.astype(np.float64)
            df["vwap"] = (
                (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan).ffill()
            )
        except Exception as e:
            print(f"get_single_symbol_kline_data error: {e}")
            df = pd.DataFrame()
        return df

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        pool = mp.Pool(mp.cpu_count() - 2)
        all_symbol_list = self.get_all_symbol_list()
        df_list = [
            pool.apply_async(self.get_single_symbol_kline_data, (symbol,))
            for symbol in all_symbol_list
        ]
        pool.close()
        pool.join()

        df_open_price = pd.concat([i.get()["open_price"] for i in df_list], axis=1).sort_index()
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        open_price_arr = df_open_price.values.astype(float)
        high_price_arr = pd.concat([i.get()["high_price"] for i in df_list], axis=1).sort_index().values
        low_price_arr = pd.concat([i.get()["low_price"] for i in df_list], axis=1).sort_index().values
        close_price_arr = pd.concat([i.get()["close_price"] for i in df_list], axis=1).sort_index().values
        vwap_arr = pd.concat([i.get()["vwap"] for i in df_list], axis=1).sort_index().values
        amount_arr = pd.concat([i.get()["amount"] for i in df_list], axis=1).sort_index().values

        print(f"finished get all symbols kline, time escaped {datetime.datetime.now() - t0}")
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
        n = len(y_true)
        r_true = pd.Series(y_true).rank(ascending=False, method="average")
        r_pred = pd.Series(y_pred).rank(ascending=False, method="average")
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x**2
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true) ** 2).sum()
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()
        return cov / np.sqrt(var_true * var_pred)

    def train(self, df_vwap, df_open, df_high, df_low, df_close, df_amount):
        windows_1d = 4 * 24
        windows_7d = windows_1d * 7

        ret_15min = df_vwap / df_vwap.shift(1) - 1
        ret_1d = df_vwap / df_vwap.shift(windows_1d) - 1
        ret_7d = df_vwap / df_vwap.shift(windows_7d) - 1
        vol_1d = ret_15min.rolling(windows_1d).std()
        vol_7d = ret_15min.rolling(windows_7d).std()
        trend_diff = ret_1d - ret_7d
        price_range_ratio = (df_high - df_low) / df_close
        candle_body_ratio = (df_close - df_open) / (df_high - df_low + 1e-6)
        vwap_deviation = (df_vwap - df_close) / df_close
        log_amount_diff = np.log1p(df_amount) - np.log1p(df_amount.shift(windows_1d))
        target = ret_1d.shift(-windows_1d)

        range_ = df_high - df_low
        normalized_close = (df_close - df_low) / (range_ + 1e-6)
        intraday_risk = ((range_) / df_close) ** 2
        closing_strength = (df_close - df_open) / (range_ + 1e-6)
        log_range_body_strength = np.log(df_high / df_low + 1e-6) * abs(df_close - df_open)

        upper_shadow = df_high - np.maximum(df_open, df_close)
        lower_shadow = np.minimum(df_open, df_close) - df_low
        shadow_ratio = (upper_shadow + lower_shadow) / (abs(df_close - df_open) + 1e-6)

        lowest_low_14 = df_low.rolling(window=14).min()
        highest_high_14 = df_high.rolling(window=14).max()
        stochastic_k = (df_close - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-6)
        stochastic_d = stochastic_k.rolling(window=3).mean()

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
            "range": range_,
            "normalized_close": normalized_close,
            "intraday_risk": intraday_risk,
            "closing_strength": closing_strength,
            "log_range_body_strength": log_range_body_strength,
            "shadow_ratio": shadow_ratio,
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
        }

        df_list = [f.stack().rename(k) for k, f in features.items()]
        df_all = pd.concat(df_list + [target.stack().rename("target")], axis=1).dropna()
        X = df_all[df_all.columns.difference(["target"])]
        y = df_all["target"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist",
                verbosity=0,
            ))
        ])

        model.fit(X, y)
        df_all["y_pred"] = model.predict(X)

        df_submit = df_all.reset_index().rename(columns={"level_0": "datetime", "level_1": "symbol"})
        df_submit = df_submit[df_submit["datetime"] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[["id", "y_pred"]].rename(columns={"y_pred": "predict_return"})

        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competion = df_submit[df_submit["id"].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competion["id"]))
        new_rows = pd.DataFrame({"id": missing_elements, "predict_return": [0] * len(missing_elements)})
        df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
        df_submit_competion.to_csv("./result/submit.csv", index=False)

        df_check = df_all.reset_index().rename(columns={"level_0": "datetime", "level_1": "symbol"})
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
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)

        self.train(df_vwap, df_open, df_high, df_low, df_close, df_amount)


if __name__ == "__main__":
    model = OlsModel()
    model.run()
