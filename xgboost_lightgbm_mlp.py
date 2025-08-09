import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression


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
        """
        两模融合（LightGBM + XGBoost），仅修改 train 函数内容。
        - 特征：在原 XGBoost 版本的基础上，融合了 V7 中效果好的波动/价差/效率/位置类特征（做了简化，避免过重）。
        - 训练：只用部分币种训练（默认 35%），降低内存占用；预测时分批遍历全部币种。
        - 后处理：分位数裁剪 + 轻微波动归一，避免极端值（利于加权Spearman）。
        - 输出：./result/submit.csv（严格匹配 submission_id.csv，缺失补0），./result/check.csv
        """
        import os, gc, warnings

        warnings.filterwarnings("ignore")
        os.makedirs("./result", exist_ok=True)

        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor
        import lightgbm as lgb

        # ========= 通用设置 =========
        dtype = np.float32
        for _df in [df_vwap, df_open, df_high, df_low, df_close, df_amount]:
            _df[:] = _df.astype(dtype)

        windows_1d = 4 * 24
        windows_7d = windows_1d * 7

        # ========= 仅用于评估的目标（未来24h收益，按PDF与样例代码定义） =========
        ret_1d = df_vwap / df_vwap.shift(windows_1d) - 1.0
        target_full = ret_1d.shift(-windows_1d).astype(dtype)

        # ========= 内嵌特征函数（按币种子集计算，降低内存） =========
        def compute_features_for(symbols):
            vwap = df_vwap[symbols]
            open_ = df_open[symbols]
            high = df_high[symbols]
            low = df_low[symbols]
            close = df_close[symbols]
            amount = df_amount[symbols]

            # 基础收益/波动
            ret15 = vwap.pct_change(1)
            f_ret_1d = vwap.pct_change(windows_1d)
            f_ret_7d = vwap.pct_change(windows_7d)
            f_vol_1d = ret15.rolling(windows_1d, min_periods=windows_1d // 2).std()
            f_vol_7d = ret15.rolling(windows_7d, min_periods=windows_7d // 2).std()

            # 价差/形态
            range_ = high - low
            f_price_range_ratio = range_ / (close + 1e-6)
            f_candle_body_ratio = (close - open_) / (range_ + 1e-6)
            f_vwap_dev = (vwap - close) / (close + 1e-6)

            # 来自 V7 的简化增强
            # 高低价差的多窗口均值与波动（缩减窗口避免内存占用）
            hl_spread = (high - low) / (vwap + 1e-6)
            f_hl_mean_96 = hl_spread.rolling(96, min_periods=48).mean()
            f_hl_std_96 = hl_spread.rolling(96, min_periods=48).std()
            f_hl_mean_336 = hl_spread.rolling(336, min_periods=168).mean()
            f_hl_std_336 = hl_spread.rolling(336, min_periods=168).std()

            # 效率比（96）
            net_change = (vwap - vwap.shift(96)).abs()
            total_change = vwap.diff().abs().rolling(96, min_periods=48).sum()
            f_eff_96 = net_change / (total_change + 1e-8)

            # 价格位置（336）
            hi_336 = high.rolling(336, min_periods=168).max()
            lo_336 = low.rolling(336, min_periods=168).min()
            f_pos_336 = (vwap - lo_336) / (hi_336 - lo_336 + 1e-8)

            # 返回窗口（336）
            f_ret_336 = vwap.pct_change(336)

            # 量能（美元量均值，336）
            volume_usd = amount * vwap
            f_volusd_336 = volume_usd.rolling(336, min_periods=168).mean()

            # RSI(96)
            delta = vwap.diff()
            gain = delta.clip(lower=0).rolling(96, min_periods=48).mean()
            loss = (-delta.clip(upper=0)).rolling(96, min_periods=48).mean()
            rs = gain / (loss + 1e-8)
            f_rsi_96 = 100 - (100 / (1 + rs))

            # Parkinson 波动(96)
            f_parkinson_96 = np.sqrt(
                ((np.log(high / low + 1e-8)) ** 2 / (4 * np.log(2)))
                .rolling(96, min_periods=48)
                .mean()
            )

            # 原脚本里的额外形态指标（轻量）
            normalized_close = (close - low) / (range_ + 1e-6)
            intraday_risk = (range_ / (close + 1e-6)) ** 2
            closing_strength = (close - open_) / (range_ + 1e-6)

            # 目标
            y = target_full[symbols]

            feats = {
                "ret_1d": f_ret_1d,
                "ret_7d": f_ret_7d,
                "vol_1d": f_vol_1d,
                "vol_7d": f_vol_7d,
                "price_range_ratio": f_price_range_ratio,
                "candle_body_ratio": f_candle_body_ratio,
                "vwap_dev": f_vwap_dev,
                "hl_mean_96": f_hl_mean_96,
                "hl_std_96": f_hl_std_96,
                "hl_mean_336": f_hl_mean_336,
                "hl_std_336": f_hl_std_336,
                "eff_96": f_eff_96,
                "pos_336": f_pos_336,
                "ret_336": f_ret_336,
                "volusd_336": f_volusd_336,
                "rsi_96": f_rsi_96,
                "parkinson_96": f_parkinson_96,
                "normalized_close": normalized_close,
                "intraday_risk": intraday_risk,
                "closing_strength": closing_strength,
            }

            return feats, y

        # ========= 训练子集（默认取 35% 币种，固定随机种子，兼顾内存与泛化） =========
        symbols = list(df_vwap.columns)
        rng = np.random.RandomState(42)
        train_symbols = sorted(
            rng.choice(
                symbols, size=max(1, int(len(symbols) * 0.35)), replace=False
            ).tolist()
        )

        feats_tr, y_tr_wide = compute_features_for(train_symbols)

        # 构造训练长表（只取训练币种，按时间×币种堆叠；在这里 dropna 避免不同窗口的 NaN）
        feat_names = list(feats_tr.keys())
        stacked_feats = [
            feats_tr[n].stack().rename(n).astype(dtype) for n in feat_names
        ]
        y_tr = y_tr_wide.stack().rename("target").astype(dtype)
        df_tr = pd.concat(stacked_feats + [y_tr], axis=1).dropna()

        X_train = df_tr[feat_names].values.astype(dtype)
        y_train = df_tr["target"].values.astype(dtype)

        # ========= 训练两模型 =========
        n_jobs = max(1, os.cpu_count() - 2)

        # XGBoost：直方图算法，收缩学习率，适度树深，内存友好
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            tree_method="hist",
            n_jobs=n_jobs,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)

        # LightGBM：来源于V7的保守化参数（避免爆内存），并限制线程
        lgbm = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=163,
            learning_rate=0.07,
            feature_fraction=0.82,
            bagging_fraction=0.70,
            bagging_freq=2,
            reg_alpha=0.71,
            reg_lambda=0.09,
            min_data_in_leaf=139,
            max_depth=10,
            n_estimators=350,
            subsample_for_bin=200000,
            n_jobs=n_jobs,
            verbose=-1,
            force_row_wise=True,
        )
        lgbm.fit(X_train, y_train)

        # ========= 可选：用时间切分的简单验证来确定融合权重 =========
        try:
            # 按时间分位切分（保证时序）
            cutoff = df_tr.index.get_level_values(0).unique().quantile(0.8)
            valid_idx = df_tr.index.get_level_values(0) >= cutoff
            X_val = df_tr.loc[valid_idx, feat_names].values.astype(dtype)
            y_val = df_tr.loc[valid_idx, "target"].values.astype(dtype)

            p_xgb = xgb.predict(X_val).astype(dtype)
            p_lgb = lgbm.predict(X_val).astype(dtype)

            # 网格搜一个简单的线性融合系数，最大化加权Spearman
            best_w, best_score = 0.6, -9.9
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                p_blend = w * p_lgb + (1 - w) * p_xgb
                score = self.weighted_spearmanr(y_val, p_blend)
                if score > best_score:
                    best_w, best_score = w, score
            w_lgb = best_w
        except Exception:
            # 回退：固定权重
            w_lgb = 0.6
        w_xgb = 1.0 - w_lgb

        # ========= 全量预测：按币种分批堆叠，避免一次性爆内存 =========
        results = []
        chunk = 25  # 25~40 之间都可以；越小越省内存
        for i in range(0, len(symbols), chunk):
            syms = symbols[i : i + chunk]
            feats_ck, y_ck_wide = compute_features_for(syms)

            # 堆叠 & 对齐
            stk = [feats_ck[n].stack().rename(n).astype(dtype) for n in feat_names]
            y_ck = y_ck_wide.stack().rename("target").astype(dtype)
            df_ck = pd.concat(stk + [y_ck], axis=1).dropna()

            if len(df_ck) == 0:
                continue

            X_ck = df_ck[feat_names].values.astype(dtype)

            # 双模预测 + 轻后处理
            pred = w_lgb * lgbm.predict(X_ck).astype(dtype) + w_xgb * xgb.predict(
                X_ck
            ).astype(dtype)

            # 分位数裁剪 + 目标波动缩放（~2%），抑制极端值
            lo, hi = np.percentile(pred, 0.25), np.percentile(pred, 99.75)
            pred = np.clip(pred, lo, hi).astype(dtype)
            std = pred.std()
            if std > 1e-8:
                pred = (pred - pred.mean()) / std * dtype(0.02)

            out = df_ck[["target"]].copy()
            out["y_pred"] = pred
            results.append(out)

            del feats_ck, y_ck_wide, stk, df_ck, X_ck, pred
            gc.collect()

        if len(results) == 0:
            raise RuntimeError(
                "No valid rows to predict after feature generation. Check data ranges/NaNs."
            )

        results_df = pd.concat(results, axis=0)

        # ========= 评估（In-sample） =========
        rho = self.weighted_spearmanr(
            results_df["target"].values, results_df["y_pred"].values
        )
        print(
            f"Weighted Spearman correlation coefficient (blend w_lgb={w_lgb:.2f}, w_xgb={w_xgb:.2f}): {rho:.4f}"
        )

        # ========= 生成提交 =========
        df_submit = results_df.reset_index().rename(
            columns={"level_0": "datetime", "level_1": "symbol"}
        )
        df_submit = df_submit[df_submit["datetime"] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[["id", "y_pred"]].rename(
            columns={"y_pred": "predict_return"}
        )

        # 对齐 submission_id.csv，缺失补0
        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()

        df_submit_comp = df_submit[df_submit["id"].isin(id_list)].copy()
        missing = list(set(id_list) - set(df_submit_comp["id"]))
        if missing:
            df_submit_comp = pd.concat(
                [df_submit_comp, pd.DataFrame({"id": missing, "predict_return": 0.0})],
                ignore_index=True,
            )

        # 排序对齐
        df_submit_comp = df_submit_comp.set_index("id").loc[id_list].reset_index()
        df_submit_comp.to_csv("./result/submit.csv", index=False)

        # 方便核对真值
        df_check = results_df.reset_index().rename(
            columns={"level_0": "datetime", "level_1": "symbol"}
        )
        df_check = df_check[df_check["datetime"] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[["id", "target"]].rename(columns={"target": "true_return"})
        df_check.to_csv("./result/check.csv", index=False)

    def run(self):
        """
        仅修改 run 函数内容：保持数据加载流程不变，转为 float32 以省内存，然后调用上面的融合训练。
        """
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

        import numpy as np
        import pandas as pd

        # 转 float32，减少一半内存占用
        df_vwap = pd.DataFrame(
            vwap_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )
        df_open = pd.DataFrame(
            open_price_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )
        df_high = pd.DataFrame(
            high_price_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )
        df_low = pd.DataFrame(
            low_price_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )
        df_close = pd.DataFrame(
            close_price_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )
        df_amount = pd.DataFrame(
            amount_arr.astype(np.float32), columns=all_symbol_list, index=time_arr
        )

        self.train(df_vwap, df_open, df_high, df_low, df_close, df_amount)


if __name__ == "__main__":
    model = OlsModel()
    model.run()
