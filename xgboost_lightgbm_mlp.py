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
        import warnings, gc, math, time

        warnings.filterwarnings("ignore")

        # --- libs needed only inside train() to respect your "only change train/run" rule ---
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBRanker
        import lightgbm as lgb
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------
        # 1) Features & target (kept from your original code, minimal edits)
        # -----------------------
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
        log_range_body_strength = np.log(df_high / df_low + 1e-6) * abs(
            df_close - df_open
        )

        upper_shadow = df_high - np.maximum(df_open, df_close)
        lower_shadow = np.minimum(df_open, df_close) - df_low
        shadow_ratio = (upper_shadow + lower_shadow) / (abs(df_close - df_open) + 1e-6)

        lowest_low_14 = df_low.rolling(window=14).min()
        highest_high_14 = df_high.rolling(window=14).max()
        stochastic_k = (df_close - lowest_low_14) / (
            highest_high_14 - lowest_low_14 + 1e-6
        )
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

        # Stack to long
        df_list = [f.stack().rename(k) for k, f in features.items()]
        df_all = pd.concat(df_list + [target.stack().rename("target")], axis=1).dropna()

        # Build groups by timestamp (cross-sectional rank per time)
        ts_index = df_all.index.get_level_values(0)
        group_sizes = pd.Series(ts_index).value_counts().sort_index()
        # bring to aligned order with df_all
        group_sizes = (
            pd.Series(ts_index).groupby(ts_index).size().values
        )  # in order of appearance

        # Train/valid split by time (80/20)
        unique_times = np.array(sorted(pd.Index(ts_index).unique()))
        split_point = int(0.8 * len(unique_times))
        train_times = set(unique_times[:split_point])
        valid_times = set(unique_times[split_point:])

        is_train = ts_index.isin(train_times)
        is_valid = ts_index.isin(valid_times)

        X_cols = df_all.columns.difference(["target"])
        X = df_all[X_cols].astype(np.float32)
        y = df_all["target"].astype(np.float32)

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X[is_train]), index=X[is_train].index, columns=X_cols
        )
        X_valid = pd.DataFrame(
            scaler.transform(X[is_valid]), index=X[is_valid].index, columns=X_cols
        )

        y_train = y[is_train].values
        y_valid = y[is_valid].values

        # Build group arrays for XGB/LGB in train/valid
        def groups_from_mask(mask):
            times = ts_index[mask]
            return pd.Series(times).groupby(times).size().values

        group_train = groups_from_mask(is_train)
        group_valid = groups_from_mask(is_valid)

        # -----------------------
        # 2) XGBoost Ranker
        # -----------------------
        xgb_ranker = XGBRanker(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="rank:pairwise",
            eval_metric="ndcg",
        )
        xgb_ranker.fit(
            X_train.values,
            y_train,
            group=group_train,
            eval_set=[(X_valid.values, y_valid)],
            eval_group=[group_valid],
            verbose=False,
        )
        xgb_pred_val = xgb_ranker.predict(X_valid.values)

        # -----------------------
        # 3) LightGBM Ranker
        # -----------------------
        lgb_ranker = lgb.LGBMRanker(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=123,
            objective="lambdarank",
            metric="ndcg",
        )
        lgb_ranker.fit(
            X_train,
            y_train,
            group=group_train.tolist(),
            eval_set=[(X_valid, y_valid)],
            eval_group=[group_valid.tolist()],
            verbose=False,
        )
        lgb_pred_val = lgb_ranker.predict(X_valid)

        # -----------------------
        # 4) MLP with Spearman-style correlation loss
        #    We correlate predictions with z-scored ranks of the target *within each timestamp*.
        # -----------------------
        # prepare tensors for train/valid
        def make_group_batches(Xdf, yvec, mask):
            # returns batches where each batch = one timestamp group
            tvals = ts_index[mask]
            X_ = Xdf.values
            y_ = yvec
            # locate contiguous blocks by timestamp in Xdf.index (same order as mask)
            times = Xdf.index.get_level_values(0).to_numpy()
            # build indices for each group:
            batches = []
            start = 0
            while start < len(times):
                t0 = times[start]
                end = start
                while end < len(times) and times[end] == t0:
                    end += 1
                batches.append((start, end))
                start = end
            return X_, y_, batches

        Xtr_np, ytr_np, tr_batches = make_group_batches(X_train, y_train, is_train)
        Xva_np, yva_np, va_batches = make_group_batches(X_valid, y_valid, is_valid)

        class MLP(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        def corr_loss(pred, target):
            # pred/target are 1D tensors from a single timestamp group
            pred = pred - pred.mean()
            target = target - target.mean()
            pred_std = pred.std(unbiased=False) + 1e-8
            targ_std = target.std(unbiased=False) + 1e-8
            corr = (pred * target).mean() / (pred_std * targ_std)
            return 1.0 - corr  # maximize correlation -> minimize 1 - corr

        # Precompute group-wise z-scored *ranks* for y (ranks are constants => no gradient issue)
        def zrank_group(y_slice):
            import scipy.stats as st

            r = st.rankdata(y_slice)  # ascending; Spearman uses ranks of y
            r = torch.tensor(r, dtype=torch.float32, device=device)
            r = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
            return r

        mlp = MLP(X_train.shape[1]).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        mlp.train()
        epochs = 10  # short & sweet; adjust if you want
        for ep in range(epochs):
            total = 0.0
            for s, e in tr_batches:
                x = torch.tensor(Xtr_np[s:e], dtype=torch.float32, device=device)
                y_group = ytr_np[s:e]
                y_rank = zrank_group(y_group)
                opt.zero_grad()
                pred = mlp(x)
                loss = corr_loss(pred, y_rank)
                loss.backward()
                opt.step()
                total += loss.item()
            # optional tiny valid check
            # print(f"MLP epoch {ep}: loss={total/len(tr_batches):.4f}")

        mlp.eval()
        with torch.no_grad():
            mlp_pred_val = []
            for s, e in va_batches:
                x = torch.tensor(Xva_np[s:e], dtype=torch.float32, device=device)
                mlp_pred_val.append(mlp(x).cpu().numpy())
            mlp_pred_val = np.concatenate(mlp_pred_val, axis=0)

        # -----------------------
        # 5) Blend weights by maximizing Spearman on validation
        # -----------------------
        def spearman_weighted(y_true, y_pred):
            # use your helper (weighted) if desired; plain Spearman per-sample also OK.
            # Here we do weighted Spearman you defined elsewhere in this class:contentReference[oaicite:2]{index=2}.
            return self.weighted_spearmanr(y_true, y_pred)

        best_rho, best_w = -1.0, (0.0, 0.0, 1.0)
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1 - w1, 11):
                w3 = 1 - w1 - w2
                pred = w1 * xgb_pred_val + w2 * lgb_pred_val + w3 * mlp_pred_val
                rho = spearman_weighted(y_valid, pred)
                if rho > best_rho:
                    best_rho, best_w = rho, (w1, w2, w3)

        print(
            f"Best blend on valid (weighted Spearman): {best_rho:.4f} with weights XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, MLP={best_w[2]:.2f}"
        )

        # -----------------------
        # 6) Refit models on ALL data, then predict for submission
        # -----------------------
        X_all = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X_cols)
        y_all = y.values

        # rebuild group sizes for ALL
        group_all = (
            pd.Series(X_all.index.get_level_values(0))
            .groupby(lambda x: x)
            .size()
            .values
        )

        # Refit XGB/LGB on all
        xgb_ranker_all = XGBRanker(
            n_estimators=xgb_ranker.get_params()["n_estimators"],
            max_depth=xgb_ranker.get_params()["max_depth"],
            learning_rate=xgb_ranker.get_params()["learning_rate"],
            subsample=xgb_ranker.get_params()["subsample"],
            colsample_bytree=xgb_ranker.get_params()["colsample_bytree"],
            tree_method="hist",
            random_state=42,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="rank:pairwise",
            eval_metric="ndcg",
        )
        xgb_ranker_all.fit(X_all.values, y_all, group=group_all, verbose=False)
        lgb_ranker_all = lgb.LGBMRanker(
            n_estimators=lgb_ranker.get_params()["n_estimators"],
            learning_rate=lgb_ranker.get_params()["learning_rate"],
            num_leaves=lgb_ranker.get_params()["num_leaves"],
            subsample=lgb_ranker.get_params()["subsample"],
            colsample_bytree=lgb_ranker.get_params()["colsample_bytree"],
            reg_lambda=lgb_ranker.get_params()["reg_lambda"],
            random_state=123,
            objective="lambdarank",
            metric="ndcg",
        )
        lgb_ranker_all.fit(X_all, y_all, group=group_all.tolist(), verbose=False)

        # Refit MLP quickly on all (few epochs)
        Xall_np = X_all.values
        times_all = X_all.index.get_level_values(0).to_numpy()
        # build groups for all
        batches_all = []
        start = 0
        while start < len(times_all):
            t0 = times_all[start]
            end = start
            while end < len(times_all) and times_all[end] == t0:
                end += 1
            batches_all.append((start, end))
            start = end

        mlp_all = MLP(X_all.shape[1]).to(device)
        opt_all = torch.optim.Adam(mlp_all.parameters(), lr=1e-3)
        mlp_all.train()
        for ep in range(5):
            for s, e in batches_all:
                x = torch.tensor(Xall_np[s:e], dtype=torch.float32, device=device)
                y_rank = zrank_group(y_all[s:e])
                opt_all.zero_grad()
                loss = corr_loss(mlp_all(x), y_rank)
                loss.backward()
                opt_all.step()
        mlp_all.eval()

        # Predictions on ALL rows
        xgb_pred_all = xgb_ranker_all.predict(X_all.values)
        lgb_pred_all = lgb_ranker_all.predict(X_all)
        with torch.no_grad():
            preds_all_mlp = []
            for s, e in batches_all:
                x = torch.tensor(Xall_np[s:e], dtype=torch.float32, device=device)
                preds_all_mlp.append(mlp_all(x).cpu().numpy())
            mlp_pred_all = np.concatenate(preds_all_mlp, axis=0)

        w1, w2, w3 = best_w
        df_all["y_pred"] = w1 * xgb_pred_all + w2 * lgb_pred_all + w3 * mlp_pred_all

        # -----------------------
        # 7) Submission + check + report Spearman
        # -----------------------
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
        df_submit_comp = df_submit[df_submit["id"].isin(id_list)]
        missing = list(set(id_list) - set(df_submit_comp["id"]))
        if missing:
            df_submit_comp = pd.concat(
                [df_submit_comp, pd.DataFrame({"id": missing, "predict_return": 0.0})],
                ignore_index=True,
            )
        df_submit_comp.to_csv("./result/submit.csv", index=False)

        # for offline check
        df_check = df_all.reset_index().rename(
            columns={"level_0": "datetime", "level_1": "symbol"}
        )
        df_check = df_check[df_check["datetime"] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[["id", "target"]].rename(columns={"target": "true_return"})
        df_check.to_csv("./result/check.csv", index=False)

        rho_w = self.weighted_spearmanr(df_all["target"], df_all["y_pred"])
        print(f"Weighted Spearman correlation coefficient (ALL): {rho_w:.4f}")

        # save for run() if needed
        self._last_submit = df_submit_comp
        self._last_rho = rho_w

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

        # Train + export
        self.train(df_vwap, df_open, df_high, df_low, df_close, df_amount)


if __name__ == "__main__":
    model = OlsModel()
    model.run()
