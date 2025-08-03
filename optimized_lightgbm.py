import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


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

    def create_advanced_features(self, df_vwap, df_open, df_high, df_low, df_close, df_amount):
        """Create advanced features for better prediction"""
        features_dict = {}
        
        # Multiple timeframe returns
        for window in [1, 2, 4, 8, 12, 24, 48, 96, 168, 336]:  # 15min to 2 weeks
            features_dict[f'return_{window}'] = df_vwap / df_vwap.shift(window) - 1
        
        # Volatility features
        for window in [24, 48, 96, 168, 336, 672]:  # 6h to 1 week
            returns_15min = df_vwap / df_vwap.shift(1) - 1
            features_dict[f'volatility_{window}'] = returns_15min.rolling(window).std(ddof=1)
            features_dict[f'volatility_skew_{window}'] = returns_15min.rolling(window).skew()
            features_dict[f'volatility_kurt_{window}'] = returns_15min.rolling(window).kurt()
        
        # Price range features
        for window in [24, 96, 168, 336]:
            features_dict[f'price_range_{window}'] = (
                df_high.rolling(window).max() - df_low.rolling(window).min()
            ) / df_vwap
            features_dict[f'high_low_ratio_{window}'] = (
                df_high.rolling(window).mean() / df_low.rolling(window).mean() - 1
            )
        
        # Volume features
        for window in [24, 96, 168, 336, 672]:
            features_dict[f'volume_sum_{window}'] = df_amount.rolling(window).sum()
            features_dict[f'volume_mean_{window}'] = df_amount.rolling(window).mean()
            features_dict[f'volume_std_{window}'] = df_amount.rolling(window).std()
            # Volume ratio
            features_dict[f'volume_ratio_{window}'] = (
                df_amount.rolling(window).mean() / df_amount.rolling(window * 2).mean()
            )
        
        # Technical indicators
        # RSI
        for window in [24, 96, 336]:
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features_dict[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Moving averages and crossovers
        for short, long in [(24, 96), (96, 336), (168, 672)]:
            ma_short = df_vwap.rolling(short).mean()
            ma_long = df_vwap.rolling(long).mean()
            features_dict[f'ma_ratio_{short}_{long}'] = ma_short / ma_long - 1
        
        # Price position within range
        for window in [96, 336, 672]:
            high_window = df_high.rolling(window).max()
            low_window = df_low.rolling(window).min()
            features_dict[f'price_position_{window}'] = (
                (df_vwap - low_window) / (high_window - low_window)
            )
        
        # Momentum features
        for lag in [24, 96, 168, 336]:
            features_dict[f'momentum_{lag}'] = df_vwap / df_vwap.shift(lag) - 1
        
        # Cross-sectional features (rank-based)
        # Create a copy of keys to avoid RuntimeError
        cross_sectional_features = {}
        for key, feat in list(features_dict.items()):  # Convert to list to avoid iteration error
            if 'return' in key or 'momentum' in key:
                # Rank across assets at each time point
                cross_sectional_features[f'{key}_rank'] = feat.rank(axis=1, pct=True)
                # Z-score normalization across assets
                cross_sectional_features[f'{key}_zscore'] = (
                    (feat - feat.mean(axis=1).values[:, None]) / 
                    feat.std(axis=1).values[:, None]
                )
        
        # Add cross-sectional features to main dict
        features_dict.update(cross_sectional_features)
        
        return features_dict

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        """Enhanced training function using LightGBM Ranker"""
        print("Starting enhanced training with LightGBM Ranker...")
        
        # Get additional data for feature engineering
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
        
        # Create DataFrames from arrays
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_open = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr)
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        
        # Create advanced features
        print("Creating advanced features...")
        features_dict = self.create_advanced_features(
            df_vwap, df_open, df_high, df_low, df_close, df_amount
        )
        
        # Add original factors
        features_dict['volatility_7d'] = df_factor1
        features_dict['momentum_7d'] = df_factor2
        features_dict['volume_sum_7d'] = df_factor3
        
        # Stack all features to long format
        print("Preparing data for training...")
        feature_dfs = []
        for name, feat in features_dict.items():
            feat_long = feat.stack()
            feat_long.name = name
            feature_dfs.append(feat_long)
        
        # Stack target
        target_long = df_target.stack()
        target_long.name = 'target'
        
        # Combine all data
        data = pd.concat(feature_dfs + [target_long], axis=1)
        data = data.dropna()
        
        # Split data for validation (last 20% for validation)
        n_samples = len(data)
        train_size = int(0.8 * n_samples)
        
        # Sort by time to ensure temporal split
        data = data.sort_index()
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]
        
        # Prepare features and targets
        feature_cols = [col for col in data.columns if col != 'target']
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_val = val_data[feature_cols]
        y_val = val_data['target']
        
        # Create query groups for ranking (group by timestamp)
        train_groups = train_data.groupby(level=0).size().values
        val_groups = val_data.groupby(level=0).size().values
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Train LightGBM Ranker
        print("Training LightGBM Ranker...")
        lgb_train = lgb.Dataset(
            X_train, 
            label=y_train,
            group=train_groups,
            free_raw_data=False
        )
        lgb_val = lgb.Dataset(
            X_val, 
            label=y_val,
            group=val_groups,
            reference=lgb_train,
            free_raw_data=False
        )
        
        # Parameters optimized for ranking
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'num_threads': mp.cpu_count() - 2,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 50,
            'max_depth': 8,
            'min_gain_to_split': 0.01,
            'device': 'cpu'  # Use CPU for M3 Max
        }
        
        # Train model
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Make predictions on all data
        print("Making predictions...")
        X_all = data[feature_cols]
        predictions = self.model.predict(X_all, num_iteration=self.model.best_iteration)
        data['y_pred'] = predictions
        
        # Prepare submission
        df_submit = data.reset_index(level=0)
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        print(df_submit)

        # Handle missing IDs
        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competion = df_submit[df_submit["id"].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competion["id"]))
        new_rows = pd.DataFrame(
            {"id": missing_elements, "predict_return": [0] * len(missing_elements)}
        )
        df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
        print(df_submit_competion.shape)
        df_submit_competion.to_csv("./result/submit.csv", index=False)

        # Save true values for checking
        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        print(df_check)
        df_check.to_csv("./result/check.csv", index=False)

        # Calculate weighted Spearman correlation
        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")
        
        # Feature importance
        print("\nTop 20 most important features:")
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        print(importance.head(20))

    def run(self):
        # call the get_all_symbol_kline function to get the K-line data and event data for all currencies
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
        self.train(
            df_24hour_rtn.shift(-windows_1d),
            df_7d_volatility,
            df_7d_momentum,
            df_amount_sum,
        )


if __name__ == "__main__":
    model = OlsModel()
    model.run()