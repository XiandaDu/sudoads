import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Use float32 to save memory
pd.options.mode.use_inf_as_na = True


class MemoryOptimizedModel:
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        # Reduce batch size for memory efficiency
        self.batch_size = 30
        # Use float32 for all calculations
        self.dtype = np.float32

    def get_all_symbol_list(self):
        parquet_name_list = os.listdir(self.train_data_path)
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
            # Use float32 to save memory
            df = df.astype(self.dtype)
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
        df_list = []
        for i in range(len(all_symbol_list)):
            df_list.append(
                pool.apply_async(
                    self.get_single_symbol_kline_data, (all_symbol_list[i],)
                )
            )
        pool.close()
        pool.join()
        
        # Process data more efficiently
        df_open_price = pd.concat(
            [i.get()["open_price"] for i in df_list], axis=1
        ).sort_index(ascending=True)
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        
        # Extract arrays with float32
        open_price_arr = df_open_price.values.astype(self.dtype)
        high_price_arr = pd.concat([i.get()["high_price"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        low_price_arr = pd.concat([i.get()["low_price"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        close_price_arr = pd.concat([i.get()["close_price"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        vwap_arr = pd.concat([i.get()["vwap"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        amount_arr = pd.concat([i.get()["amount"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        
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

    def create_critical_features(self, df_vwap, df_high, df_low, df_amount):
        """Create only the most critical features that showed high importance"""
        features = {}
        
        # 1. Core returns - proven to be important
        for window in [1, 4, 24, 96, 336]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            # Cross-sectional rank is crucial for ranking task
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # 2. Volatility - your results showed this is important
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            # Volatility rank
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
        
        # 3. Volume patterns - showed high importance in your results
        for window in [96, 336]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_mean_{window}'] = vol_mean.astype(self.dtype)
            # Volume ratio was the most important feature
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
        
        # 4. Price position - also showed importance
        for window in [336]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # 5. Simple momentum
        features['momentum_96'] = (df_vwap / df_vwap.shift(96) - 1).astype(self.dtype)
        
        # 6. RSI - showed importance
        delta = df_vwap.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=96, min_periods=48).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=96, min_periods=48).mean()
        rs = gain / (loss + 1e-8)
        features['rsi_96'] = (100 - (100 / (1 + rs))).astype(self.dtype)
        
        return features

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Memory-efficient training with chunked processing"""
        print("Starting memory-efficient training...")
        
        # Prepare features dict
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # First, train on a subset to get a model
        print("Training model on subset...")
        
        # Use first 30% of symbols for training
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.3)]
        
        # Prepare training data
        train_features = []
        feature_names = []
        
        for feat_name, feat_data in all_features.items():
            if isinstance(feat_data, pd.DataFrame):
                train_data = feat_data[train_symbols].stack()
                train_data.name = feat_name
                train_features.append(train_data)
                feature_names.append(feat_name)
        
        # Stack target
        train_target = df_target[train_symbols].stack()
        train_target.name = 'target'
        
        # Combine
        train_df = pd.concat(train_features + [train_target], axis=1)
        train_df = train_df.dropna()
        
        # Normalize features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = train_df[feature_names]
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Train model
        lgb_train = lgb.Dataset(X_train_scaled.values, label=y_train)
        
        params = {
            'objective': 'huber',  # 对异常值更稳健
            'alpha': 0.9,
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 127,  # 增加复杂度
            'learning_rate': 0.03,  # 降低学习率
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.05,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 80,
            'max_depth': 8,
            'num_threads': mp.cpu_count() - 2,
            'verbose': -1,
            'force_row_wise': True,
        }
        
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,
        )
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        # Now process all data in chunks for prediction
        print("Making predictions on all data...")
        
        chunk_size = 25
        results_list = []
        
        for i in range(0, len(all_symbol_list), chunk_size):
            chunk_symbols = all_symbol_list[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(all_symbol_list) + chunk_size - 1)//chunk_size}...")
            
            # Prepare chunk data
            chunk_features = []
            
            for feat_name in feature_names:
                if feat_name in all_features:
                    chunk_data = all_features[feat_name][chunk_symbols].stack()
                    chunk_data.name = feat_name
                    chunk_features.append(chunk_data)
            
            # Stack target
            chunk_target = df_target[chunk_symbols].stack()
            chunk_target.name = 'target'
            
            # Combine
            chunk_df = pd.concat(chunk_features + [chunk_target], axis=1)
            chunk_df = chunk_df.dropna()
            
            if len(chunk_df) == 0:
                continue
            
            # Scale features
            X_chunk = chunk_df[feature_names]
            X_chunk_scaled = pd.DataFrame(
                self.scaler.transform(X_chunk),
                index=X_chunk.index,
                columns=X_chunk.columns
            ).astype(self.dtype)
            
            # Make predictions
            predictions = self.model.predict(
                X_chunk_scaled.values, 
                num_iteration=self.model.best_iteration
            )
            
            # Store results
            chunk_results = chunk_df[['target']].copy()
            chunk_results['y_pred'] = predictions
            results_list.append(chunk_results)
            
            # Free memory
            del chunk_features, chunk_df, X_chunk, X_chunk_scaled
            gc.collect()
        
        # Combine all results
        results_df = pd.concat(results_list)
        
        return results_df

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        """Main training function with memory optimization"""
        print("Getting additional data for features...")
        
        # Get data
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
        
        # Create DataFrames with float32
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Free memory
        del open_price_arr, close_price_arr, vwap_arr, amount_arr, high_price_arr, low_price_arr
        gc.collect()
        
        # Create critical features only
        print("Creating critical features...")
        additional_features = self.create_critical_features(df_vwap, df_high, df_low, df_amount)
        
        # Free more memory
        del df_high, df_low
        gc.collect()
        
        # Train model
        results_df = self.train_efficient(
            df_target, df_factor1, df_factor2, df_factor3, 
            additional_features, all_symbol_list
        )
        
        # Calculate performance
        rho_overall = self.weighted_spearmanr(results_df['target'], results_df['y_pred'])
        print(f"\nWeighted Spearman correlation coefficient: {rho_overall:.4f}")
        
        # Prepare submission
        self._prepare_submission(results_df)
        
        # Feature importance
        if hasattr(self, 'model'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 10 most important features:")
            print(importance.head(10))

    def _prepare_submission(self, results_df):
        """Prepare submission files"""
        # Reset index to get datetime and symbol columns
        df_submit = results_df.reset_index()
        
        # Handle MultiIndex case
        if 'level_0' in df_submit.columns and 'level_1' in df_submit.columns:
            df_submit.rename(columns={'level_0': 'datetime', 'level_1': 'symbol'}, inplace=True)
        elif len(df_submit.index.names) == 2:
            df_submit.index.names = ['datetime', 'symbol']
            df_submit = df_submit.reset_index()
        
        # Select required columns
        df_submit = df_submit[['datetime', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        
        # Filter by start date
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        
        # Create ID column
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        print(df_submit.head())
        print(f"Predictions shape: {df_submit.shape}")

        # Handle missing IDs
        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competition = df_submit[df_submit["id"].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competition["id"]))
        
        # For missing elements, use 0
        if missing_elements:
            print(f"Missing {len(missing_elements)} IDs, filling with 0")
            new_rows = pd.DataFrame({
                "id": missing_elements, 
                "predict_return": [0] * len(missing_elements)
            })
            df_submit_competition = pd.concat([df_submit_competition, new_rows], ignore_index=True)
        
        print(f"Final submission shape: {df_submit_competition.shape}")
        df_submit_competition.to_csv("./result/submit_optimized.csv", index=False)

        # Save true values
        df_check = results_df.reset_index()
        
        # Handle MultiIndex case
        if 'level_0' in df_check.columns and 'level_1' in df_check.columns:
            df_check.rename(columns={'level_0': 'datetime', 'level_1': 'symbol'}, inplace=True)
        elif len(df_check.index.names) == 2:
            df_check.index.names = ['datetime', 'symbol']
            df_check = df_check.reset_index()
        
        df_check = df_check[['datetime', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        df_check.to_csv("./result/check_optimized.csv", index=False)

    def run(self):
        """Main execution function"""
        # Get initial data
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
        
        # Create minimal DataFrames needed
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Calculate basic factors
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).astype(self.dtype)
        df_15min_rtn = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        df_7d_volatility = df_15min_rtn.rolling(windows_7d, min_periods=windows_7d//2).std(ddof=1).astype(self.dtype)
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).astype(self.dtype)
        df_amount_sum = df_amount.rolling(windows_7d, min_periods=windows_7d//2).sum().astype(self.dtype)
        
        # Free memory
        del open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
        gc.collect()
        
        # Train
        self.train(
            df_24hour_rtn.shift(-windows_1d),
            df_7d_volatility,
            df_7d_momentum,
            df_amount_sum,
        )


if __name__ == "__main__":
    # Set memory-friendly options
    import sys
    if sys.platform == 'darwin':  # macOS
        os.environ['OMP_NUM_THREADS'] = '4'  # Limit threads to save memory
    
    model = MemoryOptimizedModel()
    model.run()