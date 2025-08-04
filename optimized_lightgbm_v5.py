import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json

# Use float32 to save memory
pd.options.mode.use_inf_as_na = True


class MemoryOptimizedModelV5:
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 30
        self.dtype = np.float32
        self.models = []  # For ensemble

    def get_all_symbol_list(self):
        parquet_name_list = os.listdir(self.train_data_path)
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
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
        
        df_open_price = pd.concat(
            [i.get()["open_price"] for i in df_list], axis=1
        ).sort_index(ascending=True)
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        
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

    def create_enhanced_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close):
        """Enhanced features based on v4 results + new high-impact features"""
        features = {}
        
        # 1. Core returns - proven to be important (keep existing)
        for window in [1, 4, 24, 96, 336]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # NEW: Add longer-term returns (based on return_336 importance)
        for window in [168, 672]:  # 1.75d, 7d
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # 2. Volatility features (keep existing)
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
        
        # NEW: Volatility ratio and regime
        vol_96 = returns_15min.rolling(96, min_periods=48).std()
        vol_336 = returns_15min.rolling(336, min_periods=168).std()
        features['volatility_ratio_96_336'] = (vol_96 / (vol_336 + 1e-8)).astype(self.dtype)
        
        # NEW: Volatility percentile
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            vol_percentile = vol.rolling(window*4, min_periods=window*2).rank(pct=True)
            features[f'volatility_percentile_{window}'] = vol_percentile.astype(self.dtype)
        
        # 3. Volume patterns (keep existing)
        for window in [96, 336]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_mean_{window}'] = vol_mean.astype(self.dtype)
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
        
        # NEW: Volume in USD and concentration
        volume_usd = df_amount * df_vwap
        for window in [96, 336]:
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}'] = vol_usd_mean.astype(self.dtype)
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # 4. Price position (keep existing)
        for window in [336]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # NEW: Price efficiency ratio (trending vs choppy)
        for window in [96, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            features[f'efficiency_ratio_{window}'] = (
                net_change / (total_change + 1e-8)
            ).astype(self.dtype)
        
        # 5. Momentum (keep existing + enhancements)
        features['momentum_96'] = (df_vwap / df_vwap.shift(96) - 1).astype(self.dtype)
        
        # NEW: Momentum consistency
        for window in [96, 336]:
            positive_periods = (returns_15min > 0).rolling(window, min_periods=window//2).sum()
            features[f'momentum_consistency_{window}'] = (
                positive_periods / window
            ).astype(self.dtype)
        
        # 6. RSI (keep existing)
        delta = df_vwap.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=96, min_periods=48).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=96, min_periods=48).mean()
        rs = gain / (loss + 1e-8)
        features['rsi_96'] = (100 - (100 / (1 + rs))).astype(self.dtype)
        
        # NEW: RSI divergence
        rsi = features['rsi_96']
        price_change = (df_vwap / df_vwap.shift(96) - 1)
        features['rsi_divergence'] = (
            (rsi - rsi.shift(96)) - price_change * 100
        ).astype(self.dtype)
        
        # 7. NEW: Market microstructure
        # Overnight gap
        overnight_gap = (df_open - df_close.shift(1)) / (df_close.shift(1) + 1e-8)
        features['overnight_gap_mean_96'] = overnight_gap.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        # High-low spread
        hl_spread = (df_high - df_low) / df_vwap
        features['hl_spread_96'] = hl_spread.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        # 8. NEW: Cross-sectional features
        # Relative momentum
        returns_24h = (df_vwap / df_vwap.shift(96) - 1)
        market_return = returns_24h.mean(axis=1)
        features['relative_momentum_24h'] = (
            returns_24h.sub(market_return, axis=0)
        ).astype(self.dtype)
        
        return features

    def post_process_predictions(self, predictions, method='clip_and_normalize'):
        """Post-process predictions to optimize for ranking"""
        
        if method == 'clip_and_normalize':
            # Clip extreme values
            predictions = np.clip(predictions, 
                                np.percentile(predictions, 0.5), 
                                np.percentile(predictions, 99.5))
            
            # Normalize to reasonable return range
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            if std_pred > 0:
                predictions = (predictions - mean_pred) / std_pred * 0.02  # 2% typical vol
            
        elif method == 'rank_transform':
            # Convert to ranks then back to returns
            ranks = stats.rankdata(predictions) / len(predictions)
            predictions = stats.norm.ppf(ranks, loc=0, scale=0.02)
            
        return predictions

    def train_ensemble(self, X_train_scaled, y_train, feature_names, base_params):
        """Train ensemble of models with different configurations"""
        print("Training ensemble models...")
        
        self.models = []
        self.model_weights = []
        
        # Model 1: Base parameters (best from Optuna)
        print("Training model 1/3: Base parameters")
        lgb_train = lgb.Dataset(X_train_scaled.values, label=y_train)
        
        model1 = lgb.train(
            base_params,
            lgb_train,
            num_boost_round=300,
        )
        self.models.append(model1)
        self.model_weights.append(0.4)  # Higher weight for best model
        
        # Model 2: Huber loss for robustness
        print("Training model 2/3: Huber loss")
        params2 = base_params.copy()
        params2['objective'] = 'huber'
        params2['alpha'] = 0.9
        params2['metric'] = 'mae'
        
        model2 = lgb.train(
            params2,
            lgb_train,
            num_boost_round=300,
        )
        self.models.append(model2)
        self.model_weights.append(0.3)
        
        # Model 3: Lower learning rate, more trees
        print("Training model 3/3: Lower learning rate")
        params3 = base_params.copy()
        params3['learning_rate'] = params3['learning_rate'] * 0.7
        
        model3 = lgb.train(
            params3,
            lgb_train,
            num_boost_round=400,
        )
        self.models.append(model3)
        self.model_weights.append(0.3)
        
        # Normalize weights
        self.model_weights = np.array(self.model_weights)
        self.model_weights /= self.model_weights.sum()
        
        print(f"Ensemble weights: {self.model_weights}")
        
        return self.models

    def predict_ensemble(self, X_scaled):
        """Make predictions using ensemble"""
        predictions = []
        
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Memory-efficient training with ensemble and post-processing"""
        print("Starting enhanced memory-efficient training...")
        
        # Prepare features dict
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Train on subset
        print("Preparing training data...")
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
        
        train_target = df_target[train_symbols].stack()
        train_target.name = 'target'
        
        train_df = pd.concat(train_features + [train_target], axis=1)
        train_df = train_df.dropna()
        
        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = train_df[feature_names]
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Load best parameters or use defaults
        try:
            with open('./result/best_params.json', 'r') as f:
                base_params = json.load(f)
            base_params.update({
                'boosting_type': 'gbdt',
                'num_threads': mp.cpu_count() - 2,
                'verbose': -1,
                'force_row_wise': True,
            })
            if base_params['objective'] == 'huber' and 'metric' not in base_params:
                base_params['metric'] = 'mae'
            elif 'metric' not in base_params:
                base_params['metric'] = 'rmse'
            print("Using Optuna-optimized parameters")
        except:
            base_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 163,
                'learning_rate': 0.07182861492338714,
                'feature_fraction': 0.8184386555861791,
                'bagging_fraction': 0.6985394411684631,
                'bagging_freq': 2,
                'lambda_l1': 0.7113790560603763,
                'lambda_l2': 0.09201149241276538,
                'min_data_in_leaf': 139,
                'max_depth': 10,
                'num_threads': mp.cpu_count() - 2,
                'verbose': -1,
                'force_row_wise': True,
            }
            print("Using default optimized parameters")
        
        # Train ensemble
        self.train_ensemble(X_train_scaled, y_train, feature_names, base_params)
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        # Process all data in chunks for prediction
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
            
            chunk_target = df_target[chunk_symbols].stack()
            chunk_target.name = 'target'
            
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
            
            # Make ensemble predictions
            predictions = self.predict_ensemble(X_chunk_scaled.values)
            
            # Post-process predictions
            predictions = self.post_process_predictions(predictions)
            
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
        df_open = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Free memory
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        # Create enhanced features
        print("Creating enhanced features...")
        additional_features = self.create_enhanced_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close
        )
        
        # Free more memory
        del df_high, df_low, df_open, df_close
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
        
        # Feature importance (from first model)
        if hasattr(self, 'models') and len(self.models) > 0:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 15 most important features:")
            print(importance.head(15))

    def _prepare_submission(self, results_df):
        """Prepare submission files"""
        df_submit = results_df.reset_index()
        
        if 'level_0' in df_submit.columns and 'level_1' in df_submit.columns:
            df_submit.rename(columns={'level_0': 'datetime', 'level_1': 'symbol'}, inplace=True)
        elif len(df_submit.index.names) == 2:
            df_submit.index.names = ['datetime', 'symbol']
            df_submit = df_submit.reset_index()
        
        df_submit = df_submit[['datetime', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        print(df_submit.head())
        print(f"Predictions shape: {df_submit.shape}")

        # Handle missing IDs
        df_submission_id = pd.read_csv("./result/submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competition = df_submit[df_submit["id"].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competition["id"]))
        
        if missing_elements:
            print(f"Missing {len(missing_elements)} IDs, filling with 0")
            new_rows = pd.DataFrame({
                "id": missing_elements, 
                "predict_return": [0] * len(missing_elements)
            })
            df_submit_competition = pd.concat([df_submit_competition, new_rows], ignore_index=True)
        
        print(f"Final submission shape: {df_submit_competition.shape}")
        df_submit_competition.to_csv("./result/submit_v5.csv", index=False)

        # Save true values
        df_check = results_df.reset_index()
        
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
        df_check.to_csv("./result/check_v5.csv", index=False)

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
    
    print("="*60)
    print("OPTIMIZED LIGHTGBM V5 - TARGET: 0.35+ WEIGHTED SPEARMAN")
    print("="*60)
    print("\nEnhancements in this version:")
    print("1. Extended feature set (efficiency ratio, long-term returns)")
    print("2. Ensemble of 3 models (base, huber, low LR)")
    print("3. Post-processing for ranking optimization")
    print("4. Enhanced volatility and volume features")
    print("\n" + "="*60 + "\n")
    
    model = MemoryOptimizedModelV5()
    model.run()