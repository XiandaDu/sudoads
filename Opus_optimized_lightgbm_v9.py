# optimized_lightgbm_v9.py
# Strategic version combining V7's success with targeted improvements

import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json

pd.options.mode.use_inf_as_na = True


class StrategicModelV9:
    """Model V9: Strategic combination of V7's winners with improvements"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 30
        self.dtype = np.float32
        self.models = []
        
        # Best parameters
        self.best_params = {
            "objective": "regression",
            "num_leaves": 163,
            "learning_rate": 0.07182861492338714,
            "feature_fraction": 0.8184386555861791,
            "bagging_fraction": 0.6985394411684631,
            "bagging_freq": 2,
            "lambda_l1": 0.7113790560603763,
            "lambda_l2": 0.09201149241276538,
            "min_data_in_leaf": 139,
            "max_depth": 10,
            'boosting_type': 'gbdt',
            'num_threads': mp.cpu_count() - 2,
            'verbose': -1,
            'force_row_wise': True,
            'metric': 'rmse'
        }
        
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
        """Same as V7"""
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
        """Same as V7"""
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

    def create_v7_plus_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume):
        """V7 features plus strategic enhancements"""
        features = {}
        
        # === REPLICATE V7's TOP FEATURES EXACTLY ===
        
        # RSI DIVERGENCE (V7's #1 feature with 33k importance)
        for period in [48, 96, 168]:  # Focus on 48 which was best
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}'] = rsi.astype(self.dtype)
            
            # RSI divergence (exactly as V7)
            price_change = (df_vwap / df_vwap.shift(period) - 1)
            features[f'rsi_divergence_{period}'] = (
                (rsi - rsi.shift(period)) - price_change * 100
            ).astype(self.dtype)
        
        # NEW: Additional RSI divergence variations for period 48
        rsi_48 = features['rsi_48']
        for shift in [24, 36, 72]:
            price_change = (df_vwap / df_vwap.shift(shift) - 1)
            features[f'rsi_divergence_48_{shift}'] = (
                (rsi_48 - rsi_48.shift(shift)) - price_change * 100
            ).astype(self.dtype)
        
        # DISTANCE FROM LOW (V7's #2, #4, #5 features)
        for window in [168, 336, 672]:  # These were top in V7
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            features[f'distance_from_high_{window}'] = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            features[f'distance_from_low_{window}'] = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # NEW: More granular distance windows
        for window in [96, 120, 240, 480]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            features[f'distance_from_low_{window}'] = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            if window in [96, 240]:
                features[f'distance_from_high_{window}'] = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # HL SPREAD STD (V7's #3, #6, #10 features)
        hl_spread = (df_high - df_low) / df_vwap
        for window in [48, 96, 168, 336]:
            features[f'hl_spread_{window}'] = hl_spread.rolling(window, min_periods=window//2).mean().astype(self.dtype)
            features[f'hl_spread_std_{window}'] = hl_spread.rolling(window, min_periods=window//2).std().astype(self.dtype)
            
            if window >= 48:
                spread_mean = features[f'hl_spread_{window}']
                spread_change = spread_mean / spread_mean.shift(window//2) - 1
                features[f'spread_change_{window}'] = spread_change.astype(self.dtype)
        
        # VOLATILITY (still important from V7)
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        
        for window in [48, 96, 168, 336, 672]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
            
            if window >= 96:
                vol_change = vol / vol.shift(window//2) - 1
                features[f'volatility_change_{window}'] = vol_change.astype(self.dtype)
        
        # Volatility ratio and percentiles (from V7)
        vol_96 = returns_15min.rolling(96, min_periods=48).std()
        vol_336 = returns_15min.rolling(336, min_periods=168).std()
        features['volatility_ratio_96_336'] = (vol_96 / (vol_336 + 1e-8)).astype(self.dtype)
        
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            for lookback_mult in [4, 8]:
                vol_percentile = vol.rolling(window*lookback_mult, min_periods=window*lookback_mult//2).rank(pct=True)
                features[f'volatility_percentile_{window}_{window*lookback_mult}'] = vol_percentile.astype(self.dtype)
        
        # EFFICIENCY RATIO (V7's #4 feature)
        for window in [96, 168, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            features[f'efficiency_ratio_{window}'] = (net_change / (total_change + 1e-8)).astype(self.dtype)
            
            if window == 96:
                eff_ratio = features[f'efficiency_ratio_{window}']
                features[f'efficiency_momentum_{window}'] = (eff_ratio - eff_ratio.shift(window//2)).astype(self.dtype)
        
        # RETURNS (especially long ones were important in V7)
        for window in [1, 4, 24, 96, 168, 336, 672, 1344]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # RETURN CONSISTENCY (V7's #8 feature)
        for window in [672, 1344]:
            ret_sign = (returns_15min > 0).astype(self.dtype)
            consistency = ret_sign.rolling(window, min_periods=window//2).mean()
            features[f'return_consistency_{window}'] = consistency.astype(self.dtype)
        
        # RELATIVE MOMENTUM
        for window in [24, 96, 336]:
            returns = (df_vwap / df_vwap.shift(window) - 1)
            market_return = returns.mean(axis=1)
            features[f'relative_momentum_{window}h'] = (
                returns.sub(market_return, axis=0)
            ).astype(self.dtype)
        
        # VOLUME patterns
        volume_usd = df_amount * df_vwap
        for window in [96, 336, 672]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_mean_{window}'] = vol_mean.astype(self.dtype)
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
            
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}'] = vol_usd_mean.astype(self.dtype)
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # MICROSTRUCTURE
        overnight_gap = (df_open - df_close.shift(1)) / (df_close.shift(1) + 1e-8)
        features['overnight_gap_mean_96'] = overnight_gap.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        # Yang-Zhang volatility
        for window in [96, 336]:
            overnight_var = ((np.log(df_open / df_close.shift(1))) ** 2).rolling(
                window, min_periods=window//2
            ).mean()
            oc_var = ((np.log(df_close / df_open)) ** 2).rolling(
                window, min_periods=window//2
            ).mean()
            rs_var = ((np.log(df_high / df_close) * np.log(df_high / df_open) + 
                      np.log(df_low / df_close) * np.log(df_low / df_open))).rolling(
                window, min_periods=window//2
            ).mean()
            
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_var = np.sqrt(overnight_var + k * oc_var + (1 - k) * rs_var)
            features[f'yang_zhang_vol_{window}'] = yz_var.astype(self.dtype)
        
        # Parkinson volatility
        for window in [96, 336]:
            parkinson = np.sqrt(
                ((np.log(df_high / df_low) ** 2) / (4 * np.log(2))).rolling(
                    window, min_periods=window//2
                ).mean()
            )
            features[f'parkinson_vol_{window}'] = parkinson.astype(self.dtype)
        
        # VWAP deviation
        vwap_deviation = (df_vwap - (df_high + df_low + df_close) / 3) / df_vwap
        features['vwap_deviation'] = vwap_deviation.astype(self.dtype)
        
        # Momentum
        features['momentum_96'] = (df_vwap / df_vwap.shift(96) - 1).astype(self.dtype)
        features['momentum_336'] = (df_vwap / df_vwap.shift(336) - 1).astype(self.dtype)
        
        # Momentum consistency
        for window in [96, 336]:
            positive_periods = (returns_15min > 0).rolling(window, min_periods=window//2).sum()
            features[f'momentum_consistency_{window}'] = (
                positive_periods / window
            ).astype(self.dtype)
        
        # === NEW STRATEGIC FEATURES ===
        
        # Enhanced RSI features (since RSI divergence was #1)
        rsi_96 = features['rsi_96']
        features['rsi_96_extreme'] = ((rsi_96 - 50).abs() / 50).astype(self.dtype)
        features['rsi_96_oversold'] = (rsi_96 < 30).astype(self.dtype)
        features['rsi_96_overbought'] = (rsi_96 > 70).astype(self.dtype)
        
        # Price acceleration
        for window in [96, 336]:
            returns = df_vwap / df_vwap.shift(window) - 1
            returns_change = returns - returns.shift(window)
            features[f'price_acceleration_{window}'] = returns_change.astype(self.dtype)
        
        # Trend strength (linear regression slope)
        for window in [336, 672]:
            trend = df_vwap.rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                raw=False
            )
            features[f'trend_strength_{window}'] = trend.astype(self.dtype)
        
        return features

    def train_strategic_ensemble(self, X_train_scaled, y_train, feature_names):
        """Strategic ensemble with optimized diversity"""
        print("Training strategic ensemble...")
        
        self.models = []
        self.model_weights = []
        
        base_params = self.best_params.copy()
        
        # Model 1: Best parameters
        print("Model 1/7: Best params (seed=42)")
        params1 = base_params.copy()
        params1.update({'seed': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42})
        
        lgb_train1 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model1 = lgb.train(params1, lgb_train1, num_boost_round=300)
        self.models.append(model1)
        self.model_weights.append(0.20)
        
        # Model 2: Different seed
        print("Model 2/7: Best params (seed=123)")
        params2 = base_params.copy()
        params2.update({'seed': 123, 'bagging_seed': 456, 'feature_fraction_seed': 789})
        
        lgb_train2 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model2 = lgb.train(params2, lgb_train2, num_boost_round=300)
        self.models.append(model2)
        self.model_weights.append(0.20)
        
        # Model 3: Lower learning rate
        print("Model 3/7: Lower LR, more trees")
        params3 = base_params.copy()
        params3['learning_rate'] = params3['learning_rate'] * 0.75
        params3.update({'seed': 999})
        
        lgb_train3 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model3 = lgb.train(params3, lgb_train3, num_boost_round=400)
        self.models.append(model3)
        self.model_weights.append(0.15)
        
        # Model 4: Huber loss
        print("Model 4/7: Huber loss")
        params4 = base_params.copy()
        params4.update({
            'objective': 'huber',
            'alpha': 0.9,
            'metric': 'mae',
            'seed': 2024
        })
        
        lgb_train4 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model4 = lgb.train(params4, lgb_train4, num_boost_round=300)
        self.models.append(model4)
        self.model_weights.append(0.15)
        
        # Model 5: MAE objective
        print("Model 5/7: MAE objective")
        params5 = base_params.copy()
        params5.update({
            'objective': 'regression_l1',
            'metric': 'mae',
            'seed': 5555
        })
        
        lgb_train5 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model5 = lgb.train(params5, lgb_train5, num_boost_round=300)
        self.models.append(model5)
        self.model_weights.append(0.10)
        
        # Model 6: More leaves
        print("Model 6/7: More leaves")
        params6 = base_params.copy()
        params6['num_leaves'] = 200
        params6['min_data_in_leaf'] = 100
        params6.update({'seed': 7777, 'feature_pre_filter': False})
        
        lgb_train6 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model6 = lgb.train(params6, lgb_train6, num_boost_round=280)
        self.models.append(model6)
        self.model_weights.append(0.10)
        
        # Model 7: Higher learning rate, fewer trees
        print("Model 7/7: Higher LR, fewer trees")
        params7 = base_params.copy()
        params7['learning_rate'] = params7['learning_rate'] * 1.2
        params7.update({'seed': 8888})
        
        lgb_train7 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model7 = lgb.train(params7, lgb_train7, num_boost_round=200)
        self.models.append(model7)
        self.model_weights.append(0.10)
        
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
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred

    def adaptive_post_processing(self, predictions, percentile_low=0.25, percentile_high=99.75):
        """Adaptive post-processing based on distribution"""
        
        # 1. Adaptive clipping
        predictions = np.clip(predictions, 
                            np.percentile(predictions, percentile_low), 
                            np.percentile(predictions, percentile_high))
        
        # 2. Winsorization
        from scipy.stats import mstats
        predictions = mstats.winsorize(predictions, limits=(0.002, 0.002))
        
        # 3. Normalize to realistic distribution
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        if std_pred > 0:
            # Slightly higher volatility based on V7 success
            predictions = (predictions - mean_pred) / std_pred * 0.0215
            
            # 4. Adaptive extreme handling
            extreme_threshold = 3.2
            extreme_mask = np.abs(predictions) > (0.0215 * extreme_threshold)
            predictions[extreme_mask] *= 0.88
        
        return predictions

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Memory-efficient training with 40% data"""
        print("Starting strategic training for V9...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use 40% for training as requested
        print("Preparing training data (40% of symbols)...")
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.40)]
        
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
        
        print(f"Total features: {len(feature_names)}")
        
        # Scale features - try StandardScaler for variety
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = train_df[feature_names]
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Train ensemble
        self.train_strategic_ensemble(X_train_scaled, y_train, feature_names)
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        # Process all data in chunks
        print("Making predictions on all data...")
        chunk_size = 18  # Balance between memory and speed
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
            
            # Adaptive post-processing
            predictions = self.adaptive_post_processing(predictions)
            
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
        """Main training function"""
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
        
        # Create DataFrames
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_open = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Calculate volume
        df_volume = df_amount / (df_vwap + 1e-8)
        
        # Free memory
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        # Create features
        print("Creating V7+ strategic features...")
        additional_features = self.create_v7_plus_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume
        )
        
        # Free memory
        del df_high, df_low, df_open, df_close, df_volume
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
        if hasattr(self, 'models') and len(self.models) > 0:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 25 most important features:")
            print(importance.head(25))
            
            # Save model info
            model_info = {
                'score': float(rho_overall),
                'timestamp': datetime.datetime.now().isoformat(),
                'n_models': len(self.models),
                'ensemble_weights': self.model_weights.tolist(),
                'top_features': importance.head(25)['feature'].tolist(),
                'total_features': len(self.feature_names)
            }
            
            with open('./result/model_v9_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"\nTotal features used: {len(self.feature_names)}")

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
        df_submit_competition.to_csv("./result/submit_v9.csv", index=False)

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
        
        # Create DataFrames
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Calculate factors
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
        os.environ['OMP_NUM_THREADS'] = '4'
    
    print("="*70)
    print("STRATEGIC MODEL V9 - COMBINING V7 SUCCESS WITH ENHANCEMENTS")
    print("="*70)
    print("\nKey strategies:")
    print("1. Replicate V7's exact top features")
    print("2. Add strategic RSI divergence variations (48 period focus)")
    print("3. Enhanced distance features with more windows")
    print("4. 40% training data (vs 35% in V7)")
    print("5. 7-model ensemble with optimized diversity")
    print("6. Adaptive post-processing")
    print("7. ~120 features (between V7's 114 and V8's 150)")
    print("8. Price acceleration and trend strength features")
    print("\n" + "="*70 + "\n")
    
    model = StrategicModelV9()
    model.run()