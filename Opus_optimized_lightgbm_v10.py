# optimized_lightgbm_v10.py
# Final optimization focusing on proven winners with feature selection

import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json

pd.options.mode.use_inf_as_na = True


class FinalModelV10:
    """Model V10: Final push with feature selection and stacking"""
    
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
        """Same as V9"""
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
        """Same as V9"""
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

    def create_focused_winner_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume):
        """Ultra-focused on proven winners from V9"""
        features = {}
        
        # === ULTRA PRIORITY 1: DISTANCE FROM LOW_96 (34k importance!) ===
        # Create multiple variations of this winner
        for window in [72, 84, 96, 108, 120]:  # Focus around 96
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            # Primary distance feature
            dist_from_low = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'distance_from_low_{window}'] = dist_from_low
            
            # Squared distance (emphasize extremes)
            features[f'distance_from_low_{window}_sq'] = (dist_from_low ** 2).astype(self.dtype)
            
            # Log distance
            features[f'distance_from_low_{window}_log'] = np.log1p(dist_from_low.clip(lower=0)).astype(self.dtype)
            
            # Distance rank
            features[f'distance_from_low_{window}_rank'] = dist_from_low.rank(axis=1, pct=True).astype(self.dtype)
        
        # Standard distance features for other important windows
        for window in [168, 240, 336, 480, 672]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            features[f'distance_from_low_{window}'] = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'distance_from_high_{window}'] = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # === ULTRA PRIORITY 2: RSI DIVERGENCE_48 (27k importance) ===
        # Create comprehensive RSI features
        for period in [36, 42, 48, 54, 60, 72, 96]:
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}'] = rsi.astype(self.dtype)
            
            # Standard divergence
            price_change = (df_vwap / df_vwap.shift(period) - 1)
            features[f'rsi_divergence_{period}'] = (
                (rsi - rsi.shift(period)) - price_change * 100
            ).astype(self.dtype)
        
        # Special focus on RSI_48 variations
        rsi_48 = features['rsi_48']
        for shift in [12, 24, 36, 48, 72, 96]:
            price_change = (df_vwap / df_vwap.shift(shift) - 1)
            features[f'rsi_divergence_48_{shift}'] = (
                (rsi_48 - rsi_48.shift(shift)) - price_change * 100
            ).astype(self.dtype)
        
        # RSI momentum and extremes
        features['rsi_48_momentum'] = (rsi_48 - rsi_48.shift(24)).astype(self.dtype)
        features['rsi_48_acceleration'] = (
            features['rsi_48_momentum'] - features['rsi_48_momentum'].shift(12)
        ).astype(self.dtype)
        features['rsi_48_extreme'] = ((rsi_48 - 50).abs() / 50).astype(self.dtype)
        
        # === PRIORITY 3: SPREAD CHANGE (12k importance) ===
        hl_spread = (df_high - df_low) / df_vwap
        
        for window in [24, 36, 48, 72, 96, 168, 336]:
            spread_mean = hl_spread.rolling(window, min_periods=window//2).mean()
            features[f'hl_spread_{window}'] = spread_mean.astype(self.dtype)
            
            spread_std = hl_spread.rolling(window, min_periods=window//2).std()
            features[f'hl_spread_std_{window}'] = spread_std.astype(self.dtype)
            
            if window >= 48:
                spread_change = spread_mean / spread_mean.shift(window//2) - 1
                features[f'spread_change_{window}'] = spread_change.astype(self.dtype)
                
                # Spread change acceleration
                features[f'spread_change_accel_{window}'] = (
                    spread_change - spread_change.shift(window//4)
                ).astype(self.dtype)
        
        # === PRIORITY 4: VOLATILITY (10k importance each for 96 and 672) ===
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        
        for window in [48, 72, 96, 168, 336, 672]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
            
            if window >= 96:
                vol_change = vol / vol.shift(window//2) - 1
                features[f'volatility_change_{window}'] = vol_change.astype(self.dtype)
        
        # Volatility ratios
        vol_96 = features['volatility_96']
        vol_672 = features['volatility_672']
        features['volatility_ratio_96_672'] = (vol_96 / (vol_672 + 1e-8)).astype(self.dtype)
        
        # === TREND STRENGTH (showed up as important in V9) ===
        for window in [336, 672, 1344]:
            # Linear regression trend
            trend = df_vwap.rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                raw=False
            )
            features[f'trend_strength_{window}'] = trend.astype(self.dtype)
            
            # Trend consistency
            trend_sign = (df_vwap.diff(window//10) > 0).astype(self.dtype)
            features[f'trend_consistency_{window}'] = trend_sign.rolling(
                window, min_periods=window//2
            ).mean().astype(self.dtype)
        
        # === RETURNS AND CONSISTENCY ===
        for window in [1, 4, 24, 96, 168, 336, 672, 1344]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # Return consistency
        for window in [672, 1344]:
            ret_sign = (returns_15min > 0).astype(self.dtype)
            consistency = ret_sign.rolling(window, min_periods=window//2).mean()
            features[f'return_consistency_{window}'] = consistency.astype(self.dtype)
        
        # === EFFICIENCY RATIO ===
        for window in [96, 168, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            features[f'efficiency_ratio_{window}'] = (net_change / (total_change + 1e-8)).astype(self.dtype)
        
        # === RELATIVE FEATURES ===
        for window in [24, 96]:
            returns = (df_vwap / df_vwap.shift(window) - 1)
            market_return = returns.mean(axis=1)
            features[f'relative_momentum_{window}h'] = (
                returns.sub(market_return, axis=0)
            ).astype(self.dtype)
        
        # === VOLUME (keep minimal) ===
        volume_usd = df_amount * df_vwap
        for window in [96, 336]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
            
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # === INTERACTION FEATURES (between top performers) ===
        # Distance × RSI divergence
        if 'distance_from_low_96' in features and 'rsi_divergence_48' in features:
            features['distance_96_x_rsi_div_48'] = (
                features['distance_from_low_96'] * features['rsi_divergence_48'] / 100
            ).astype(self.dtype)
        
        # Volatility × Spread change
        if 'volatility_96' in features and 'spread_change_48' in features:
            features['vol_96_x_spread_48'] = (
                features['volatility_96'] * features['spread_change_48']
            ).astype(self.dtype)
        
        # Distance × Volatility
        if 'distance_from_low_96' in features and 'volatility_672' in features:
            features['distance_96_x_vol_672'] = (
                features['distance_from_low_96'] * features['volatility_672']
            ).astype(self.dtype)
        
        return features

    def train_final_ensemble(self, X_train_scaled, y_train, feature_names, X_val_scaled=None, y_val=None):
        """Final ensemble with optional validation for early stopping"""
        print("Training final ensemble...")
        
        self.models = []
        self.model_weights = []
        
        base_params = self.best_params.copy()
        
        # Create validation set if provided
        valid_sets = None
        if X_val_scaled is not None and y_val is not None:
            lgb_val = lgb.Dataset(X_val_scaled.values, label=y_val, free_raw_data=False)
            valid_sets = [lgb_val]
        
        # Model 1: Best parameters
        print("Model 1/8: Best params")
        params1 = base_params.copy()
        params1.update({'seed': 42})
        
        lgb_train1 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        if valid_sets:
            model1 = lgb.train(params1, lgb_train1, valid_sets=valid_sets, 
                             num_boost_round=500,
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            model1 = lgb.train(params1, lgb_train1, num_boost_round=300)
        self.models.append(model1)
        self.model_weights.append(0.20)
        
        # Model 2: Different seed
        print("Model 2/8: Different seed")
        params2 = base_params.copy()
        params2.update({'seed': 123})
        
        lgb_train2 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model2 = lgb.train(params2, lgb_train2, num_boost_round=300)
        self.models.append(model2)
        self.model_weights.append(0.15)
        
        # Model 3: Lower LR
        print("Model 3/8: Lower LR")
        params3 = base_params.copy()
        params3['learning_rate'] *= 0.7
        params3.update({'seed': 999})
        
        lgb_train3 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model3 = lgb.train(params3, lgb_train3, num_boost_round=420)
        self.models.append(model3)
        self.model_weights.append(0.15)
        
        # Model 4: Huber
        print("Model 4/8: Huber")
        params4 = base_params.copy()
        params4.update({'objective': 'huber', 'alpha': 0.9, 'metric': 'mae', 'seed': 2024})
        
        lgb_train4 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model4 = lgb.train(params4, lgb_train4, num_boost_round=300)
        self.models.append(model4)
        self.model_weights.append(0.125)
        
        # Model 5: MAE
        print("Model 5/8: MAE")
        params5 = base_params.copy()
        params5.update({'objective': 'regression_l1', 'metric': 'mae', 'seed': 5555})
        
        lgb_train5 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model5 = lgb.train(params5, lgb_train5, num_boost_round=300)
        self.models.append(model5)
        self.model_weights.append(0.125)
        
        # Model 6: More leaves
        print("Model 6/8: More leaves")
        params6 = base_params.copy()
        params6.update({'num_leaves': 200, 'seed': 7777})
        
        lgb_train6 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model6 = lgb.train(params6, lgb_train6, num_boost_round=280)
        self.models.append(model6)
        self.model_weights.append(0.10)
        
        # Model 7: Deeper trees
        print("Model 7/8: Deeper trees")
        params7 = base_params.copy()
        params7.update({'max_depth': 12, 'min_data_in_leaf': 100, 'seed': 8888, 'feature_pre_filter': False})
        
        lgb_train7 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model7 = lgb.train(params7, lgb_train7, num_boost_round=300)
        self.models.append(model7)
        self.model_weights.append(0.05)
        
        # Model 8: Tweedie regression
        print("Model 8/8: Tweedie")
        params8 = base_params.copy()
        params8.update({'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'seed': 9999})
        
        lgb_train8 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model8 = lgb.train(params8, lgb_train8, num_boost_round=280)
        self.models.append(model8)
        self.model_weights.append(0.05)
        
        # Normalize weights
        self.model_weights = np.array(self.model_weights)
        self.model_weights /= self.model_weights.sum()
        
        print(f"Ensemble weights: {self.model_weights}")
        
        return self.models

    def predict_ensemble(self, X_scaled):
        """Ensemble predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred * weight)
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred

    def optimized_post_processing(self, predictions):
        """Optimized post-processing"""
        
        # 1. Clip extremes
        predictions = np.clip(predictions, 
                            np.percentile(predictions, 0.15), 
                            np.percentile(predictions, 99.85))
        
        # 2. Winsorize
        from scipy.stats import mstats
        predictions = mstats.winsorize(predictions, limits=(0.0015, 0.0015))
        
        # 3. Normalize
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        if std_pred > 0:
            predictions = (predictions - mean_pred) / std_pred * 0.022
            
            # 4. Smooth extremes
            extreme_mask = np.abs(predictions) > (0.022 * 3)
            predictions[extreme_mask] *= 0.85
        
        return predictions

    def train_with_feature_selection(self, df_target, df_factor1, df_factor2, df_factor3, 
                                    additional_features, all_symbol_list):
        """Training with feature selection"""
        print("Starting training with feature selection...")
        
        # Prepare all features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use 40% for training
        print("Preparing training data (40% of symbols)...")
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.40)]
        
        # Prepare data
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
        
        print(f"Initial features: {len(feature_names)}")
        
        # Quick feature importance check
        print("Running feature selection...")
        X_temp = train_df[feature_names]
        y_temp = train_df['target'].values
        
        # Quick model for feature importance
        from sklearn.preprocessing import RobustScaler
        scaler_temp = RobustScaler()
        X_temp_scaled = scaler_temp.fit_transform(X_temp)
        
        lgb_quick = lgb.Dataset(X_temp_scaled, label=y_temp)
        params_quick = self.best_params.copy()
        model_quick = lgb.train(params_quick, lgb_quick, num_boost_round=100)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model_quick.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Select top 70 features
        top_features = importance.head(70)['feature'].tolist()
        print(f"Selected top {len(top_features)} features")
        
        # Retrain with selected features
        X_train = train_df[top_features]
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Try QuantileTransformer for different scaling
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        # Train final ensemble
        self.train_final_ensemble(X_train_scaled, y_train, top_features)
        
        self.feature_names = top_features
        self.scaler = scaler
        
        # Process all data
        print("Making predictions...")
        chunk_size = 20
        results_list = []
        
        for i in range(0, len(all_symbol_list), chunk_size):
            chunk_symbols = all_symbol_list[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(all_symbol_list) + chunk_size - 1)//chunk_size}...")
            
            # Prepare chunk
            chunk_features = []
            for feat_name in top_features:
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
            
            # Scale and predict
            X_chunk = chunk_df[top_features]
            X_chunk_scaled = pd.DataFrame(
                self.scaler.transform(X_chunk),
                index=X_chunk.index,
                columns=X_chunk.columns
            ).astype(self.dtype)
            
            predictions = self.predict_ensemble(X_chunk_scaled.values)
            predictions = self.optimized_post_processing(predictions)
            
            chunk_results = chunk_df[['target']].copy()
            chunk_results['y_pred'] = predictions
            results_list.append(chunk_results)
            
            del chunk_features, chunk_df, X_chunk, X_chunk_scaled
            gc.collect()
        
        results_df = pd.concat(results_list)
        return results_df

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        """Main training"""
        print("Getting data...")
        
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
        df_volume = df_amount / (df_vwap + 1e-8)
        
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        print("Creating winner features...")
        additional_features = self.create_focused_winner_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume
        )
        
        del df_high, df_low, df_open, df_close, df_volume
        gc.collect()
        
        # Train with feature selection
        results_df = self.train_with_feature_selection(
            df_target, df_factor1, df_factor2, df_factor3, 
            additional_features, all_symbol_list
        )
        
        # Calculate performance
        rho_overall = self.weighted_spearmanr(results_df['target'], results_df['y_pred'])
        print(f"\nWeighted Spearman: {rho_overall:.4f}")
        
        # Prepare submission
        self._prepare_submission(results_df)
        
        # Feature importance
        if hasattr(self, 'models') and len(self.models) > 0:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 20 features (from selected 70):")
            print(importance.head(20))
            
            # Save info
            model_info = {
                'score': float(rho_overall),
                'timestamp': datetime.datetime.now().isoformat(),
                'n_models': len(self.models),
                'ensemble_weights': self.model_weights.tolist(),
                'top_features': importance.head(20)['feature'].tolist(),
                'total_features': len(self.feature_names)
            }
            
            with open('./result/model_v10_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)

    def _prepare_submission(self, results_df):
        """Prepare submission"""
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
        df_submit_competition.to_csv("./result/submit_v10.csv", index=False)

    def run(self):
        """Main execution"""
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
        
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).astype(self.dtype)
        df_15min_rtn = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        df_7d_volatility = df_15min_rtn.rolling(windows_7d, min_periods=windows_7d//2).std(ddof=1).astype(self.dtype)
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).astype(self.dtype)
        df_amount_sum = df_amount.rolling(windows_7d, min_periods=windows_7d//2).sum().astype(self.dtype)
        
        del open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
        gc.collect()
        
        self.train(
            df_24hour_rtn.shift(-windows_1d),
            df_7d_volatility,
            df_7d_momentum,
            df_amount_sum,
        )


if __name__ == "__main__":
    import sys
    if sys.platform == 'darwin':
        os.environ['OMP_NUM_THREADS'] = '4'
    
    print("="*70)
    print("FINAL MODEL V10 - MAXIMIZING PROVEN WINNERS")
    print("="*70)
    print("\nKey optimizations:")
    print("1. MASSIVE focus on distance_from_low_96 (multiple variations)")
    print("2. Comprehensive RSI_48 divergence features")  
    print("3. Feature selection: Top 70 features only")
    print("4. QuantileTransformer for better scaling")
    print("5. 8-model ensemble including Tweedie")
    print("6. Interaction features between top winners")
    print("7. Trend strength and consistency")
    print("8. Optimized post-processing")
    print("\n" + "="*70 + "\n")
    
    model = FinalModelV10()
    model.run()