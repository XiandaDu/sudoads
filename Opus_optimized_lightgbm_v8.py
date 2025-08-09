# optimized_lightgbm_v8.py
# Ultimate version targeting 0.57+ based on V7's success (0.5549)

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

pd.options.mode.use_inf_as_na = True


class UltimateModelV8:
    """Model V8: Pushing beyond 0.57 with focused feature engineering"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 30
        self.dtype = np.float32
        self.models = []
        
        # Best parameters from V6/V7
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

    def create_focused_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume):
        """Ultra-focused features based on V7's top performers"""
        features = {}
        
        # === ULTRA PRIORITY: RSI DIVERGENCE (Top feature with 33k importance!) ===
        # Expand RSI divergence with multiple variations
        for period in [24, 36, 48, 60, 72, 96, 120, 168]:
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}'] = rsi.astype(self.dtype)
            
            # Multiple divergence calculations
            for div_period in [period, period//2, period*2]:
                if div_period >= 24:
                    price_change = (df_vwap / df_vwap.shift(div_period) - 1)
                    features[f'rsi_divergence_{period}_{div_period}'] = (
                        (rsi - rsi.shift(div_period)) - price_change * 100
                    ).astype(self.dtype)
            
            # RSI momentum
            features[f'rsi_momentum_{period}'] = (rsi - rsi.shift(period//2)).astype(self.dtype)
            
            # RSI extremes
            features[f'rsi_extreme_{period}'] = ((rsi - 50).abs() / 50).astype(self.dtype)
        
        # === SUPER CRITICAL: DISTANCE FROM LOW (2nd, 4th, 5th top features!) ===
        # Massively expand distance metrics
        for window in [48, 72, 96, 120, 168, 240, 336, 480, 672, 960]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            # Distance from low (percentage)
            dist_from_low = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'distance_from_low_{window}'] = dist_from_low
            
            # Distance from high
            dist_from_high = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            features[f'distance_from_high_{window}'] = dist_from_high
            
            # Relative position (0 to 1)
            price_pos = (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            features[f'price_position_{window}'] = price_pos.astype(self.dtype)
            
            # Position momentum
            if window >= 96:
                pos_change = price_pos - price_pos.shift(window//4)
                features[f'position_momentum_{window}'] = pos_change.astype(self.dtype)
            
            # Distance ratio
            features[f'distance_ratio_{window}'] = (
                dist_from_low / (dist_from_high + 1e-8)
            ).astype(self.dtype)
            
            # Breakout indicators
            features[f'near_high_{window}'] = (dist_from_high < 0.02).astype(self.dtype)
            features[f'near_low_{window}'] = (dist_from_low < 0.02).astype(self.dtype)
        
        # === HIGH PRIORITY: HL SPREAD STD (3rd, 6th, 10th features) ===
        hl_spread = (df_high - df_low) / df_vwap
        
        for window in [24, 36, 48, 72, 96, 120, 168, 240, 336, 480]:
            # Mean spread
            spread_mean = hl_spread.rolling(window, min_periods=window//2).mean()
            features[f'hl_spread_{window}'] = spread_mean.astype(self.dtype)
            
            # Spread standard deviation (key feature!)
            spread_std = hl_spread.rolling(window, min_periods=window//2).std()
            features[f'hl_spread_std_{window}'] = spread_std.astype(self.dtype)
            
            # Spread volatility
            spread_vol = spread_std / (spread_mean + 1e-8)
            features[f'spread_volatility_{window}'] = spread_vol.astype(self.dtype)
            
            # Spread expansion/contraction
            if window >= 48:
                spread_change = spread_mean / spread_mean.shift(window//2) - 1
                features[f'spread_change_{window}'] = spread_change.astype(self.dtype)
                
                # Spread acceleration
                spread_accel = spread_change - spread_change.shift(window//4)
                features[f'spread_acceleration_{window}'] = spread_accel.astype(self.dtype)
            
            # Spread percentile
            spread_pct = spread_mean.rolling(window*4, min_periods=window*2).rank(pct=True)
            features[f'spread_percentile_{window}'] = spread_pct.astype(self.dtype)
        
        # === IMPORTANT: VOLATILITY (still valuable) ===
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        
        for window in [48, 72, 96, 120, 168, 240, 336, 480, 672]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
            
            # Volatility regime
            if window >= 96:
                vol_change = vol / vol.shift(window//2) - 1
                features[f'volatility_change_{window}'] = vol_change.astype(self.dtype)
                
                # Volatility of volatility
                vol_vol = vol.rolling(window//2).std()
                features[f'vol_of_vol_{window}'] = vol_vol.astype(self.dtype)
        
        # === RETURN CONSISTENCY (important in V7) ===
        for window in [672, 960, 1344, 2016]:  # Longer windows showed importance
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
            
            # Return consistency
            ret_sign = (returns_15min > 0).astype(self.dtype)
            consistency = ret_sign.rolling(window, min_periods=window//2).mean()
            features[f'return_consistency_{window}'] = consistency.astype(self.dtype)
            
            # Trend strength
            if window >= 672:
                trend = df_vwap.rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                features[f'trend_strength_{window}'] = trend.astype(self.dtype)
        
        # === EFFICIENCY RATIO (good in V7) ===
        for window in [48, 72, 96, 120, 168, 240, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            eff_ratio = net_change / (total_change + 1e-8)
            features[f'efficiency_ratio_{window}'] = eff_ratio.astype(self.dtype)
            
            # Efficiency momentum
            if window >= 96:
                eff_change = eff_ratio - eff_ratio.shift(window//2)
                features[f'efficiency_momentum_{window}'] = eff_change.astype(self.dtype)
        
        # === INTERACTION FEATURES (between top performers) ===
        # RSI × Distance from low
        for period in [48, 96]:
            rsi = features[f'rsi_{period}']
            for window in [168, 336]:
                if f'distance_from_low_{window}' in features:
                    dist = features[f'distance_from_low_{window}']
                    features[f'rsi_{period}_x_dist_low_{window}'] = (rsi * dist / 100).astype(self.dtype)
        
        # Spread × Volatility
        for window in [96, 168]:
            if f'hl_spread_std_{window}' in features and f'volatility_{window}' in features:
                features[f'spread_vol_interaction_{window}'] = (
                    features[f'hl_spread_std_{window}'] * features[f'volatility_{window}']
                ).astype(self.dtype)
        
        # === VOLUME (keep essential ones) ===
        volume_usd = df_amount * df_vwap
        for window in [96, 336, 672]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_mean_{window}'] = vol_mean.astype(self.dtype)
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
            
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # === RELATIVE MOMENTUM ===
        for window in [24, 96, 336]:
            returns = (df_vwap / df_vwap.shift(window) - 1)
            market_return = returns.mean(axis=1)
            features[f'relative_momentum_{window}h'] = (
                returns.sub(market_return, axis=0)
            ).astype(self.dtype)
        
        # === MICROSTRUCTURE ===
        overnight_gap = (df_open - df_close.shift(1)) / (df_close.shift(1) + 1e-8)
        for window in [48, 96]:
            features[f'overnight_gap_mean_{window}'] = overnight_gap.rolling(
                window, min_periods=window//2
            ).mean().astype(self.dtype)
        
        # === Advanced volatility measures ===
        # Yang-Zhang (most accurate volatility)
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
        
        return features

    def train_ultimate_ensemble(self, X_train_scaled, y_train, feature_names):
        """Ultimate ensemble with 8 diverse models"""
        print("Training ultimate 8-model ensemble...")
        
        self.models = []
        self.model_weights = []
        
        base_params = self.best_params.copy()
        
        # Model 1: Best parameters
        print("Model 1/8: Best params (seed=42)")
        params1 = base_params.copy()
        params1.update({'seed': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42})
        
        lgb_train1 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model1 = lgb.train(params1, lgb_train1, num_boost_round=300)
        self.models.append(model1)
        self.model_weights.append(0.20)
        
        # Model 2: Different seed
        print("Model 2/8: Best params (seed=123)")
        params2 = base_params.copy()
        params2.update({'seed': 123, 'bagging_seed': 456, 'feature_fraction_seed': 789})
        
        lgb_train2 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model2 = lgb.train(params2, lgb_train2, num_boost_round=300)
        self.models.append(model2)
        self.model_weights.append(0.15)
        
        # Model 3: Lower learning rate
        print("Model 3/8: Lower LR")
        params3 = base_params.copy()
        params3['learning_rate'] = params3['learning_rate'] * 0.7
        params3.update({'seed': 999})
        
        lgb_train3 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model3 = lgb.train(params3, lgb_train3, num_boost_round=420)
        self.models.append(model3)
        self.model_weights.append(0.15)
        
        # Model 4: Huber loss
        print("Model 4/8: Huber loss")
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
        self.model_weights.append(0.125)
        
        # Model 5: MAE objective
        print("Model 5/8: MAE objective")
        params5 = base_params.copy()
        params5.update({
            'objective': 'regression_l1',
            'metric': 'mae',
            'seed': 5555
        })
        
        lgb_train5 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model5 = lgb.train(params5, lgb_train5, num_boost_round=300)
        self.models.append(model5)
        self.model_weights.append(0.125)
        
        # Model 6: More leaves
        print("Model 6/8: More leaves")
        params6 = base_params.copy()
        params6['num_leaves'] = 200
        params6.update({'seed': 7777})
        
        lgb_train6 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model6 = lgb.train(params6, lgb_train6, num_boost_round=280)
        self.models.append(model6)
        self.model_weights.append(0.10)
        
        # Model 7: Deeper trees
        print("Model 7/8: Deeper trees")
        params7 = base_params.copy()
        params7['max_depth'] = 12
        params7['min_data_in_leaf'] = 100
        params7.update({'seed': 8888, 'feature_pre_filter': False})
        
        lgb_train7 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model7 = lgb.train(params7, lgb_train7, num_boost_round=300)
        self.models.append(model7)
        self.model_weights.append(0.05)
        
        # Model 8: High regularization
        print("Model 8/8: High regularization")
        params8 = base_params.copy()
        params8['lambda_l1'] = params8['lambda_l1'] * 1.3
        params8['lambda_l2'] = params8['lambda_l2'] * 1.3
        params8.update({'seed': 9999})
        
        lgb_train8 = lgb.Dataset(X_train_scaled.values, label=y_train, free_raw_data=False)
        model8 = lgb.train(params8, lgb_train8, num_boost_round=320)
        self.models.append(model8)
        self.model_weights.append(0.05)
        
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

    def refined_post_processing(self, predictions, percentile_low=0.2, percentile_high=99.8):
        """Refined post-processing based on V7 success"""
        
        # 1. Adaptive clipping
        predictions = np.clip(predictions, 
                            np.percentile(predictions, percentile_low), 
                            np.percentile(predictions, percentile_high))
        
        # 2. Winsorization
        from scipy.stats import mstats
        predictions = mstats.winsorize(predictions, limits=(0.001, 0.001))
        
        # 3. Normalize
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        if std_pred > 0:
            predictions = (predictions - mean_pred) / std_pred * 0.022  # Slightly higher vol
            
            # 4. Smooth extremes
            extreme_threshold = 3.5
            extreme_mask = np.abs(predictions) > (0.022 * extreme_threshold)
            predictions[extreme_mask] *= 0.9
        
        return predictions

    def select_top_features(self, feature_importance_df, top_n=80):
        """Select only the top N most important features"""
        top_features = feature_importance_df.head(top_n)['feature'].tolist()
        return top_features

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Memory-efficient training with feature selection"""
        print("Starting ultimate training for V8...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use 40% for training
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
        
        print(f"Initial features: {len(feature_names)}")
        
        # First pass: Train a quick model to get feature importance
        print("Running feature selection...")
        from sklearn.preprocessing import RobustScaler
        scaler_temp = RobustScaler()
        X_temp = train_df[feature_names]
        X_temp_scaled = pd.DataFrame(
            scaler_temp.fit_transform(X_temp),
            index=X_temp.index,
            columns=X_temp.columns
        ).astype(self.dtype)
        
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Quick model for feature importance
        params_quick = self.best_params.copy()
        params_quick['num_boost_round'] = 100
        lgb_quick = lgb.Dataset(X_temp_scaled.values, label=y_train)
        model_quick = lgb.train(params_quick, lgb_quick, num_boost_round=100)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model_quick.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = self.select_top_features(importance, top_n=80)
        print(f"Selected top {len(top_features)} features")
        
        # Retrain with selected features
        X_train = train_df[top_features]
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        # Train ultimate ensemble
        self.train_ultimate_ensemble(X_train_scaled, y_train, top_features)
        
        self.feature_names = top_features
        self.scaler = scaler
        
        # Process all data in chunks
        print("Making predictions on all data...")
        chunk_size = 20
        results_list = []
        
        for i in range(0, len(all_symbol_list), chunk_size):
            chunk_symbols = all_symbol_list[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(all_symbol_list) + chunk_size - 1)//chunk_size}...")
            
            # Prepare chunk data with selected features only
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
            
            # Scale features
            X_chunk = chunk_df[top_features]
            X_chunk_scaled = pd.DataFrame(
                self.scaler.transform(X_chunk),
                index=X_chunk.index,
                columns=X_chunk.columns
            ).astype(self.dtype)
            
            # Make ensemble predictions
            predictions = self.predict_ensemble(X_chunk_scaled.values)
            
            # Refined post-processing
            predictions = self.refined_post_processing(predictions)
            
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
        
        # Create focused features
        print("Creating focused features based on V7 insights...")
        additional_features = self.create_focused_features(
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
            print("\nTop 20 most important features (from selected):")
            print(importance.head(20))
            
            # Save model info
            model_info = {
                'score': float(rho_overall),
                'timestamp': datetime.datetime.now().isoformat(),
                'n_models': len(self.models),
                'ensemble_weights': self.model_weights.tolist(),
                'top_features': importance.head(20)['feature'].tolist(),
                'total_features_selected': len(self.feature_names),
                'total_features_created': len(additional_features)
            }
            
            with open('./result/model_v8_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"\nFeatures: {len(self.feature_names)} selected from {len(additional_features)} created")

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
        df_submit_competition.to_csv("./result/submit_v8.csv", index=False)

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
    print("ULTIMATE MODEL V8 - TARGETING 0.57+ WEIGHTED SPEARMAN")
    print("="*70)
    print("\nKey improvements over V7 (0.5549):")
    print("1. MASSIVE expansion of RSI divergence features (top performer)")
    print("2. EXTENSIVE distance from low/high features (2nd-5th best)")
    print("3. ENHANCED HL spread std with acceleration metrics")
    print("4. Feature selection: Top 80 features only")
    print("5. 8-model ensemble (vs 6 in V7)")
    print("6. 40% training data (vs 35% in V7)")
    print("7. Interaction features between top performers")
    print("8. Refined post-processing")
    print("\n" + "="*70 + "\n")
    
    model = UltimateModelV8()
    model.run()