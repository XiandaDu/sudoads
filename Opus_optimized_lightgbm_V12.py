# optimized_lightgbm_v12_breakthrough.py
# Revolutionary approach targeting 0.6+ with ranking optimization

import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from scipy import stats
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')
import json

pd.options.mode.use_inf_as_na = True


class RevolutionaryModelV12:
    """V12: Revolutionary approach with ranking optimization"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.dtype = np.float32
        self.models = []
        
        # Optimized parameters for ranking
        self.ranking_params = {
            "objective": "lambdarank",  # Ranking objective!
            "metric": "ndcg",
            "num_leaves": 127,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 3,
            "lambda_l1": 0.5,
            "lambda_l2": 0.1,
            "min_data_in_leaf": 100,
            "max_depth": 8,
            'boosting_type': 'gbdt',
            'num_threads': mp.cpu_count() - 2,
            'verbose': -1,
            'force_row_wise': True,
            'label_gain': list(range(100))  # For ranking
        }
        
        # Regression params (backup)
        self.regression_params = {
            "objective": "regression",
            "num_leaves": 163,
            "learning_rate": 0.06,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.65,
            "bagging_freq": 2,
            "lambda_l1": 1.2,  # Higher regularization
            "lambda_l2": 0.3,
            "min_data_in_leaf": 150,
            "max_depth": 9,
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
        """Get all symbol kline data"""
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
        """Weighted Spearman correlation"""
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

    def create_revolutionary_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close):
        """Revolutionary feature engineering focused on ranking signals"""
        features = {}
        
        # === CORE WINNING FEATURES (from V7/V9 success) ===
        
        # 1. RSI Divergence (Critical for success)
        for period in [48]:  # Focus on best period
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}'] = rsi.astype(self.dtype)
            
            # Multiple divergence calculations
            for div_period in [24, 48, 96]:
                price_change = (df_vwap / df_vwap.shift(div_period) - 1)
                features[f'rsi_divergence_{period}_{div_period}'] = (
                    (rsi - rsi.shift(div_period)) - price_change * 100
                ).astype(self.dtype)
        
        # 2. Distance from Low (Critical)
        for window in [96, 168, 336]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            dist_from_low = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'distance_from_low_{window}'] = dist_from_low
            features[f'distance_from_high_{window}'] = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # 3. Volatility (Important)
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [96, 336, 672]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
        
        # 4. HL Spread
        hl_spread = (df_high - df_low) / df_vwap
        for window in [48, 96, 336]:
            features[f'hl_spread_std_{window}'] = hl_spread.rolling(
                window, min_periods=window//2
            ).std().astype(self.dtype)
            
            spread_mean = hl_spread.rolling(window, min_periods=window//2).mean()
            features[f'spread_change_{window}'] = (
                spread_mean / spread_mean.shift(window//2) - 1
            ).astype(self.dtype)
        
        # === REVOLUTIONARY NEW FEATURES ===
        
        # 5. Ranking-Specific Features
        for window in [96, 336]:
            ret = (df_vwap / df_vwap.shift(window) - 1)
            
            # Cross-sectional rank
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
            
            # Rank momentum
            rank_current = df_vwap.rank(axis=1, pct=True)
            rank_past = df_vwap.shift(window).rank(axis=1, pct=True)
            features[f'rank_momentum_{window}'] = (rank_current - rank_past).astype(self.dtype)
            
            # Percentile features
            features[f'return_{window}_percentile'] = ret.rolling(
                window*4, min_periods=window*2
            ).rank(pct=True).astype(self.dtype)
        
        # 6. Market Regime Detection
        market_vol = returns_15min.mean(axis=1).rolling(336).std()
        vol_percentile = market_vol.rolling(336*4).rank(pct=True)
        
        # High/Low volatility regime
        features['high_vol_regime'] = (vol_percentile > 0.7).astype(self.dtype)
        features['low_vol_regime'] = (vol_percentile < 0.3).astype(self.dtype)
        
        # 7. Relative Strength
        for window in [24, 96]:
            returns = (df_vwap / df_vwap.shift(window) - 1)
            market_return = returns.mean(axis=1)
            relative_return = returns.sub(market_return, axis=0)
            
            # Relative strength rank
            features[f'relative_strength_{window}'] = relative_return.astype(self.dtype)
            features[f'relative_strength_rank_{window}'] = relative_return.rank(
                axis=1, pct=True
            ).astype(self.dtype)
            
            # Winners/Losers
            features[f'is_winner_{window}'] = (relative_return > 0).astype(self.dtype)
            features[f'is_extreme_winner_{window}'] = (
                relative_return.rank(axis=1, pct=True) > 0.9
            ).astype(self.dtype)
            features[f'is_extreme_loser_{window}'] = (
                relative_return.rank(axis=1, pct=True) < 0.1
            ).astype(self.dtype)
        
        # 8. Mean Reversion Signals
        for window in [96, 336]:
            # Z-score
            ma = df_vwap.rolling(window, min_periods=window//2).mean()
            std = df_vwap.rolling(window, min_periods=window//2).std()
            z_score = (df_vwap - ma) / (std + 1e-8)
            features[f'z_score_{window}'] = z_score.astype(self.dtype)
            
            # Extreme z-score flags
            features[f'extreme_high_{window}'] = (z_score > 2).astype(self.dtype)
            features[f'extreme_low_{window}'] = (z_score < -2).astype(self.dtype)
        
        # 9. Momentum Quality
        for window in [96, 336]:
            returns = df_vwap / df_vwap.shift(window) - 1
            
            # Smoothness of momentum
            momentum_volatility = returns.rolling(window//2).std()
            momentum_mean = returns.rolling(window//2).mean()
            features[f'momentum_quality_{window}'] = (
                momentum_mean / (momentum_volatility + 1e-8)
            ).astype(self.dtype)
        
        # 10. Volume Patterns
        volume_usd = df_amount * df_vwap
        for window in [96, 336]:
            vol_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            vol_rank = vol_mean.rank(axis=1, pct=True)
            features[f'volume_rank_{window}'] = vol_rank.astype(self.dtype)
            
            # Volume surge
            vol_ratio = vol_mean / vol_mean.shift(window)
            features[f'volume_surge_{window}'] = (vol_ratio > 2).astype(self.dtype)
        
        # === INTERACTION FEATURES ===
        
        # Key interactions from successful models
        if 'distance_from_low_96' in features and 'rsi_divergence_48_48' in features:
            features['magic_interaction'] = (
                features['distance_from_low_96'] * features['rsi_divergence_48_48'] / 100
            ).astype(self.dtype)
        
        # Regime-conditional features
        if 'high_vol_regime' in features:
            for key in ['volatility_336', 'distance_from_low_168']:
                if key in features:
                    features[f'{key}_high_vol'] = (
                        features[key] * features['high_vol_regime']
                    ).astype(self.dtype)
                    features[f'{key}_low_vol'] = (
                        features[key] * features['low_vol_regime']
                    ).astype(self.dtype)
        
        return features

    def prepare_ranking_data(self, X, y):
        """Prepare data for ranking objective"""
        # Group by timestamp
        groups = []
        current_group_size = 0
        
        for timestamp in X.index.get_level_values(0).unique():
            timestamp_size = len(X.loc[timestamp])
            groups.append(timestamp_size)
        
        # Convert targets to ranks within each timestamp
        y_ranked = []
        for timestamp in X.index.get_level_values(0).unique():
            timestamp_targets = y[X.index.get_level_values(0) == timestamp]
            # Convert to ranks (higher return = higher rank)
            ranks = rankdata(timestamp_targets, method='average')
            # Normalize to [0, 99] for LambdaRank
            ranks_normalized = (ranks - 1) / (len(ranks) - 1) * 99
            y_ranked.extend(ranks_normalized)
        
        return np.array(y_ranked), groups

    def train_ranking_ensemble(self, X_train_scaled, y_train, feature_names):
        """Train ensemble with ranking objective"""
        print("Training ranking ensemble...")
        
        self.models = []
        self.model_weights = []
        
        # Prepare ranking data
        y_ranked, groups = self.prepare_ranking_data(X_train_scaled, y_train)
        
        # Model 1: LambdaRank
        print("Model 1/4: LambdaRank")
        try:
            lgb_train_rank = lgb.Dataset(
                X_train_scaled.values, 
                label=y_ranked,
                group=groups[:100]  # Use subset of groups for memory
            )
            
            model1 = lgb.train(
                self.ranking_params,
                lgb_train_rank,
                num_boost_round=200
            )
            self.models.append(model1)
            self.model_weights.append(0.3)
        except:
            print("LambdaRank failed, using regression")
            lgb_train = lgb.Dataset(X_train_scaled.values, label=y_train)
            model1 = lgb.train(self.regression_params, lgb_train, num_boost_round=300)
            self.models.append(model1)
            self.model_weights.append(0.3)
        
        # Model 2: Regression with high regularization
        print("Model 2/4: High regularization")
        params2 = self.regression_params.copy()
        params2['lambda_l1'] = 1.5
        params2['lambda_l2'] = 0.5
        params2['seed'] = 42
        
        lgb_train2 = lgb.Dataset(X_train_scaled.values, label=y_train)
        model2 = lgb.train(params2, lgb_train2, num_boost_round=300)
        self.models.append(model2)
        self.model_weights.append(0.25)
        
        # Model 3: Huber (robust to outliers)
        print("Model 3/4: Huber")
        params3 = self.regression_params.copy()
        params3.update({
            'objective': 'huber',
            'alpha': 0.9,
            'metric': 'mae',
            'seed': 123
        })
        
        lgb_train3 = lgb.Dataset(X_train_scaled.values, label=y_train)
        model3 = lgb.train(params3, lgb_train3, num_boost_round=300)
        self.models.append(model3)
        self.model_weights.append(0.25)
        
        # Model 4: Quantile regression (median)
        print("Model 4/4: Quantile")
        params4 = self.regression_params.copy()
        params4.update({
            'objective': 'quantile',
            'alpha': 0.5,
            'metric': 'quantile',
            'seed': 999
        })
        
        lgb_train4 = lgb.Dataset(X_train_scaled.values, label=y_train)
        model4 = lgb.train(params4, lgb_train4, num_boost_round=300)
        self.models.append(model4)
        self.model_weights.append(0.20)
        
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

    def revolutionary_post_processing(self, predictions, X_original=None):
        """Revolutionary post-processing focused on ranking"""
        
        # 1. Initial clipping (gentle)
        predictions = np.clip(predictions, 
                            np.percentile(predictions, 0.1), 
                            np.percentile(predictions, 99.9))
        
        # 2. Rank-based calibration
        ranks = rankdata(predictions)
        n = len(predictions)
        
        # Transform ranks to normal distribution
        # This preserves ranking while creating better distribution
        uniform_ranks = (ranks - 0.5) / n
        normal_scores = stats.norm.ppf(uniform_ranks)
        
        # Clip extreme values from normal transformation
        normal_scores = np.clip(normal_scores, -3, 3)
        
        # 3. Scale to target distribution
        # Target: mean=0, std=0.022 (2.2% daily volatility)
        predictions_calibrated = normal_scores * 0.022
        
        # 4. Preserve extreme signals (top/bottom 5%)
        top_mask = ranks > (n * 0.95)
        bottom_mask = ranks < (n * 0.05)
        
        # Enhance extreme signals
        predictions_calibrated[top_mask] *= 1.1
        predictions_calibrated[bottom_mask] *= 1.1
        
        # 5. Final safety check
        predictions_calibrated = np.clip(predictions_calibrated, -0.15, 0.15)
        
        return predictions_calibrated

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Efficient training with focus on ranking"""
        print("Starting revolutionary training...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use 30% for training (like successful V7)
        print("Preparing training data (30% of symbols)...")
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.30)]
        
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
        
        print(f"Total features: {len(feature_names)}")
        
        # Feature selection based on quick importance check
        X_temp = train_df[feature_names]
        y_temp = train_df['target'].values
        
        # Quick feature importance
        print("Running feature selection...")
        from sklearn.preprocessing import RobustScaler
        scaler_temp = RobustScaler()
        X_temp_scaled = pd.DataFrame(
            scaler_temp.fit_transform(X_temp),
            index=X_temp.index,
            columns=X_temp.columns
        )
        
        lgb_quick = lgb.Dataset(X_temp_scaled.values, label=y_temp)
        model_quick = lgb.train(
            self.regression_params, 
            lgb_quick, 
            num_boost_round=50
        )
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model_quick.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Select top 50 features
        top_features = importance.head(50)['feature'].tolist()
        print(f"Selected top {len(top_features)} features")
        
        # Retrain with selected features
        X_train = train_df[top_features]
        y_train = train_df['target'].values.astype(self.dtype)
        
        # Use QuantileTransformer for better distribution
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(
            n_quantiles=1000,
            output_distribution='uniform',
            random_state=42
        )
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        # Train ranking ensemble
        self.train_ranking_ensemble(X_train_scaled, y_train, top_features)
        
        self.feature_names = top_features
        self.scaler = scaler
        
        # Free memory
        del train_features, train_df, X_train, X_train_scaled
        gc.collect()
        
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
            
            # Process each timestamp separately for better ranking
            chunk_results = chunk_df[['target']].copy()
            processed_predictions = []
            
            for timestamp in chunk_df.index.get_level_values(0).unique():
                timestamp_mask = chunk_df.index.get_level_values(0) == timestamp
                timestamp_preds = predictions[timestamp_mask]
                
                # Apply ranking-based post-processing per timestamp
                if len(timestamp_preds) > 1:
                    timestamp_preds = self.revolutionary_post_processing(timestamp_preds)
                
                processed_predictions.extend(timestamp_preds)
            
            chunk_results['y_pred'] = processed_predictions
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
        
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        print("Creating revolutionary features...")
        additional_features = self.create_revolutionary_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close
        )
        
        del df_high, df_low, df_open, df_close
        gc.collect()
        
        # Train
        results_df = self.train_efficient(
            df_target, df_factor1, df_factor2, df_factor3, 
            additional_features, all_symbol_list
        )
        
        # Calculate performance
        rho_overall = self.weighted_spearmanr(results_df['target'], results_df['y_pred'])
        print(f"\n{'='*60}")
        print(f"🎯 Final Weighted Spearman: {rho_overall:.4f}")
        print(f"{'='*60}")
        
        # Prepare submission
        self._prepare_submission(results_df)
        
        # Save info
        if hasattr(self, 'models') and len(self.models) > 0:
            # Get feature importance from first model
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 features:")
            print(importance.head(15))
            
            model_info = {
                'score': float(rho_overall),
                'timestamp': datetime.datetime.now().isoformat(),
                'n_models': len(self.models),
                'model_weights': self.model_weights.tolist(),
                'top_features': importance.head(15)['feature'].tolist(),
                'total_features': len(self.feature_names)
            }
            
            with open('./result/model_v12_breakthrough_info.json', 'w') as f:
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
        df_submit_competition.to_csv("./result/submit_v12_breakthrough.csv", index=False)

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
    print("🚀 MODEL V12 - REVOLUTIONARY BREAKTHROUGH FOR 0.6+")
    print("="*70)
    print("\n💡 Revolutionary Features:")
    print("1. RANKING OPTIMIZATION: LambdaRank objective")
    print("2. RANK-BASED POST-PROCESSING: Per-timestamp calibration")
    print("3. REGIME DETECTION: Adaptive to market conditions")
    print("4. WINNER/LOSER SIGNALS: Extreme performance flags")
    print("5. MOMENTUM QUALITY: Smooth vs choppy trends")
    print("6. SIMPLIFIED ARCHITECTURE: 30% training (like V7)")
    print("7. TOP 50 FEATURES: Focused on proven winners")
    print("8. QUANTILE TRANSFORMER: Better feature distribution")
    print("\n" + "="*70 + "\n")
    
    model = RevolutionaryModelV12()
    model.run()