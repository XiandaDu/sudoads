# optimized_lightgbm_v6.py
# Stabilized version targeting consistent 0.40+ performance

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


class StabilizedModelV6:
    """Model V6: Focus on stability and consistency at 0.40+ level"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 30
        self.dtype = np.float32
        self.models = []
        
        # Lock in the best parameters that achieved 0.4074
        self.best_params = {
            "objective": "regression",
            "num_leaves": 163,
            "learning_rate": 0.07182861492338714,
            "feature_fraction": 0.8184386555861791,
            "bagging_fraction": 0.6985394411684631,
            "bagging_freq": 2,
            "lambda_l1": 0.7113790560603763,  # Strong L1 - key to success!
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
        """Same as V5"""
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
        """Same as V5"""
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
        """Same features as V5 - they worked well!"""
        features = {}
        
        # 1. Core returns
        for window in [1, 4, 24, 96, 336]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # Long-term returns
        for window in [168, 672]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # 2. Volatility features
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
        
        # Volatility ratio and regime
        vol_96 = returns_15min.rolling(96, min_periods=48).std()
        vol_336 = returns_15min.rolling(336, min_periods=168).std()
        features['volatility_ratio_96_336'] = (vol_96 / (vol_336 + 1e-8)).astype(self.dtype)
        
        # Volatility percentile
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            vol_percentile = vol.rolling(window*4, min_periods=window*2).rank(pct=True)
            features[f'volatility_percentile_{window}'] = vol_percentile.astype(self.dtype)
        
        # 3. Volume patterns
        for window in [96, 336]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_mean_{window}'] = vol_mean.astype(self.dtype)
            features[f'volume_ratio_{window}'] = (
                vol_mean / vol_mean.shift(window)
            ).astype(self.dtype)
        
        # Volume in USD
        volume_usd = df_amount * df_vwap
        for window in [96, 336]:
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}'] = vol_usd_mean.astype(self.dtype)
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # 4. Price position
        for window in [336]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # Price efficiency ratio
        for window in [96, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            features[f'efficiency_ratio_{window}'] = (
                net_change / (total_change + 1e-8)
            ).astype(self.dtype)
        
        # 5. Momentum
        features['momentum_96'] = (df_vwap / df_vwap.shift(96) - 1).astype(self.dtype)
        
        # Momentum consistency
        for window in [96, 336]:
            positive_periods = (returns_15min > 0).rolling(window, min_periods=window//2).sum()
            features[f'momentum_consistency_{window}'] = (
                positive_periods / window
            ).astype(self.dtype)
        
        # 6. RSI
        delta = df_vwap.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=96, min_periods=48).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=96, min_periods=48).mean()
        rs = gain / (loss + 1e-8)
        features['rsi_96'] = (100 - (100 / (1 + rs))).astype(self.dtype)
        
        # RSI divergence
        rsi = features['rsi_96']
        price_change = (df_vwap / df_vwap.shift(96) - 1)
        features['rsi_divergence'] = (
            (rsi - rsi.shift(96)) - price_change * 100
        ).astype(self.dtype)
        
        # 7. Market microstructure
        overnight_gap = (df_open - df_close.shift(1)) / (df_close.shift(1) + 1e-8)
        features['overnight_gap_mean_96'] = overnight_gap.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        hl_spread = (df_high - df_low) / df_vwap
        features['hl_spread_96'] = hl_spread.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        # 8. Cross-sectional features
        returns_24h = (df_vwap / df_vwap.shift(96) - 1)
        market_return = returns_24h.mean(axis=1)
        features['relative_momentum_24h'] = (
            returns_24h.sub(market_return, axis=0)
        ).astype(self.dtype)
        
        return features

    def train_robust_ensemble(self, X_train_scaled, y_train, feature_names):
        """Train a more robust ensemble with locked parameters"""
        print("Training robust ensemble with locked best parameters...")
        
        self.models = []
        self.model_weights = []
        
        # Base parameters that achieved 0.4074
        base_params = self.best_params.copy()
        
        # Model 1: Exact best parameters (highest weight)
        print("Model 1/5: Best parameters (seed=42)")
        params1 = base_params.copy()
        params1.update({'seed': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42})
        
        lgb_train = lgb.Dataset(X_train_scaled.values, label=y_train)
        model1 = lgb.train(params1, lgb_train, num_boost_round=300)
        self.models.append(model1)
        self.model_weights.append(0.35)  # Highest weight for best params
        
        # Model 2: Different seed
        print("Model 2/5: Best parameters (seed=123)")
        params2 = base_params.copy()
        params2.update({'seed': 123, 'bagging_seed': 456, 'feature_fraction_seed': 789})
        
        model2 = lgb.train(params2, lgb_train, num_boost_round=300)
        self.models.append(model2)
        self.model_weights.append(0.25)
        
        # Model 3: Slightly lower learning rate
        print("Model 3/5: Lower learning rate")
        params3 = base_params.copy()
        params3['learning_rate'] = params3['learning_rate'] * 0.85
        params3.update({'seed': 999})
        
        model3 = lgb.train(params3, lgb_train, num_boost_round=350)
        self.models.append(model3)
        self.model_weights.append(0.20)
        
        # Model 4: Huber loss
        print("Model 4/5: Huber loss")
        params4 = base_params.copy()
        params4.update({
            'objective': 'huber',
            'alpha': 0.9,
            'metric': 'mae',
            'seed': 2024
        })
        
        model4 = lgb.train(params4, lgb_train, num_boost_round=300)
        self.models.append(model4)
        self.model_weights.append(0.10)
        
        # Model 5: Slightly simplified
        print("Model 5/5: Simplified model")
        params5 = base_params.copy()
        params5['num_leaves'] = 140
        params5['min_data_in_leaf'] = 160
        params5.update({'seed': 8888})
        
        model5 = lgb.train(params5, lgb_train, num_boost_round=300)
        self.models.append(model5)
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

    def post_process_predictions(self, predictions, method='conservative_clip'):
        """Conservative post-processing to maintain stability"""
        
        if method == 'conservative_clip':
            # More conservative clipping for stability
            predictions = np.clip(predictions, 
                                np.percentile(predictions, 0.3), 
                                np.percentile(predictions, 99.7))
            
            # Normalize conservatively
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            if std_pred > 0:
                # Scale to typical crypto volatility
                predictions = (predictions - mean_pred) / std_pred * 0.018  # 1.8% vol
                
        return predictions

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Memory-efficient training"""
        print("Starting robust training with locked parameters...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use consistent training set size
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
        
        # Train robust ensemble
        self.train_robust_ensemble(X_train_scaled, y_train, feature_names)
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        # Process all data in chunks
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
            
            # Conservative post-processing
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
        
        # Free memory
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        # Create features
        print("Creating enhanced features...")
        additional_features = self.create_enhanced_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close
        )
        
        # Free memory
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
        
        # Feature importance
        if hasattr(self, 'models') and len(self.models) > 0:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 15 most important features:")
            print(importance.head(15))
            
            # Save model info
            model_info = {
                'score': rho_overall,
                'timestamp': datetime.datetime.now().isoformat(),
                'n_models': len(self.models),
                'ensemble_weights': self.model_weights.tolist(),
                'top_features': importance.head(10)['feature'].tolist()
            }
            
            with open('./result/model_v6_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)

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
        df_submit_competition.to_csv("./result/submit_v6.csv", index=False)

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
    print("STABILIZED MODEL V6 - MAINTAINING 0.40+ WEIGHTED SPEARMAN")
    print("="*70)
    print("\nUsing locked parameters that achieved 0.4074:")
    print("- Strong L1 regularization (0.711)")
    print("- 5-model ensemble with different seeds")
    print("- Conservative post-processing")
    print("- Focus on stability over marginal gains")
    print("\n" + "="*70 + "\n")
    
    model = StabilizedModelV6()
    model.run()