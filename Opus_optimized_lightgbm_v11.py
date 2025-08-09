# optimized_lightgbm_v11_stacking.py
# Breakthrough version with meta-learning and improved feature selection

import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import lightgbm as lgb
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json

pd.options.mode.use_inf_as_na = True


class MetaLearningModel:
    """V11: Meta-learning with stacking for breakthrough performance"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 30
        self.dtype = np.float32
        self.base_models = []
        self.meta_model = None
        
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

    def create_core_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close):
        """Create core features based on proven winners"""
        features = {}
        
        # === DISTANCE FROM LOW (Top performer) ===
        for window in [72, 84, 96, 108, 120, 168, 240, 336, 480, 672]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            
            dist_from_low = ((df_vwap - low_roll) / df_vwap).astype(self.dtype)
            features[f'distance_from_low_{window}'] = dist_from_low
            
            # Special for 96
            if window == 96:
                features[f'distance_from_low_{window}_sq'] = (dist_from_low ** 2).astype(self.dtype)
                features[f'distance_from_low_{window}_rank'] = dist_from_low.rank(axis=1, pct=True).astype(self.dtype)
            
            if window in [96, 168, 336, 672]:
                features[f'distance_from_high_{window}'] = ((high_roll - df_vwap) / df_vwap).astype(self.dtype)
            
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # === RSI DIVERGENCE ===
        for period in [36, 42, 48, 60, 72, 96]:
            delta = df_vwap.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}'] = rsi.astype(self.dtype)
            
            price_change = (df_vwap / df_vwap.shift(period) - 1)
            features[f'rsi_divergence_{period}'] = (
                (rsi - rsi.shift(period)) - price_change * 100
            ).astype(self.dtype)
        
        # RSI_48 variations
        rsi_48 = features['rsi_48']
        for shift in [24, 36, 72]:
            price_change = (df_vwap / df_vwap.shift(shift) - 1)
            features[f'rsi_divergence_48_{shift}'] = (
                (rsi_48 - rsi_48.shift(shift)) - price_change * 100
            ).astype(self.dtype)
        
        # === SPREAD ===
        hl_spread = (df_high - df_low) / df_vwap
        for window in [48, 96, 168, 336]:
            features[f'hl_spread_{window}'] = hl_spread.rolling(window, min_periods=window//2).mean().astype(self.dtype)
            features[f'hl_spread_std_{window}'] = hl_spread.rolling(window, min_periods=window//2).std().astype(self.dtype)
            
            if window >= 48:
                spread_mean = features[f'hl_spread_{window}']
                features[f'spread_change_{window}'] = (spread_mean / spread_mean.shift(window//2) - 1).astype(self.dtype)
        
        # === VOLATILITY ===
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [48, 96, 168, 336, 672]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
            
            if window in [96, 336]:
                features[f'volatility_change_{window}'] = (vol / vol.shift(window//2) - 1).astype(self.dtype)
        
        # === RETURNS ===
        for window in [1, 4, 24, 96, 168, 336, 672, 1344]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # === EFFICIENCY ===
        for window in [96, 168, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window, min_periods=window//2).sum()
            features[f'efficiency_ratio_{window}'] = (net_change / (total_change + 1e-8)).astype(self.dtype)
        
        # === VOLUME ===
        volume_usd = df_amount * df_vwap
        for window in [96, 336]:
            vol_mean = df_amount.rolling(window, min_periods=window//2).mean()
            features[f'volume_ratio_{window}'] = (vol_mean / vol_mean.shift(window)).astype(self.dtype)
            
            vol_usd_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}_rank'] = vol_usd_mean.rank(axis=1, pct=True).astype(self.dtype)
        
        # === INTERACTIONS ===
        if 'distance_from_low_96' in features and 'rsi_divergence_48' in features:
            features['dist_96_x_rsi_48'] = (
                features['distance_from_low_96'] * features['rsi_divergence_48'] / 100
            ).astype(self.dtype)
        
        return features

    def get_averaged_feature_importance(self, models, feature_names):
        """Get averaged feature importance across all models"""
        importance_sum = np.zeros(len(feature_names))
        
        for i, model in enumerate(models):
            importance = model.feature_importance(importance_type='gain')
            # Normalize importance for each model
            if importance.sum() > 0:
                importance = importance / importance.sum()
            importance_sum += importance
        
        # Average
        importance_avg = importance_sum / len(models)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_avg
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def train_base_models_with_cv(self, X_train, y_train, n_folds=3):
        """Train base models with cross-validation to get OOF predictions"""
        print(f"Training base models with {n_folds}-fold CV...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_predictions = []
        self.base_models = []
        
        # Different model configurations
        model_configs = [
            {'name': 'Base', 'params': {}},
            {'name': 'Lower LR', 'params': {'learning_rate': self.best_params['learning_rate'] * 0.7}},
            {'name': 'Huber', 'params': {'objective': 'huber', 'alpha': 0.9, 'metric': 'mae'}},
            {'name': 'MAE', 'params': {'objective': 'regression_l1', 'metric': 'mae'}},
            {'name': 'More leaves', 'params': {'num_leaves': 200}}
        ]
        
        for config in model_configs:
            print(f"Training {config['name']} model...")
            model_oof = np.zeros(len(y_train))
            fold_models = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train[val_idx]
                
                # Prepare parameters
                params = self.best_params.copy()
                params.update(config['params'])
                params['seed'] = fold + 42
                
                # Train model
                lgb_train = lgb.Dataset(X_fold_train.values, label=y_fold_train)
                lgb_val = lgb.Dataset(X_fold_val.values, label=y_fold_val)
                
                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_val],
                    num_boost_round=300,
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                
                fold_models.append(model)
                
                # Get OOF predictions
                model_oof[val_idx] = model.predict(X_fold_val.values, num_iteration=model.best_iteration)
            
            oof_predictions.append(model_oof)
            self.base_models.extend(fold_models)
        
        # Stack OOF predictions
        oof_stack = np.column_stack(oof_predictions)
        
        print(f"OOF predictions shape: {oof_stack.shape}")
        
        return oof_stack

    def train_meta_model(self, oof_predictions, y_train):
        """Train meta-model on OOF predictions"""
        print("Training meta-model...")
        
        # Add diversity features
        meta_features = pd.DataFrame(oof_predictions)
        meta_features.columns = [f'base_pred_{i}' for i in range(oof_predictions.shape[1])]
        
        # Add statistical features
        meta_features['mean_pred'] = oof_predictions.mean(axis=1)
        meta_features['std_pred'] = oof_predictions.std(axis=1)
        meta_features['min_pred'] = oof_predictions.min(axis=1)
        meta_features['max_pred'] = oof_predictions.max(axis=1)
        meta_features['median_pred'] = np.median(oof_predictions, axis=1)
        
        # Train meta-model
        meta_params = {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'metric': 'rmse'
        }
        
        lgb_meta = lgb.Dataset(meta_features.values, label=y_train)
        self.meta_model = lgb.train(meta_params, lgb_meta, num_boost_round=100)
        
        # Calculate meta-model performance
        meta_pred = self.meta_model.predict(meta_features.values)
        meta_score = self.weighted_spearmanr(y_train, meta_pred)
        print(f"Meta-model training score: {meta_score:.4f}")
        
        return meta_features.columns.tolist()

    def predict_with_stacking(self, X, base_models, meta_model, meta_feature_names):
        """Make predictions using stacking"""
        # Get base model predictions
        base_preds = []
        for model in base_models:
            pred = model.predict(X, num_iteration=model.best_iteration)
            base_preds.append(pred)
        
        base_preds = np.column_stack(base_preds)
        
        # Create meta features
        meta_features = pd.DataFrame(base_preds[:, :len(meta_feature_names)-5])  # Exclude statistical features
        meta_features.columns = [f'base_pred_{i}' for i in range(meta_features.shape[1])]
        
        # Add statistical features
        meta_features['mean_pred'] = base_preds.mean(axis=1)
        meta_features['std_pred'] = base_preds.std(axis=1)
        meta_features['min_pred'] = base_preds.min(axis=1)
        meta_features['max_pred'] = base_preds.max(axis=1)
        meta_features['median_pred'] = np.median(base_preds, axis=1)
        
        # Meta-model prediction
        final_pred = meta_model.predict(meta_features.values)
        
        return final_pred

    def post_process(self, predictions):
        """Post-processing"""
        predictions = np.clip(predictions, 
                            np.percentile(predictions, 0.15), 
                            np.percentile(predictions, 99.85))
        
        from scipy.stats import mstats
        predictions = mstats.winsorize(predictions, limits=(0.002, 0.002))
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        if std_pred > 0:
            predictions = (predictions - mean_pred) / std_pred * 0.022
            extreme_mask = np.abs(predictions) > (0.022 * 3)
            predictions[extreme_mask] *= 0.87
        
        return predictions

    def train_stacking(self, df_target, df_factor1, df_factor2, df_factor3, 
                      additional_features, all_symbol_list):
        """Train with stacking"""
        print("Starting stacking training...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use 35% for training
        print("Preparing training data (35% of symbols)...")
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.35)]
        
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
        
        # Scale features
        X_train = train_df[feature_names]
        y_train = train_df['target'].values.astype(self.dtype)
        
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        ).astype(self.dtype)
        
        # Train base models with CV
        oof_predictions = self.train_base_models_with_cv(X_train_scaled, y_train)
        
        # Train meta-model
        meta_feature_names = self.train_meta_model(oof_predictions, y_train)
        
        # Get averaged feature importance
        importance = self.get_averaged_feature_importance(self.base_models, feature_names)
        print("\nTop 15 features (averaged across all models):")
        print(importance.head(15))
        
        # Select top features
        top_features = importance.head(60)['feature'].tolist()
        print(f"\nUsing top {len(top_features)} features for final prediction")
        
        self.feature_names = top_features
        self.scaler = scaler
        self.meta_feature_names = meta_feature_names
        
        # Process all data
        print("\nMaking final predictions with stacking...")
        chunk_size = 15
        results_list = []
        
        for i in range(0, len(all_symbol_list), chunk_size):
            chunk_symbols = all_symbol_list[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(all_symbol_list) + chunk_size - 1)//chunk_size}...")
            
            # Prepare chunk
            chunk_features = []
            for feat_name in feature_names:  # Use all features, not just top
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
            X_chunk = chunk_df[feature_names]
            X_chunk_scaled = pd.DataFrame(
                self.scaler.transform(X_chunk),
                index=X_chunk.index,
                columns=X_chunk.columns
            ).astype(self.dtype)
            
            # Stacking prediction
            predictions = self.predict_with_stacking(
                X_chunk_scaled.values, 
                self.base_models, 
                self.meta_model,
                self.meta_feature_names
            )
            predictions = self.post_process(predictions)
            
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
        
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr
        gc.collect()
        
        print("Creating features...")
        additional_features = self.create_core_features(
            df_vwap, df_high, df_low, df_amount, df_open, df_close
        )
        
        del df_high, df_low, df_open, df_close
        gc.collect()
        
        # Train with stacking
        results_df = self.train_stacking(
            df_target, df_factor1, df_factor2, df_factor3, 
            additional_features, all_symbol_list
        )
        
        # Calculate performance
        rho_overall = self.weighted_spearmanr(results_df['target'], results_df['y_pred'])
        print(f"\n{'='*50}")
        print(f"Final Weighted Spearman: {rho_overall:.4f}")
        print(f"{'='*50}")
        
        # Prepare submission
        self._prepare_submission(results_df)
        
        # Save model info
        model_info = {
            'score': float(rho_overall),
            'timestamp': datetime.datetime.now().isoformat(),
            'n_base_models': len(self.base_models),
            'has_meta_model': True,
            'total_features': len(self.feature_names)
        }
        
        with open('./result/model_v11_stacking_info.json', 'w') as f:
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
        df_submit_competition.to_csv("./result/submit_v11_stacking.csv", index=False)

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
    print("MODEL V11 - META-LEARNING WITH STACKING")
    print("="*70)
    print("\n🚀 Breakthrough Features:")
    print("1. STACKING: Two-level learning architecture")
    print("2. Cross-validation for OOF predictions")
    print("3. Meta-model learns from base model diversity")
    print("4. Averaged feature importance across ALL models")
    print("5. Statistical meta-features (mean, std, min, max, median)")
    print("6. 5 diverse base models × 3 folds = 15 base models")
    print("7. Optimized for Weighted Spearman correlation")
    print("\n" + "="*70 + "\n")
    
    model = MetaLearningModel()
    model.run()