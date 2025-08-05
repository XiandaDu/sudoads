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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA Available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("CUDA not available, using CPU")
    device = torch.device('cpu')

pd.options.mode.use_inf_as_na = True


class NeuralNetworkRanker(nn.Module):
    """Neural network for learning complex ranking patterns"""
    
    def __init__(self, input_dim, hidden_sizes=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class GPUEnhancedModelV7:
    """Advanced model using GPU acceleration and neural networks"""
    
    def __init__(self):
        self.train_data_path = "./avenir-hku-web/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.batch_size = 50  # Larger batch size with server
        self.dtype = np.float32
        self.models = []
        self.nn_model = None
        
        # Best parameters from V6
        self.best_params = {
            "objective": "regression",
            "num_leaves": 255,  # Increased for server
            "learning_rate": 0.07182861492338714,
            "feature_fraction": 0.8184386555861791,
            "bagging_fraction": 0.6985394411684631,
            "bagging_freq": 2,
            "lambda_l1": 0.7113790560603763,
            "lambda_l2": 0.09201149241276538,
            "min_data_in_leaf": 100,  # Slightly reduced
            "max_depth": 12,  # Increased depth
            'boosting_type': 'gbdt',
            'num_threads': -1,  # Use all threads
            'verbose': -1,
            'metric': 'rmse'
        }
        
        # GPU parameters for LightGBM (if GPU version installed)
        self.gpu_params = self.best_params.copy()
        self.gpu_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,  # Use float32 for speed
        })

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
        """Parallel data loading with more workers"""
        t0 = datetime.datetime.now()
        pool = mp.Pool(mp.cpu_count())  # Use all cores on server
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
        volume_arr = pd.concat([i.get()["volume"] for i in df_list], axis=1).sort_index(ascending=True).values.astype(self.dtype)
        
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
            volume_arr,
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

    def create_advanced_features(self, df_vwap, df_high, df_low, df_amount, df_open, df_close, df_volume):
        """Advanced features including new ones for 0.41+"""
        features = {}
        
        # 1. Keep all V6 features (they work well)
        # Core returns
        for window in [1, 4, 24, 96, 168, 336, 672]:
            ret = (df_vwap / df_vwap.shift(window) - 1).astype(self.dtype)
            features[f'return_{window}'] = ret
            features[f'return_{window}_rank'] = ret.rank(axis=1, pct=True).astype(self.dtype)
        
        # Volatility features
        returns_15min = (df_vwap / df_vwap.shift(1) - 1).astype(self.dtype)
        for window in [24, 48, 96, 168, 336]:  # More windows
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            features[f'volatility_{window}'] = vol.astype(self.dtype)
            features[f'volatility_{window}_rank'] = vol.rank(axis=1, pct=True).astype(self.dtype)
        
        # Volatility ratios
        vol_24 = returns_15min.rolling(24, min_periods=12).std()
        vol_96 = returns_15min.rolling(96, min_periods=48).std()
        vol_336 = returns_15min.rolling(336, min_periods=168).std()
        features['volatility_ratio_24_96'] = (vol_24 / (vol_96 + 1e-8)).astype(self.dtype)
        features['volatility_ratio_96_336'] = (vol_96 / (vol_336 + 1e-8)).astype(self.dtype)
        
        # Volatility percentile
        for window in [96, 336]:
            vol = returns_15min.rolling(window, min_periods=window//2).std()
            vol_percentile = vol.rolling(window*4, min_periods=window*2).rank(pct=True)
            features[f'volatility_percentile_{window}'] = vol_percentile.astype(self.dtype)
        
        # 2. NEW: Advanced microstructure features
        # Amihud illiquidity
        for window in [24, 96]:
            illiquidity = (returns_15min.abs() / (df_volume * df_vwap + 1)).rolling(window).mean() * 1e9
            features[f'amihud_illiquidity_{window}'] = illiquidity.astype(self.dtype)
        
        # Kyle's lambda (price impact)
        for window in [96]:
            price_changes = df_vwap.diff()
            signed_volume = df_volume * np.sign(price_changes)
            cumulative_signed_volume = signed_volume.rolling(window).sum()
            cumulative_price_change = price_changes.rolling(window).sum()
            kyle_lambda = cumulative_price_change / (cumulative_signed_volume + 1)
            features[f'kyle_lambda_{window}'] = kyle_lambda.astype(self.dtype)
        
        # 3. NEW: Order book imbalance proxy
        # Using high/low/close to estimate order pressure
        bid_pressure = (df_close - df_low) / (df_high - df_low + 1e-8)
        ask_pressure = (df_high - df_close) / (df_high - df_low + 1e-8)
        order_imbalance = bid_pressure - ask_pressure
        
        for window in [24, 96]:
            features[f'order_imbalance_{window}'] = order_imbalance.rolling(window).mean().astype(self.dtype)
            features[f'order_imbalance_std_{window}'] = order_imbalance.rolling(window).std().astype(self.dtype)
        
        # 4. NEW: Realized variance measures
        for window in [96, 336]:
            # Realized variance
            rv = (returns_15min ** 2).rolling(window).sum()
            features[f'realized_variance_{window}'] = rv.astype(self.dtype)
            
            # Realized skewness
            rs = (returns_15min ** 3).rolling(window).sum() / (rv ** 1.5 + 1e-8)
            features[f'realized_skewness_{window}'] = rs.astype(self.dtype)
            
            # Realized kurtosis
            rk = (returns_15min ** 4).rolling(window).sum() / (rv ** 2 + 1e-8)
            features[f'realized_kurtosis_{window}'] = rk.astype(self.dtype)
        
        # 5. NEW: Intraday patterns
        # Time-weighted returns (emphasize recent)
        for window in [96, 336]:
            weights = np.exp(-np.arange(window) / (window / 4))[::-1]
            weights = weights / weights.sum()
            
            # Weighted return (simplified - proper implementation would use rolling apply)
            weighted_ret = df_vwap.rolling(window).mean()  # Placeholder
            features[f'time_weighted_return_{window}'] = (
                weighted_ret / df_vwap - 1
            ).astype(self.dtype)
        
        # 6. NEW: Cross-asset correlations and beta
        # Market return
        market_return = returns_15min.mean(axis=1)
        
        for window in [96, 336]:
            # Rolling beta
            cov_with_market = returns_15min.rolling(window).cov(market_return)
            market_var = market_return.rolling(window).var()
            beta = cov_with_market.div(market_var, axis=0)
            features[f'market_beta_{window}'] = beta.astype(self.dtype)
            
            # Idiosyncratic volatility
            systematic_return = beta.mul(market_return, axis=0)
            idio_return = returns_15min.sub(systematic_return, axis=0)
            idio_vol = idio_return.rolling(window).std()
            features[f'idiosyncratic_vol_{window}'] = idio_vol.astype(self.dtype)
        
        # 7. Volume features (enhanced)
        volume_usd = df_amount * df_vwap
        
        for window in [24, 96, 336]:
            vol_mean = volume_usd.rolling(window, min_periods=window//2).mean()
            features[f'volume_usd_{window}'] = vol_mean.astype(self.dtype)
            features[f'volume_usd_{window}_rank'] = vol_mean.rank(axis=1, pct=True).astype(self.dtype)
            
            # Volume volatility
            vol_std = volume_usd.rolling(window).std()
            features[f'volume_volatility_{window}'] = (vol_std / (vol_mean + 1e-8)).astype(self.dtype)
            
            # Volume concentration (Herfindahl)
            vol_share = volume_usd / volume_usd.sum(axis=1).values[:, None]
            vol_hhi = (vol_share ** 2).rolling(window).mean()
            features[f'volume_concentration_{window}'] = vol_hhi.astype(self.dtype)
        
        # 8. Price efficiency and trends
        for window in [48, 96, 336]:
            net_change = (df_vwap - df_vwap.shift(window)).abs()
            total_change = (df_vwap.diff().abs()).rolling(window).sum()
            features[f'efficiency_ratio_{window}'] = (
                net_change / (total_change + 1e-8)
            ).astype(self.dtype)
            
            # Trend strength
            ma = df_vwap.rolling(window).mean()
            trend_strength = (df_vwap - ma) / ma
            features[f'trend_strength_{window}'] = trend_strength.astype(self.dtype)
        
        # 9. All other features from V6
        # Price position
        for window in [168, 336]:
            high_roll = df_high.rolling(window, min_periods=window//2).max()
            low_roll = df_low.rolling(window, min_periods=window//2).min()
            features[f'price_position_{window}'] = (
                (df_vwap - low_roll) / (high_roll - low_roll + 1e-8)
            ).astype(self.dtype)
        
        # Momentum
        features['momentum_96'] = (df_vwap / df_vwap.shift(96) - 1).astype(self.dtype)
        
        for window in [96, 336]:
            positive_periods = (returns_15min > 0).rolling(window, min_periods=window//2).sum()
            features[f'momentum_consistency_{window}'] = (
                positive_periods / window
            ).astype(self.dtype)
        
        # RSI
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
        
        # Market microstructure
        overnight_gap = (df_open - df_close.shift(1)) / (df_close.shift(1) + 1e-8)
        features['overnight_gap_mean_96'] = overnight_gap.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        hl_spread = (df_high - df_low) / df_vwap
        features['hl_spread_96'] = hl_spread.rolling(96, min_periods=48).mean().astype(self.dtype)
        
        # Relative momentum
        returns_24h = (df_vwap / df_vwap.shift(96) - 1)
        market_return_24h = returns_24h.mean(axis=1)
        features['relative_momentum_24h'] = (
            returns_24h.sub(market_return_24h, axis=0)
        ).astype(self.dtype)
        
        return features

    def train_neural_ranker(self, X_train, y_train, X_val, y_val):
        """Train neural network for ranking"""
        print("Training neural network ranker on GPU...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.nn_model = NeuralNetworkRanker(
            input_dim, 
            hidden_sizes=[1024, 512, 256, 128],  # Larger network
            dropout=0.3
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            self.nn_model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.nn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.nn_model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.nn_model.state_dict(), './result/best_nn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step(avg_val_loss)
        
        # Load best model
        self.nn_model.load_state_dict(torch.load('./result/best_nn_model.pth'))
        
        return self.nn_model

    def train_mega_ensemble(self, X_train_scaled, y_train, feature_names):
        """Train large ensemble with GPU support"""
        print("Training mega ensemble (10 models)...")
        
        self.models = []
        self.model_weights = []
        
        # Try GPU params first
        use_gpu = False
        try:
            # Test if GPU version is available
            test_params = self.gpu_params.copy()
            test_train = lgb.Dataset(X_train_scaled[:100].values, label=y_train[:100])
            test_model = lgb.train(test_params, test_train, num_boost_round=1)
            use_gpu = True
            print("GPU acceleration enabled for LightGBM!")
        except:
            print("GPU not available for LightGBM, using CPU")
            use_gpu = False
        
        base_params = self.gpu_params if use_gpu else self.best_params
        
        # Train 10 diverse models
        model_configs = [
            # Core models with best params
            (base_params.copy(), 300, 0.15, "Base params seed 42"),
            (base_params.copy(), 300, 0.15, "Base params seed 123"),
            (base_params.copy(), 300, 0.15, "Base params seed 999"),
            
            # Lower learning rate models
            (dict(base_params, learning_rate=0.05), 400, 0.10, "Lower LR"),
            (dict(base_params, learning_rate=0.03), 500, 0.10, "Very low LR"),
            
            # Different objectives
            (dict(base_params, objective='huber', alpha=0.9, metric='mae'), 300, 0.10, "Huber loss"),
            (dict(base_params, objective='regression_l1', metric='mae'), 300, 0.05, "MAE loss"),
            
            # Different tree structures
            (dict(base_params, num_leaves=320, max_depth=14), 250, 0.10, "Deeper trees"),
            (dict(base_params, num_leaves=128, min_data_in_leaf=200), 350, 0.05, "Simpler trees"),
            
            # Different regularization
            (dict(base_params, lambda_l1=1.0, lambda_l2=0.5), 300, 0.05, "Strong regularization"),
        ]
        
        lgb_train = lgb.Dataset(X_train_scaled.values, label=y_train)
        
        for i, (params, n_rounds, weight, desc) in enumerate(model_configs):
            print(f"Training model {i+1}/10: {desc}")
            
            # Add different seeds
            params['seed'] = 42 * (i + 1)
            params['bagging_seed'] = 42 * (i + 1) + 1
            params['feature_fraction_seed'] = 42 * (i + 1) + 2
            
            model = lgb.train(params, lgb_train, num_boost_round=n_rounds)
            self.models.append(model)
            self.model_weights.append(weight)
        
        # Normalize weights
        self.model_weights = np.array(self.model_weights)
        self.model_weights /= self.model_weights.sum()
        
        print(f"Ensemble weights: {self.model_weights}")
        
        return self.models

    def predict_ensemble(self, X_scaled, use_nn=True):
        """Make predictions using mega ensemble + neural network"""
        predictions = []
        
        # LightGBM predictions
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred * weight)
        
        lgb_pred = np.sum(predictions, axis=0)
        
        # Neural network predictions
        if use_nn and self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                nn_pred = self.nn_model(X_tensor).cpu().numpy().squeeze()
            
            # Blend predictions (80% LightGBM, 20% NN)
            final_pred = 0.8 * lgb_pred + 0.2 * nn_pred
        else:
            final_pred = lgb_pred
        
        return final_pred

    def advanced_post_processing(self, predictions, method='multi_stage'):
        """Advanced post-processing for ranking optimization"""
        
        if method == 'multi_stage':
            # Stage 1: Remove extreme outliers
            predictions = np.clip(predictions, 
                                np.percentile(predictions, 0.1), 
                                np.percentile(predictions, 99.9))
            
            # Stage 2: Apply isotonic regression for monotonicity
            from sklearn.isotonic import IsotonicRegression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            
            # Fit on ranks
            ranks = stats.rankdata(predictions)
            iso_reg.fit(predictions, ranks)
            predictions = iso_reg.transform(predictions)
            
            # Stage 3: Transform to target distribution
            # Use power transform to match typical return distribution
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            predictions = pt.fit_transform(predictions.reshape(-1, 1)).ravel()
            
            # Stage 4: Final scaling
            predictions = predictions * 0.018  # 1.8% typical volatility
            
        return predictions

    def train_efficient(self, df_target, df_factor1, df_factor2, df_factor3, 
                       additional_features, all_symbol_list):
        """Enhanced training with GPU support"""
        print("Starting GPU-enhanced training...")
        
        # Prepare features
        all_features = additional_features.copy()
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2
        all_features['volume_sum_7d'] = df_factor3
        
        # Use more data on server
        print("Preparing training data (using 50% for training)...")
        train_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.5)]
        
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
        
        # Split for neural network validation
        n_samples = len(train_df)
        train_size = int(0.8 * n_samples)
        
        train_df_sorted = train_df.sort_index()
        train_split = train_df_sorted.iloc[:train_size]
        val_split = train_df_sorted.iloc[train_size:]
        
        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
        X_train_split = train_split[feature_names]
        X_val_split = val_split[feature_names]
        
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        
        y_train_split = train_split['target'].values.astype(self.dtype)
        y_val_split = val_split['target'].values.astype(self.dtype)
        
        # Train neural network
        if CUDA_AVAILABLE:
            self.train_neural_ranker(X_train_scaled, y_train_split, X_val_scaled, y_val_split)
        
        # Train mega ensemble on full training data
        X_train_full = train_df[feature_names]
        X_train_full_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_full),
            index=X_train_full.index,
            columns=X_train_full.columns
        ).astype(self.dtype)
        
        y_train_full = train_df['target'].values.astype(self.dtype)
        
        self.train_mega_ensemble(X_train_full_scaled, y_train_full, feature_names)
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        # Process all data
        print("Making predictions on all data...")
        chunk_size = 100  # Larger chunks on server
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
            X_chunk_scaled = self.scaler.transform(X_chunk)
            
            # Make ensemble predictions
            predictions = self.predict_ensemble(X_chunk_scaled, use_nn=CUDA_AVAILABLE)
            
            # Advanced post-processing
            predictions = self.advanced_post_processing(predictions)
            
            # Store results
            chunk_results = chunk_df[['target']].copy()
            chunk_results['y_pred'] = predictions
            results_list.append(chunk_results)
        
        # Combine all results
        results_df = pd.concat(results_list)
        
        return results_df

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        """Main training function"""
        print("Getting additional data for features...")
        
        # Get data including volume
        (
            all_symbol_list,
            time_arr,
            open_price_arr,
            high_price_arr,
            low_price_arr,
            close_price_arr,
            vwap_arr,
            amount_arr,
            volume_arr,
        ) = self.get_all_symbol_kline()
        
        # Create DataFrames
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_open = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        df_volume = pd.DataFrame(volume_arr, columns=all_symbol_list, index=time_arr, dtype=self.dtype)
        
        # Free memory
        del vwap_arr, amount_arr, high_price_arr, low_price_arr, open_price_arr, close_price_arr, volume_arr
        gc.collect()
        
        # Create advanced features
        print("Creating advanced features...")
        additional_features = self.create_advanced_features(
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
        
        # Feature importance from first model
        if hasattr(self, 'models') and len(self.models) > 0:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[0].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            print("\nTop 20 most important features:")
            print(importance.head(20))

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
        df_submit_competition.to_csv("./result/submit_v7_gpu.csv", index=False)

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
            volume_arr,
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
        del open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr, volume_arr
        gc.collect()
        
        # Train
        self.train(
            df_24hour_rtn.shift(-windows_1d),
            df_7d_volatility,
            df_7d_momentum,
            df_amount_sum,
        )


if __name__ == "__main__":
    print("="*80)
    print("GPU-ENHANCED MODEL V7 - ADVANCED STRATEGIES FOR 0.41+")
    print("="*80)
    print("\nEnhancements in this version:")
    print("1. GPU acceleration (if available)")
    print("2. Neural network ranker for complex patterns")
    print("3. 10-model mega ensemble")
    print("4. Advanced microstructure features")
    print("5. Multi-stage post-processing")
    print("6. 50% training data (server advantage)")
    print("\n" + "="*80 + "\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    model = GPUEnhancedModelV7()
    model.run()