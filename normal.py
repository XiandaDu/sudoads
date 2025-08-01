import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
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
        # Use more CPU cores for faster processing
        n_cores = max(1, mp.cpu_count() - 1)
        pool = mp.Pool(n_cores)
        
        all_symbol_list = self.get_all_symbol_list()
        print(f"Loading {len(all_symbol_list)} symbols using {n_cores} cores...")
        
        # Submit all tasks at once
        df_list = []
        for symbol in all_symbol_list:
            df_list.append(pool.apply_async(self.get_single_symbol_kline_data, (symbol,)))
            
        pool.close()
        pool.join()
        # Collect results more efficiently - only get what we need
        print("Collecting results...")
        results = [task.get() for task in df_list]
        
        # Filter out empty DataFrames
        valid_results = [(symbol, df) for symbol, df in zip(all_symbol_list, results) if not df.empty]
        valid_symbols = [symbol for symbol, _ in valid_results]
        valid_dfs = [df for _, df in valid_results]
        
        print(f"Successfully loaded {len(valid_results)} out of {len(all_symbol_list)} symbols")
        
        # Concatenate more efficiently - create all arrays at once
        price_data = {}
        for col in ["open_price", "high_price", "low_price", "close_price", "vwap", "amount"]:
            price_data[col] = pd.concat([df[col] for df in valid_dfs], axis=1, keys=valid_symbols).sort_index()
            
        # Get time array from any price series
        time_arr = pd.to_datetime(pd.Series(price_data["vwap"].index), unit="ms").values
        
        # Convert to numpy arrays
        arrays = {}
        for col in price_data.keys():
            arrays[f"{col}_arr"] = price_data[col].values.astype(float)
            
        all_symbol_list = valid_symbols
        elapsed = datetime.datetime.now() - t0
        print(f"Finished loading data in {elapsed}")
        
        return (
            all_symbol_list,
            time_arr,
            arrays["open_price_arr"],
            arrays["high_price_arr"],
            arrays["low_price_arr"],
            arrays["close_price_arr"],
            arrays["vwap_arr"],
            arrays["amount_arr"],
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

    def create_advanced_features(self, df_vwap, df_amount):
        """Create optimized features with proper alignment"""
        print("Creating advanced features...")
        
        # Calculate returns first
        returns = df_vwap.pct_change()
        log_returns = np.log(df_vwap / df_vwap.shift(1))
        
        # Define time windows (in 15-minute intervals)
        windows = {
            'short': [4, 8, 16, 24],  # 1h, 2h, 4h, 6h
            'medium': [96, 288, 672],  # 1d, 3d, 7d  
            'long': [1344, 2880]  # 14d, 30d
        }
        
        features_dict = {}
        
        # Price momentum features (multi-timeframe)
        for window in windows['medium']:
            features_dict[f'momentum_{window}'] = df_vwap.pct_change(window)
            features_dict[f'log_momentum_{window}'] = np.log(df_vwap / df_vwap.shift(window))
            
        # Volatility features (realized volatility)
        for window in windows['medium']:
            features_dict[f'volatility_{window}'] = returns.rolling(window, min_periods=int(window*0.5)).std() * np.sqrt(window)
            features_dict[f'log_volatility_{window}'] = log_returns.rolling(window, min_periods=int(window*0.5)).std() * np.sqrt(window)
            
        # Volume features 
        for window in windows['medium']:
            features_dict[f'volume_mean_{window}'] = df_amount.rolling(window, min_periods=int(window*0.5)).mean()
            features_dict[f'volume_std_{window}'] = df_amount.rolling(window, min_periods=int(window*0.5)).std()
            features_dict[f'volume_trend_{window}'] = df_amount.rolling(window, min_periods=int(window*0.5)).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
            
        # Price position in range (mean reversion signals)
        for window in windows['medium']:
            rolling_min = df_vwap.rolling(window, min_periods=int(window*0.5)).min()
            rolling_max = df_vwap.rolling(window, min_periods=int(window*0.5)).max()
            features_dict[f'price_position_{window}'] = (df_vwap - rolling_min) / (rolling_max - rolling_min + 1e-8)
            
        # Technical indicators
        for window in [96, 672]:  # 1d, 7d
            # RSI-like momentum indicator
            delta = returns
            gain = delta.where(delta > 0, 0).rolling(window, min_periods=int(window*0.5)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=int(window*0.5)).mean()
            features_dict[f'rsi_{window}'] = gain / (gain + loss + 1e-8)
            
        # Short-term mean reversion
        for window in windows['short']:
            features_dict[f'short_mom_{window}'] = df_vwap.pct_change(window)
            
        # Volume-price relationship
        for window in windows['medium']:
            price_change = df_vwap.pct_change()
            vol_change = df_amount.pct_change()
            corr_window = min(window, 96)  # Cap correlation window
            features_dict[f'price_volume_corr_{window}'] = price_change.rolling(corr_window, min_periods=int(corr_window*0.5)).corr(vol_change)
            
        print(f"Created {len(features_dict)} features")
        return features_dict

    def train(self, df_target, df_factor1, df_factor2, df_factor3, df_vwap=None, df_amount=None):
        """Optimized training with proper data alignment and cross-sectional processing"""
        print("Starting optimized training...")
        
        # Create feature dictionary with basic factors first
        all_features = {}
        
        # Basic factors - ensure proper alignment
        all_features['volatility_7d'] = df_factor1
        all_features['momentum_7d'] = df_factor2 
        all_features['volume_7d'] = df_factor3
        
        # Add advanced features if provided
        if df_vwap is not None and df_amount is not None:
            advanced_features = self.create_advanced_features(df_vwap, df_amount)
            all_features.update(advanced_features)
            
        # Stack all features at once to maintain alignment
        print("Stacking features...")
        feature_data = {}
        
        for name, df in all_features.items():
            stacked = df.stack()
            stacked.name = name
            feature_data[name] = stacked
            
        # Stack target
        target_stacked = df_target.stack()
        target_stacked.name = 'target'
        
        # Combine all data
        print("Combining data...")
        data = pd.concat(list(feature_data.values()) + [target_stacked], axis=1)
        
        print(f"Data shape before cleaning: {data.shape}")
        
        # Clean data more efficiently
        data = data.replace([np.inf, -np.inf], np.nan)
        initial_rows = len(data)
        data = data.dropna()
        print(f"Dropped {initial_rows - len(data)} rows with NaN/inf values")
        
        if len(data) < 1000:
            print(f"Warning: Only {len(data)} valid samples")
            if len(data) == 0:
                return None
                
        feature_cols = [col for col in data.columns if col != 'target']
        print(f"Training with {len(feature_cols)} features and {len(data)} samples")
        
        # Cross-sectional normalization - more efficient
        print("Applying cross-sectional ranking...")
        
        # Group by timestamp and rank within each group
        def rank_cross_section(group):
            # Rank each feature within the cross-section
            for col in feature_cols:
                if len(group[col].unique()) > 1:  # Only rank if there's variation
                    group[col] = group[col].rank(pct=True, method='average')
                else:
                    group[col] = 0.5  # Neutral rank if no variation
            return group
            
        # Apply ranking by timestamp
        data_ranked = data.groupby(level=0, group_keys=False).apply(rank_cross_section)
        
        # Feature selection - remove low-variance features
        feature_variance = data_ranked[feature_cols].var()
        valid_features = feature_variance[feature_variance > 0.001].index.tolist()
        print(f"Selected {len(valid_features)} features with sufficient variance")
        
        if len(valid_features) == 0:
            print("No features with sufficient variance!")
            return None
            
        X = data_ranked[valid_features]
        y = data_ranked['target']
        
        # Model ensemble with better hyperparameters
        
        models = {
            'ridge': Ridge(alpha=0.1, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42, max_iter=1000),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        }
        
        best_model = None
        best_score = -2
        best_name = None
        
        for name, model in models.items():
            try:
                model.fit(X, y)
                y_pred = model.predict(X)
                score = self.weighted_spearmanr(y, y_pred)
                print(f"{name}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"Error with {name}: {e}")
                
        if best_model is None:
            print("Fallback to LinearRegression")
            best_model = LinearRegression()
            best_model.fit(X, y)
            best_name = "linear"
            best_score = self.weighted_spearmanr(y, best_model.predict(X))
            
        print(f"Best model: {best_name} (score: {best_score:.4f})")
        
        # Generate predictions
        predictions = best_model.predict(X)
        
        # Create submission DataFrame more efficiently
        submission_data = data_ranked.reset_index()
        submission_data['predict_return'] = predictions
        submission_data = submission_data[['level_0', 'level_1', 'predict_return']]
        submission_data.columns = ['datetime', 'symbol', 'predict_return']
        submission_data = submission_data[submission_data['datetime'] >= self.start_datetime]
        submission_data['id'] = submission_data['datetime'].astype(str) + '_' + submission_data['symbol']
        
        final_submission = submission_data[['id', 'predict_return']]
        print(f"Generated {len(final_submission)} predictions")
        
        # Save results
        self._save_submission(final_submission)
        
        print(f"Final score: {best_score:.4f}")
        return best_model
        
    def _save_submission(self, df_submit):
        """Save submission file with proper handling of missing IDs"""
        # Handle submission file path  
        submission_id_path = "./result/submission_id.csv"
        alt_paths = ["./submission_id.csv", "../submission_id.csv", "./avenir-hku-web/submission_id.csv"]
        
        for path in [submission_id_path] + alt_paths:
            if os.path.exists(path):
                submission_id_path = path
                break
                
        if os.path.exists(submission_id_path):
            df_submission_id = pd.read_csv(submission_id_path)
            id_list = df_submission_id["id"].tolist()
            df_submit_competition = df_submit[df_submit["id"].isin(id_list)]
            missing_elements = list(set(id_list) - set(df_submit_competition["id"]))
            
            if missing_elements:
                print(f"Filling {len(missing_elements)} missing IDs with 0")
                new_rows = pd.DataFrame(
                    {"id": missing_elements, "predict_return": [0] * len(missing_elements)}
                )
                df_submit_competition = pd.concat([df_submit_competition, new_rows], ignore_index=True)
            
            os.makedirs("./result", exist_ok=True)
            df_submit_competition.to_csv("./result/submit.csv", index=False)
            print(f"Submission saved with {len(df_submit_competition)} predictions")
        else:
            print(f"Warning: submission_id.csv not found")
            os.makedirs("./result", exist_ok=True)
            df_submit.to_csv("./result/submit.csv", index=False)

    def run(self):
        # call the get_all_symbol_kline function to get the K-line data and event data for all currencies
        (
            all_symbol_list,
            time_arr,
            _,  # open_price_arr - not used
            _,  # high_price_arr - not used
            _,  # low_price_arr - not used
            _,  # close_price_arr - not used
            vwap_arr,
            amount_arr,
        ) = self.get_all_symbol_kline()
        
        # convert arrays into DataFrames
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        
        # Volume calculation removed - we'll use amount directly for efficiency
        
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        
        # Calculate basic factors (same as before)
        df_24hour_rtn = df_vwap / df_vwap.shift(windows_1d) - 1
        df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
        df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
        df_7d_momentum = df_vwap / df_vwap.shift(windows_7d) - 1
        df_amount_sum = df_amount.rolling(windows_7d).sum()
        
        # Enhanced training with optimized feature creation
        print("Starting enhanced model training...")
        self.train(
            df_24hour_rtn.shift(-windows_1d),  # target: next 24h return
            df_7d_volatility,                  # factor 1: volatility
            df_7d_momentum,                    # factor 2: momentum  
            df_amount_sum,                     # factor 3: volume
            df_vwap,                          # raw price data for advanced features
            df_amount                         # raw amount data for advanced features
        )


if __name__ == "__main__":
    model = OlsModel()
    model.run()