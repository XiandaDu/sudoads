# optuna_optimization.py
# Run this file to optimize your model parameters

import optuna
import numpy as np
import pandas as pd
from optimized_lightgbm_v4 import MemoryOptimizedModel
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import gc


def run_optuna_optimization(n_trials=30):
    """Run Optuna optimization on your model"""
    
    # Initialize model to get data
    base_model = MemoryOptimizedModel()
    
    print("Loading data for optimization...")
    # Get data using your existing method
    (
        all_symbol_list,
        time_arr,
        open_price_arr,
        high_price_arr,
        low_price_arr,
        close_price_arr,
        vwap_arr,
        amount_arr,
    ) = base_model.get_all_symbol_kline()
    
    # Create DataFrames
    df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr, dtype=np.float32)
    df_high = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr, dtype=np.float32)
    df_low = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr, dtype=np.float32)
    df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr, dtype=np.float32)
    
    # Calculate factors
    windows_1d = 4 * 24 * 1
    windows_7d = 4 * 24 * 7
    
    df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).astype(np.float32)
    df_15min_rtn = (df_vwap / df_vwap.shift(1) - 1).astype(np.float32)
    df_7d_volatility = df_15min_rtn.rolling(windows_7d, min_periods=windows_7d//2).std(ddof=1).astype(np.float32)
    df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).astype(np.float32)
    df_amount_sum = df_amount.rolling(windows_7d, min_periods=windows_7d//2).sum().astype(np.float32)
    
    # Create features
    print("Creating features...")
    additional_features = base_model.create_critical_features(df_vwap, df_high, df_low, df_amount)
    
    # Prepare all features
    all_features = additional_features.copy()
    all_features['volatility_7d'] = df_7d_volatility
    all_features['momentum_7d'] = df_7d_momentum
    all_features['volume_sum_7d'] = df_amount_sum
    
    # Target
    df_target = df_24hour_rtn.shift(-windows_1d)
    
    # Free memory
    del open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
    gc.collect()
    
    # Use subset of data for faster optimization (15% of symbols)
    opt_symbols = all_symbol_list[:int(len(all_symbol_list) * 0.15)]
    
    print(f"Using {len(opt_symbols)} symbols for optimization")
    
    # Prepare data
    feature_dfs = []
    feature_names = []
    
    for feat_name, feat_data in all_features.items():
        if isinstance(feat_data, pd.DataFrame):
            opt_data = feat_data[opt_symbols].stack()
            opt_data.name = feat_name
            feature_dfs.append(opt_data)
            feature_names.append(feat_name)
    
    target_opt = df_target[opt_symbols].stack()
    target_opt.name = 'target'
    
    opt_df = pd.concat(feature_dfs + [target_opt], axis=1)
    opt_df = opt_df.dropna()
    
    # Split data
    n_samples = len(opt_df)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    opt_df_sorted = opt_df.sort_index()
    train_df = opt_df_sorted.iloc[:train_size]
    val_df = opt_df_sorted.iloc[train_size:train_size+val_size]
    
    # Scale features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[feature_names])
    X_val = scaler.transform(val_df[feature_names])
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Define objective function
    def objective(trial):
        params = {
            'objective': trial.suggest_categorical('objective', ['regression', 'regression_l1', 'huber']),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'verbose': -1,
            'num_threads': 4,
            'force_row_wise': True,
            'metric': 'rmse'
        }
        
        if params['objective'] == 'huber':
            params['alpha'] = trial.suggest_float('alpha', 0.7, 0.95)
            params['metric'] = 'mae'
        
        # Train model
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )
        
        # Predict
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calculate weighted spearman
        val_results = pd.DataFrame({
            'true': y_val,
            'pred': y_pred
        }, index=val_df.index)
        
        # Calculate by timestamp
        correlations = []
        for timestamp in val_results.index.get_level_values(0).unique()[:100]:  # Sample timestamps
            ts_data = val_results.loc[timestamp]
            if len(ts_data) > 5:
                corr = base_model.weighted_spearmanr(ts_data['true'].values, ts_data['pred'].values)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    # Create and run study
    print("\nStarting Optuna optimization...")
    print("This will take some time. Progress will be shown below.\n")
    
    study = optuna.create_study(direction='maximize')
    
    # Add callback to show progress
    def callback(study, trial):
        print(f"Trial {trial.number} finished with value: {trial.value:.4f}")
        if trial.number % 10 == 0:
            print(f"  Best so far: {study.best_value:.4f}")
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE!")
    print("="*50)
    print(f"\nBest Weighted Spearman: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    import json
    with open('./result/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print("\nBest parameters saved to ./result/best_params.json")
    
    # Save full study
    import pickle
    with open('./result/optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    return study.best_params


def apply_best_params():
    """Apply the best parameters found by Optuna to your model"""
    import json
    
    # Load best parameters
    with open('./result/best_params.json', 'r') as f:
        best_params = json.load(f)
    
    print("Loaded best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Modify your model's train_efficient method to use these parameters
    # You'll need to update the params dict in train_efficient
    
    return best_params


if __name__ == "__main__":
    # Step 1: Install Optuna if not already installed
    # pip install optuna
    
    # Step 2: Run optimization
    best_params = run_optuna_optimization(n_trials=200)  # Start with 30, increase to 100+ for better results
    
    # Step 3: The best parameters are now saved
    # To use them, modify your train_efficient method in optimized_lightgbm_v2.py
    # Replace the params dict with the optimized parameters