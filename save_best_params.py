# save_best_params.py
# Save the parameters that achieved 0.4074

import json

# These are the parameters that achieved 0.4074
best_params = {
    "objective": "regression",
    "num_leaves": 163,
    "learning_rate": 0.07182861492338714,
    "feature_fraction": 0.8184386555861791,
    "bagging_fraction": 0.6985394411684631,
    "bagging_freq": 2,
    "lambda_l1": 0.7113790560603763,
    "lambda_l2": 0.09201149241276538,
    "min_data_in_leaf": 139,
    "max_depth": 10
}

# Save to a specific file
with open('./result/best_params_locked.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("Best parameters (0.4074 score) saved to ./result/best_params_locked.json")

# Also create a backup
with open('./result/best_params_0.4074_backup.json', 'w') as f:
    json.dump(best_params, f, indent=2)
    
print("Backup saved to ./result/best_params_0.4074_backup.json")