import numpy as np


def generate_mock_raw_data(num_simulations=1000, num_features=5, seed=42):
    rng = np.random.default_rng(seed)
    params_arr = rng.uniform(0.05, 0.45, size=(num_simulations, 1))
    data_arr = (params_arr * 4.0) + rng.normal(0, 0.05, size=(num_simulations, num_features))
    return data_arr, params_arr
