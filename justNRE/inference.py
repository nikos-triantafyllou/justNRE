import numpy as np
import pandas as pd


def log_ratio_vectorized(obs_norm, theta_array, nre_ensemble, normalize_fn):
    """
    Vectorized version of log_ratio for multiple theta samples.

    Parameters:
    - obs_norm: shape (d_obs,) - observation (raw, not yet normalized)
    - theta_array: shape (N, d_theta) - parameter samples (raw, not yet normalized)
    - nre_ensemble: list of trained NN models, each outputs P(joint) probability
    - normalize_fn: the normalize() closure returned by nre.data.normalize,
      fit on the same training stats used to train nre_ensemble

    Returns:
    - log_ratios: shape (N,) - log-ratios log(p_joint / p_marg)
    """
    N = theta_array.shape[0]
    # Repeat obs_norm N times and concatenate with theta_array
    obs_tile = np.tile(obs_norm, (N, 1))  # shape (N, d_obs)
    input_array = np.hstack((obs_tile, theta_array))  # shape (N, d_obs + d_theta)

    input_array = normalize_fn(input_array)

    df = pd.DataFrame(input_array)

    y_pred = np.zeros(N)
    for model in nre_ensemble:
        y_pred += model.predict(df, verbose=0).flatten()  # shape (N,)

    y_pred /= len(nre_ensemble)

    # Avoid division by zero or log(0)
    eps = 1e-16
    y_pred = np.clip(y_pred, eps, 1 - eps)
    logr = np.log(y_pred) - np.log((1 - y_pred))
    return logr


def run_grid_search(param_names, loglike_vector, prior_transform, n_points_per_dim=20, vectorized_prior=True):
    num_dims = len(param_names)

    # 1. Create the grid spaced evenly BETWEEN 0 and 1 (excluding exact 0.0 and 1.0)
    # This avoids the index out-of-bounds error at 1.0
    axis_grids = [np.linspace(0.0, 1.0, n_points_per_dim + 2)[1:-1] for _ in range(num_dims)]
    mesh = np.meshgrid(*axis_grids, indexing='ij')
    u_samples = np.vstack([m.ravel() for m in mesh]).T

    # 2. Transform the unit grid to the physical parameter space
    if vectorized_prior:
        param_samples = prior_transform(u_samples)
    else:
        param_samples = np.array([prior_transform(u) for u in u_samples])

    # 3. Evaluate the vectorized log-likelihood
    loglikes = loglike_vector(param_samples)

    # 4. Find the maximum likelihood point
    best_idx = np.argmax(loglikes)
    best_fit_point = param_samples[best_idx]

    # 5. Package results
    result = {
        'maximum_likelihood': {
            'point': best_fit_point.tolist(),
            'loglike': loglikes[best_idx]
        },
        'weighted_samples': {
            'points': param_samples.tolist(),
            'loglikelihoods': loglikes.tolist()
        }
    }

    return result
