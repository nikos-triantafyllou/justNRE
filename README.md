   <div align="center">
     <img src="images/justNRE.png">
   </div>

An Occam's razor inspired Neural Ratio Estimation code for simulation-based inference
======================================================================================


A lightweight package for Neural Ratio Estimation (NRE), a simulation-based
inference (SBI) technique: train a binary classifier to distinguish samples
drawn from the joint distribution `p(x, theta)` versus the product of
marginals `p(x)p(theta)`, then use the classifier's output as an estimate of
the likelihood-to-evidence ratio, `p(x|theta) / p(x)`, for downstream
inference (e.g. nested sampling).

## Installation

From the project root (same folder as `pyproject.toml`):

```bash
pip install -e .
```

If you're on a machine without outbound internet access (e.g. an offline
compute cluster), and `setuptools`/`wheel` are already available locally:

```bash
pip install -e . --no-build-isolation
```

or just simply 
```bash
pip install git+https://github.com/nikos-triantafyllou/justNRE.git
```

## Quick start

```python
import justNRE

# 1. Get data + parameters, shape (n_sims, n_observables) and (n_sims, n_params)
data_arr, params_arr = justNRE.generate_mock_raw_data()

# 2. Build the joint-vs-marginal classification dataset
instances, targets = justNRE.prepare_for_NRE(data_arr, params_arr)

# 3. Split and normalize
(X_train, y_train), (X_val, y_val), (X_test, y_test) = justNRE.split(
    instances, targets, [80, 10, 10]
)
(X_train_norm, X_val_norm, X_test_norm), denormalize, normalize = justNRE.normalize(
    (X_train, X_val, X_test), norm_type="standard"
)

# 4. Train an ensemble of classifiers
ensemble, loss_figs = justNRE.train_nre_ensemble(
    X_train_norm, y_train, X_val_norm, y_val,
    num_models=5, epochs=150, batch_size=128, patience=40,
)

# 5. Evaluate
from sklearn.metrics import roc_auc_score
y_proba = justNRE.ensemble_predict(ensemble, X_test, normalize_fn=normalize)
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 6. Use the trained ensemble as a log-ratio estimator for inference
#    (e.g. with ultranest -- see tutorial_notebook.ipynb for a full example)
def loglike_vector(theta_samples):
    return justNRE.log_ratio_vectorized(obs_true, theta_samples, ensemble, normalize)
```

See `tutorial_notebook.ipynb` for the complete, runnable end-to-end example
(mock data -> training -> diagnostics -> nested-sampling inference -> corner
plot).

## Package layout

```
justNRE/
├── data.py           # prepare_for_NRE, split, normalize
├── training.py       # build_smooth_mlp, train_nre_ensemble, ensemble_predict
├── inference.py      # log_ratio_vectorized, run_grid_search
├── diagnostics.py    # plot_confusion_matrix, cornerplot1
└── mock.py           # generate_mock_raw_data (for the tutorial only)
```

- **`data.py`** builds the balanced joint/marginal training set from raw
  `(data, params)` pairs, splits it into train/val/test, and normalizes it
  using statistics fit only on the training split.
- **`training.py`** defines the MLP architecture and the ensemble training
  loop. Hyperparameters (dropout, learning rate, batch size, epochs,
  patience, etc.) are passed as explicit function arguments rather than
  module-level globals.
- **`inference.py`** turns ensemble predictions into a log-likelihood-ratio
  function suitable for a sampler like `ultranest`, plus a simple grid-search
  utility.
- **`diagnostics.py`** has plotting helpers (confusion matrix, corner plot)
  that build and return `matplotlib` `Figure` objects rather than drawing on
  implicit global state, so the caller decides whether to show/save them.

## A note on the prior

The classifier only ever sees joint examples with `theta` in the range of
your training simulations. If you evaluate the log-ratio estimator outside
that range (e.g. an overly wide sampler prior), you're extrapolating and the
ratio estimate is not reliable. Set your `prior_transform` bounds to match
the actual range your simulator was run over — the tutorial notebook does
this by deriving `l_bounds`/`u_bounds` directly from `params_arr`.

## Requirements

See `pyproject.toml`. Notably: `numpy`, `pandas`, `tensorflow`,
`scikit-learn`, `matplotlib`, `seaborn`, `corner`, `ultranest`.
