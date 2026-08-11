from .data import prepare_for_NRE, split, normalize
from .training import build_smooth_mlp, train_nre_ensemble, ensemble_predict
from .inference import log_ratio_vectorized, run_grid_search
from .diagnostics import plot_confusion_matrix, cornerplot1
from .mock import generate_mock_raw_data

__version__ = "0.1.0"

__all__ = [
    "prepare_for_NRE",
    "split",
    "normalize",
    "build_smooth_mlp",
    "train_nre_ensemble",
    "ensemble_predict",
    "log_ratio_vectorized",
    "run_grid_search",
    "plot_confusion_matrix",
    "cornerplot1",
    "generate_mock_raw_data",
]
