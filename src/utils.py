"""Utility functions: paths, config loading, seed setting, helpers."""

import os
import random
from pathlib import Path

import numpy as np
import yaml

# -- Project Paths ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_config(config_name: str = "model_configs.yaml") -> dict:
    """Load a YAML configuration file from configs/."""
    config_path = CONFIGS_DIR / config_name
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dirs():
    """Ensure all project directories exist."""
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, DOCS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# -- Figure Defaults --------------------------------------------------------
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Publication-quality matplotlib defaults
PLOT_STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
}


def setup_plotting():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_STYLE)
    # Color-blind friendly palette for accessibilty
    import seaborn as sns
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")


# -- GPU Device Detection ---------------------------------------------------

def detect_lgb_device() -> str:
    """Detect the best available LightGBM device.

    Returns 'cuda' if the CUDA backend is compiled in and a GPU is present,
    otherwise 'cpu'.  The OpenCL ('gpu') backend is intentionally skipped
    because it is slower than CPU on modern hardware and segfaults on exit.
    """
    try:
        import lightgbm as lgb
        _ds = lgb.Dataset([[0]], [0])
        lgb.train({"device": "cuda", "num_iterations": 1, "verbose": -1},
                   _ds, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def detect_xgb_device() -> str:
    """Return 'cuda' if XGBoost can use CUDA, else 'cpu'."""
    try:
        from xgboost import XGBRegressor
        import numpy as _np
        _np.random.seed(0)
        m = XGBRegressor(n_estimators=1, tree_method="hist", device="cuda",
                         verbosity=0)
        m.fit(_np.random.rand(10, 2), _np.random.rand(10))
        return "cuda"
    except Exception:
        return "cpu"


def detect_catboost_device() -> str:
    """Return 'GPU' if CatBoost can use the GPU, else 'CPU'."""
    try:
        from catboost import CatBoostRegressor
        import numpy as _np
        _np.random.seed(0)
        m = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        m.fit(_np.random.rand(10, 2), _np.random.rand(10))
        return "GPU"
    except Exception:
        return "CPU"


_DEVICE_CACHE: dict = {}


def get_device(framework: str) -> str:
    """Cached device detection for 'lgb', 'xgb', or 'catboost'.

    Call once at script startup and reuse the result.  Results are cached
    so repeated calls are free.
    """
    if framework not in _DEVICE_CACHE:
        if framework == "lgb":
            _DEVICE_CACHE[framework] = detect_lgb_device()
        elif framework == "xgb":
            _DEVICE_CACHE[framework] = detect_xgb_device()
        elif framework == "catboost":
            _DEVICE_CACHE[framework] = detect_catboost_device()
        else:
            raise ValueError(f"Unknown framework: {framework}")
    return _DEVICE_CACHE[framework]
