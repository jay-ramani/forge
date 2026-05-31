# Fused Observation & Remote-Sensing Ground Enhancement (FORGE) 🛰️🌦️

> **Satellite data based weather prediction augmented with terrestrial sensor data**

FORGE is a Python machine learning framework for weather forecasting that fuses multi-source observational data — primarily satellite-derived imagery and measurements — with ground-level terrestrial sensor readings. By combining the macro-scale spatial coverage of orbital platforms with the high-frequency, point-accurate readings of surface sensors, Forge produces more accurate and localised weather predictions than either source alone could achieve.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Configuration](#configuration)
  - [Config File Format](#config-file-format)
  - [Pipeline](#pipeline)
  - [Unions (Feature Fusion)](#unions-feature-fusion)
  - [Hyperparameter Search](#hyperparameter-search)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Testing a Model](#testing-a-model)
  - [CLI Overrides](#cli-overrides)
- [Customisation](#customisation)
  - [Custom Data Loader](#custom-data-loader)
  - [Custom Model](#custom-model)
  - [Custom Optimizer](#custom-optimizer)
- [Data Sources](#data-sources)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Weather prediction at a fine spatial and temporal resolution is a hard problem. Satellite platforms provide excellent broad-area coverage but are limited in temporal cadence and struggle with near-surface boundary-layer phenomena. Terrestrial sensors — weather stations, IoT motes, MEMS pressure/humidity/temperature arrays — offer dense temporal sampling but sparse geographic coverage.

Forge addresses this complementarity through a **sensor-fusion** approach: satellite data (e.g. GOES, MSG, Sentinel, or custom raster extracts) and terrestrial sensor time-series are ingested via pluggable data loaders, transformed through a configurable scikit-learn–compatible pipeline, and optimised with automated hyperparameter search (grid, random, or Bayesian). The result is a trained predictive model that leverages the best of both worlds.

---

## Key Features

- **Dual-source data ingestion** — dedicated loaders for satellite raster/spectral data and terrestrial sensor tabular time-series.
- **Sensor fusion pipelines** — `FeatureUnion`-style merging of heterogeneous feature streams within a single, declarative pipeline.
- **JSON-driven configuration** — all data paths, pipeline steps, hyperparameter grids, cross-validation strategies, and scoring metrics are expressed in a single `.json` file with no code changes required.
- **Automated hyperparameter optimisation** — out-of-the-box support for `GridSearchCV`, `RandomizedSearchCV`, and Bayesian search via `scikit-optimize`.
- **Extensible abstract base classes** — `BaseDataLoader`, `BaseModel`, and `BaseOptimizer` make it straightforward to plug in new data sources, algorithms, or evaluation logic.
- **Reproducible experiments** — random seeds, train/test splits, and full config snapshots are saved alongside model artefacts.
- **Debug mode** — step-by-step pipeline introspection with intermediate output shapes at every stage.
- **Compatible with the scikit-learn ecosystem** — works with scikit-learn, sktime, and tsfresh out of the box.

---

## Architecture

```
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│      Satellite Data Source   │    │    Terrestrial Sensor Network    │
│  (raster imagery, spectral   │    │  (weather stations, IoT nodes,   │
│   bands, retrieval products) │    │   MEMS arrays, reanalysis grids) │
└──────────────────┬──────────────┘    └──────────────────┬──────────────────┘
                 │                                    │
                 ▼                                    ▼
        ┌───────────────────────────────────────────────────┐
        │               BaseDataLoader                  │
        │   • Train / test split        • Shuffling     │
        │   • Stratification            • Normalisation │
        └─────────────────────────┬─────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────────┐
        │                  BaseModel                   │
        │   Pipeline of scikit-learn–compatible steps: │
        │     Scaler → Feature Engineering → Regressor │
        │   Unions for parallel feature stream fusion  │
        └───────────────────────┬──────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────────┐
        │               BaseOptimizer                │
        │   • GridSearchCV / RandomizedSearchCV /    │
        │     BayesSearchCV hyperparameter tuning    │
        │   • Cross-validation                       │
        │   • Model + report persistence             │
        └───────────────────────┬────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Saved Artefacts  │
                   │  model · config ·  │
                   │  train report ·    │
                   │  test report       │
                   └──────────────────────┘
```

---

## Folder Structure

```
forge/
│
├── main.py                        # Entry point — training and optional testing
│
├── requirements.txt               # Python dependencies
│
├── base/                          # Abstract base classes
│   ├── base_data_loader.py        #   Train/test split, shuffling
│   ├── base_model.py              #   Pipeline construction and step management
│   └── base_optimizer.py         #   Optimisation, model save/load, report export
│
├── configs/                       # JSON experiment configurations
│   ├── config_classification.json #   Example: cloud-type classification
│   └── config_regression.json    #   Example: temperature / precipitation regression
│
├── data/                          # Default location for input data files
│   ├── satellite/                 #   Satellite-derived features (raster extracts, etc.)
│   └── sensors/                  #   Terrestrial sensor time-series (CSV, HDF5, etc.)
│
├── data_loaders/
│   └── data_loaders.py           # Concrete data loaders for satellite + sensor data
│
├── models/
│   ├── __init__.py               # Registry: maps pipeline step names to classes
│   └── models.py                 # Concrete model definitions
│
├── optimizers/
│   └── optimizers.py             # Concrete optimizers (regression, classification)
│
├── saved/                         # Output artefacts (auto-created at runtime)
│   ├── <ExperimentName>/
│   │   ├── model.pkl
│   │   ├── config.json
│   │   ├── report_train.txt
│   │   └── report_test.txt
│
├── utils/
│   ├── parse_config.py           # ConfigParser: merges JSON config with CLI args
│   ├── parse_params.py           # Hyperparameter grid helpers
│   └── utils.py                  # General utilities
│
└── wrappers/
    ├── wrappers.py               # Modified sklearn estimators (e.g. PLS wrapper)
    └── data_transformations.py   # Custom transforms (e.g. Savitzky-Golay filter)
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.7
- [Anaconda](https://www.anaconda.com/) (recommended for environment management)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/jay-ramani/forge.git
cd forge
```

**2. Create and activate a virtual environment**

```bash
conda create -n forge-env python=3.9
conda activate forge-env
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

---

## Configuration

All experiment parameters live in a single JSON config file. No source changes are needed to switch datasets, algorithms, or search strategies.

### Config File Format

```json
{
    "name": "WeatherRegression",

    "model": {
        "type": "Model",
        "args": {
            "pipeline": ["scaler", "feature_union", "GBR"],
            "unions": {
                "feature_union": ["satellite_features", "sensor_features"]
            }
        }
    },

    "tuned_parameters": [{
        "GBR__n_estimators":  [100, 300, 500],
        "GBR__max_depth":     [3, 5, 7],
        "GBR__learning_rate": [0.01, 0.05, 0.1]
    }],

    "optimizer": "OptimizerRegression",

    "search_method": {
        "type": "BayesSearchCV",
        "args": {
            "n_iter":   50,
            "n_jobs":   -1,
            "verbose":  1,
            "refit":    true
        }
    },

    "cross_validation": {
        "type": "RepeatedKFold",
        "args": {
            "n_splits":    5,
            "n_repeats":   5,
            "random_state": 42
        }
    },

    "data_loader": {
        "type": "WeatherDataLoader",
        "args": {
            "data_path":     "data/",
            "shuffle":       true,
            "test_split":    0.2,
            "stratify":      false,
            "random_state":  42
        }
    },

    "score":       "min neg_mean_absolute_error",
    "test_model":  true,
    "debug":       false,
    "save_dir":    "saved/"
}
```

| Field | Description |
|---|---|
| `name` | Experiment name — used as the save sub-directory |
| `model.args.pipeline` | Ordered list of processing steps by registered name |
| `model.args.unions` | Named parallel feature unions for sensor fusion |
| `tuned_parameters` | Hyperparameter grid in scikit-learn double-underscore notation |
| `optimizer` | Name of the optimizer class to use |
| `search_method` | Search algorithm and its arguments |
| `cross_validation` | CV strategy and its arguments |
| `data_loader` | Data loader class and data-path / split settings |
| `score` | `"min"` or `"max"` followed by a scikit-learn metric name |
| `test_model` | Whether to evaluate the best model on the held-out test set |
| `debug` | Print step-by-step pipeline outputs |
| `save_dir` | Root directory for saved artefacts |

### Pipeline

Steps must be registered in `models/__init__.py` before use:

```python
from wrappers import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

methods_dict = {
    'scaler':            StandardScaler,
    'GBR':               GradientBoostingRegressor,
    'satellite_features': SatelliteFeatureTransformer,   # custom wrapper
    'sensor_features':   SensorFeatureTransformer,       # custom wrapper
}
```

### Unions (Feature Fusion)

Unions are the core sensor-fusion mechanism. They apply multiple transformers **in parallel** to the input data and concatenate their outputs before feeding the result to the next pipeline step:

```json
"pipeline": ["scaler", "sat_sensor_union", "RF"],
"unions": {
    "sat_sensor_union": ["satellite_features", "sensor_features"]
}
```

Hyperparameters inside a union follow the `union_name__step_name__param` convention:

```json
"tuned_parameters": [{
    "sat_sensor_union__satellite_features__n_components": [5, 10, 20],
    "sat_sensor_union__sensor_features__window_size":     [3, 6, 12]
}]
```

### Hyperparameter Search

| Method | When to use |
|---|---|
| `GridSearchCV` | Small, well-known parameter grids; exhaustive coverage needed |
| `RandomizedSearchCV` | Large grids; discovering new promising regions |
| `BayesSearchCV` | Maximum sample efficiency; long-running model evaluations |

Increase throughput by setting `"n_jobs": -1` to use all available CPU cores.

---

## Usage

### Training a Model

```bash
python main.py -c configs/config_regression.json
```

### Testing a Model

Set `"test_model": true` in the config file. After optimisation completes, Forge will automatically load the best model, run inference on the held-out test set, and write `report_test.txt` to the save directory.

### CLI Overrides

Common parameters can be overridden at runtime without editing the config file:

```bash
# Override the number of cross-validation repeats
python main.py -c configs/config_regression.json --cross_validation 20
```

Custom CLI arguments are registered in `main.py` via `collections.namedtuple`:

```python
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['-cv', '--cross_validation'], type=int, target='cross_validation;args;n_repeats'),
]
```

The `target` uses semicolons as key separators to traverse the nested config dictionary.

---

## Customisation

### Custom Data Loader

1. Inherit `BaseDataLoader` from `base/base_data_loader.py`.
2. Load your satellite imagery and sensor CSVs (or any format) and assign to the handler attributes:

```python
from base import BaseDataLoader

class WeatherDataLoader(BaseDataLoader):
    def __init__(self, data_path, training, **kwargs):
        super().__init__(**kwargs)

        satellite_df = load_satellite_features(data_path)   # your loader
        sensor_df    = load_sensor_timeseries(data_path)    # your loader
        combined     = fuse(satellite_df, sensor_df)        # merge on timestamp/grid

        if training:
            self.dh.X_data = combined.drop(columns=['target'])
            self.dh.y_data = combined['target']
        else:
            self.dh.X_data_test = combined.drop(columns=['target'])
            self.dh.y_data_test = combined['target']
```

Refer to `data_loaders/data_loaders.py` for a complete example.

### Custom Model

1. Inherit `BaseModel` from `base/base_model.py`.
2. Implement `created_model()` which returns a fitted pipeline:

```python
from base import BaseModel

class WeatherModel(BaseModel):
    def __init__(self, pipeline, **kwargs):
        steps = self.create_steps(pipeline)
        self.model = Pipeline(steps=steps)

    def created_model(self):
        return self.model
```

Refer to `models/models.py` for a complete example.

### Custom Optimizer

1. Inherit `BaseOptimizer` from `base/base_optimizer.py`.
2. Implement `fitted_model()`.  Optionally implement `create_train_report()` and `create_test_report()` for domain-specific metrics (e.g. RMSE, Brier score, Critical Success Index):

```python
from base import BaseOptimizer
from sklearn.metrics import mean_absolute_error, r2_score

class OptimizerRegression(BaseOptimizer):
    def fitted_model(self):
        self.search_method.fit(self.dh.X_data, self.dh.y_data)
        return self.search_method.best_estimator_

    def create_test_report(self, y_true, y_pred):
        return {
            'MAE':  mean_absolute_error(y_true, y_pred),
            'R2':   r2_score(y_true, y_pred),
        }
```

Refer to `optimizers/optimizers.py` for a complete example.

---

## Data Sources

Forge is data-source agnostic. Typical inputs for weather prediction tasks include:

**Satellite platforms**
- GOES-16/17/18 (ABI Level-2 products: cloud-top temperature, total precipitable water, derived motion winds)
- Meteosat / MSG (SEVIRI brightness temperatures, derived rainfall rates)
- Sentinel-3 OLCI/SLSTR (sea surface temperature, fire radiative power)
- MODIS / VIIRS (aerosol optical depth, snow cover)
- ERA5 reanalysis gridded fields (used as a high-quality satellite-era proxy)

**Terrestrial sensor networks**
- Automated Surface Observing System (ASOS) stations
- Mesonet networks (Oklahoma Mesonet, West Texas Mesonet, etc.)
- Amateur / citizen-science networks (Weather Underground Personal Weather Stations)
- IoT and MEMS deployments (BME280, SHT31, BMP390-class sensors)
- Radiosonde / upper-air sounding data

Prepare your data as tabular files (CSV, Parquet, HDF5) or structured NumPy arrays and plug in a custom `BaseDataLoader`.

---

## Roadmap

- [ ] Deep learning backends (PyTorch / TensorFlow pipelines via sklearn wrappers)
- [ ] Native support for NetCDF / GRIB2 satellite product ingestion
- [ ] Spatial cross-validation strategies (block CV, buffered leave-one-out)
- [ ] MLflow / Weights & Biases experiment tracking integration
- [ ] Docker image for reproducible deployment
- [ ] Real-time inference mode with streaming sensor data

See the [open issues](https://github.com/jay-ramani/forge/issues) to request a feature or report a bug.

---

## Contributing

Contributions are welcome and greatly appreciated.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.
