# ds_utils

Utilities for Python data science projects.

## Installation

From the main project's root directory:

```bash
pip install -e ./src
```

## Package structure

```
ds_utils/
    __init__.py
    eda.py
    mlflow_utils.py
    ...
```

## Main features

- **EDA (Exploratory Data Analysis):**
  - `generate_full_report(df, target_col, name, correlation_threshold)`
  - Automatic reports for correlation, distribution and outliers.

- **MLflow Utils:**
  - MLflow experiment initialization and management.
  - Metrics, artifacts and model logging.

- **Other utilities:**
  - Path validation, artifact handling, notebook helpers.

## Usage example

```python
from ds_utils.eda import generate_full_report
import pandas as pd

df = pd.read_csv('data/my_dataset.csv')
generate_full_report(df, target_col='target', name='My Dataset')
```

## Requirements
- Python >= 3.8
- pandas, numpy, matplotlib, seaborn, scikit-learn, mlflow

## Development

- For local development, install in editable mode:
  ```bash
  pip install -e ./src
  ```
- Add your modules in the `ds_utils` folder.

## License
MIT

