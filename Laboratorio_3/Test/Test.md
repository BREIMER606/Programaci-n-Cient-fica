# Tests

This folder contains small test scripts that validate the behaviour of the analysis code.  
The goal is to check that key functions in `Src/` work as expected on simple, synthetic data, without running the full pipeline or depending on the real dataset.

## Purpose of this folder

- Provide **unit and integration tests** for the main analysis functions (e.g. class definition, feature extraction, Random Forest training).
- Detect obvious bugs (missing columns, unexpected `NaN` values, wrong output shapes, failed model training) early.
- Serve as minimal, executable examples of how to call the functions in `Src.analysis`, independent of `Main.py`.

## Difference with `Main.py`

- `Main.py`
  - Orchestrates the **entire pipeline**: preprocessing → visualization → analysis on the real CSV files.
  - Reads and writes data under `data/` and `results/`, and is intended for full end‑to‑end runs.

- `tests/test_analysis.py` (and other test files)
  - Create **small synthetic DataFrames** in memory, instead of loading the full dataset.
  - Call individual functions from `Src.analysis` (and optionally `Src.preprocessing`) to verify:
    - that new columns are created correctly,
    - that outputs have the expected size and type,
    - that models can be trained and evaluated without errors.
  - Run quickly and do not replace the full analysis; they **guard against regressions** when the code is modified.

## How to run the tests

From the project root (`LABORATORIO_3/`), with the Conda environment activated:

conda activate LABORATORIO_3
python -m unittest tests/test_analysis.py


You can add more test files following the same pattern (e.g. `test_preprocessing.py`) to cover additional modules under `Src/`.
