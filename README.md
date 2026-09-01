# Testing the TP2 property of option prices

This repository contains the code and saved results for my MSc thesis. The analysis covers:

1. TP2 and RR2 inequalities in SPX option prices;
2. the same tests on constructed FX option surfaces and
3. trading strategies based on SPX violations.

## How to use this repository

The main analysis code is contained in:

```text
src/spx.py       SPX data preparation and TP2/RR2 tests
src/fx.py        FX surface preparation and TP2/RR2 tests
src/trading.py   trading backtests and factor regressions
```

The repository can be used in two ways:

- the notebooks reproduce the thesis tables and figures from the saved results in `outputs/`;
- the Python commands below rerun the analysis from the raw data.

## Viewing the thesis results

The notebooks use the saved results included in the repository. They do not rerun the full analysis.

From the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Run the notebooks in this order:

```text
01_spx_results.ipynb
02_fx_results.ipynb
03_trading_results.ipynb
```

## Repository layout

```text
src/             analysis code
notebooks/       notebooks producing the thesis tables and figures
outputs/         saved CSV results used by the notebooks
figures/         figures produced by the notebooks
data/README.md   required raw-data folder structure
```

## Data

The raw data are provided separately in the accompanying Google Drive folder because they are too large for GitHub. Copy its `data` folder into the repository without changing the folder structure.

The SPX option data come from OptionMetrics IvyDB US through WRDS. The annual option-price tables are `optionm_all.opprcdYYYY` or `optionm.opprcdYYYY`, depending on the subscription. SPX is identified by OptionMetrics `secid = 108105`. The annual files for 2000–2025 are stored as:

```text
data/raw/optionMetricsSpxYYYY.csv
```

The supporting SPX files are:

| File | Source |
|---|---|
| `spx_forward_prices.csv` | OptionMetrics annual `fwdpr` tables, filtered to SPX |
| `spx_option_settlement_types.csv` | `optionid` and `am_settlement` from the annual option tables |
| `spx_delta_exclusions.parquet` | contracts with invalid OptionMetrics deltas |
| `spx_market_data.csv` | SPX and VIX closes from the annual `secprd` tables |
| `spx_settlement_prices.csv` | SPX closes and official Cboe SET values |

The FX files are the original and repaired surfaces from the earlier MATH70128 FX project. They are included in the Google Drive folder and are not downloaded from WRDS.

## Rerunning the analysis

First copy the Google Drive data into `data/` and install the requirements as shown above. Run these commands from the repository folder.

### SPX tests

```powershell
python -m src.spx data/raw generated/spx --start 2000 --end 2025 `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet
```

### FX tests

Original surfaces:

```powershell
python -m src.fx data/fx/original generated/fx
```

The repaired folders are run in the same way. For example:

```powershell
python -m src.fx data/fx/repaired_l1ba_pc generated/fx_l1ba_pc `
  --method L1BA-PC --original-folder data/fx/original
```

### Trading analysis

```powershell
python -m src.trading data/raw generated/trading --spec all `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet `
  --settlement-prices data/auxiliary/spx_settlement_prices.csv `
  --market-data data/auxiliary/spx_market_data.csv `
  --paper-results outputs/trading/paper_reported_by_year.csv
```

Approximate runtimes on the machine used for this thesis are:

- SPX tests: several hours for 2000–2025;
- FX tests: a few minutes per folder;
- trading analysis: one day or longer when running `--spec all`.

The trading analysis is the longest part of the rerun. It produces the paper replication, the six practical specifications and the factor regressions. Running all specifications may take one day or longer.

Fresh results are written to `generated/`, so the submitted results in `outputs/` are not overwritten. The two folders can then be compared. To use the new results in the notebooks, change the notebook input paths from `outputs/` to `generated/`.