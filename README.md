# Testing the TP2 property of option prices

This repository contains the code and saved results for my MSc thesis. The
project has three parts:

1. Testing TP2 and RR2 inequalities in SPX option prices.
2. repeating the tests on constructed FX option surfaces.
3. Checking whether the SPX violations lead to a realistic trading strategy.

The best way to review the work is to run the three notebooks. They use the
small CSV files already included in `outputs/`, so the raw OptionMetrics data
are not needed and the notebooks should run quickly.

## Running the results notebooks

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
01_spx_results.ipynb       SPX tests and descriptive results
02_fx_results.ipynb        FX extension
03_trading_results.ipynb   trading backtests and regressions
```

They recreate the thesis tables and figures from the saved results. They do
not launch the long raw-data calculations.

## Repository layout

```text
src/spx.py       SPX data cleaning and TP2/RR2 tests
src/fx.py        FX surface preparation and TP2/RR2 tests
src/trading.py   trading backtests and factor regressions
notebooks/       short notebooks used to present the results
outputs/         compact CSV results used by the notebooks
figures/         figures produced by the notebooks
data/README.md   required raw-data folder layout
```

## Where the data came from

The large raw files are supplied separately in the accompanying Google Drive
folder because they are too large for GitHub. The folder structure must be
copied into `data/` as shown in `data/README.md`.

The SPX data come from OptionMetrics IvyDB US through WRDS. In WRDS, the annual option
price tables are `optionm_all.opprcdYYYY` (or `optionm.opprcdYYYY`, depending on
the subscription). I selected SPX using OptionMetrics `secid = 108105` and
downloaded 2000-2025, one annual CSV per year. These are saved as
`data/raw/optionMetricsSpxYYYY.csv`.

The supporting SPX files were obtained as follows:

| File | Source |
|---|---|
| `spx_forward_prices.csv` | OptionMetrics annual `fwdpr` tables on WRDS, filtered to SPX `secid = 108105` |
| `spx_option_settlement_types.csv` | `optionid` and `am_settlement` from the annual OptionMetrics `opprcdYYYY` tables |
| `spx_delta_exclusions.parquet` | dates and contracts with an invalid OptionMetrics delta, taken from the same annual option tables |
| `spx_market_data.csv` | SPX and VIX closing levels from the annual OptionMetrics `secprd` tables |
| `spx_settlement_prices.csv` | SPX closes for PM-settled contracts and official Cboe SET values for AM-settled contracts |

The FX files are the original and repaired option surfaces produced in the
earlier MATH70128 FX surface project. They are also supplied in the Google
Drive folder; they are not downloaded from WRDS by this repository.

## Rebuilding the raw results

This is optional. The included `outputs/` files are enough to review the thesis
results. Full SPX and trading runs take several hours.

SPX tests:

```powershell
python -m src.spx data/raw generated/spx --start 2000 --end 2025 `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet
```

Original FX surfaces:

```powershell
python -m src.fx data/fx/original generated/fx
```

The repaired L1, L1BA and L1BA-PC folders are run in the same way. For example:

```powershell
python -m src.fx data/fx/repaired_l1ba_pc generated/fx_l1ba_pc `
  --method L1BA-PC --original-folder data/fx/original
```

Trading backtests:

```powershell
python -m src.trading data/raw generated/trading --spec all `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet `
  --settlement-prices data/auxiliary/spx_settlement_prices.csv `
  --market-data data/auxiliary/spx_market_data.csv `
  --paper-results outputs/trading/paper_reported_by_year.csv
```

The trading command is the heaviest run. It produces the paper-style
replication, the six practical specifications, and the factor regressions.
