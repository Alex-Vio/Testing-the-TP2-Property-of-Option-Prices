# TP2/RR2 thesis code

This is the compact submission version of the code used in the thesis.  It
contains the three empirical parts that remain in the final draft:

1. TP2/RR2 tests on SPX option chains;
2. the FX surface extension; and
3. the trading backtests.

The notebooks read the saved CSV results and recreate the thesis tables and
figures.  They do not start the multi-hour raw-data calculations.  The raw
calculations are kept separately in `src` so that the methods can still be
checked and rerun.

The included CSVs are the compact evidence used in the thesis, not example
data.  The three source files can rebuild every calculated table used by the
notebooks.  The annual results transcribed from the paper are retained as a
reference input rather than presented as a calculation of this project.

## Files

```
src/spx.py                 SPX cleaning, forwards and TP2/RR2 inequalities
src/fx.py                  FX surface conversion and the same inequalities
src/trading.py             the six Chapter 5 specifications and regressions
notebooks/01_spx_results.ipynb
notebooks/02_fx_results.ipynb
notebooks/03_trading_results.ipynb
outputs/                   compact results used by the notebooks
data/README.md             raw-data layout
```

There are no download, WRDS-ingestion or general data-exploration notebooks.
Only analyses reported in Chapters 3-5 of the thesis are included.

## Quick reproduction

Create an environment and install the small set of packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The notebooks use simple paths relative to their own folder, so start Jupyter
there:

```powershell
cd notebooks
jupyter lab
```

All three notebooks can then be run from top to bottom.  They write figures to
`figures/` and do not alter the saved CSV results.

Each notebook contains short comments on what the tables show and where the
interpretation is limited.  The detailed economic discussion remains in the
thesis rather than being duplicated in code.

## Raw calculations

The annual OptionMetrics files are larger than GitHub's individual-file limit.
They are supplied separately and should be placed as described in
`data/README.md`.  Rebuilding the full 2000-2025 results is intentionally not a
notebook switch because it takes many hours.

Main SPX test:

```powershell
python -m src.spx data/raw generated/spx --start 2000 --end 2025 `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet
```

This is a heavy run.  The same annual pass also creates the data-coverage,
quote-quality, calendar-monotonicity and violation-duration tables used by the
SPX notebook.

Original FX surfaces:

```powershell
python -m src.fx data/fx/original generated/fx
```

Each repaired sample is run in the same way.  For example, L1BA-PC is compared
with the original surfaces on exactly the same pair-dates using:

```powershell
python -m src.fx data/fx/repaired_l1ba_pc generated/fx_l1ba_pc `
  --method L1BA-PC --original-folder data/fx/original
```

The corresponding L1 and L1BA runs use `--method L1` and `--method L1BA`.
Their small method and pair summaries are combined in `outputs/fx` for the
results notebook.  Once all four result folders exist, combine them with:

```powershell
python -m src.fx generated/fx generated/submission_fx `
  --combine-repairs generated/fx_l1 generated/fx_l1ba generated/fx_l1ba_pc
```

Trading calculations:

```powershell
python -m src.trading data/raw generated/trading --spec all `
  --forwards data/auxiliary/spx_forward_prices.csv `
  --settlement-flags data/auxiliary/spx_option_settlement_types.csv `
  --delta-exclusions data/auxiliary/spx_delta_exclusions.parquet `
  --settlement-prices data/auxiliary/spx_settlement_prices.csv `
  --market-data data/auxiliary/spx_market_data.csv `
  --paper-results outputs/trading/paper_reported_by_year.csv
```

After the five portfolio simulations and the paper-style strong-signal test,
this command also rebuilds the two six-stage tables and the factor regressions.

The trading code uses cash-based sizing for every specification.  The paper
cash return and marked-equity return therefore use the same midpoint trade
path.  The practical specifications use next-day bid/ask execution, lagged
five-observation volume, a 25% maximum relative spread, a 10% participation
cap, netting, whole contracts and a USD 1 commission per contract.
