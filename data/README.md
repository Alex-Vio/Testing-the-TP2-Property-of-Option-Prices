# Data layout

The result notebooks do not require the raw data.  To rerun the source scripts,
place the supplied files in the following layout.

```
data/
  raw/
    optionMetricsSpx2000.csv
    ...
    optionMetricsSpx2025.csv
  auxiliary/
    spx_forward_prices.csv
    spx_option_settlement_types.csv
    spx_delta_exclusions.parquet
    spx_settlement_prices.csv
    spx_market_data.csv
  fx/
    original/                 option_surface_PAIR.csv files
    repaired_l1/              call_surface_PAIR_l1.csv files
    repaired_l1ba/            call_surface_PAIR_l1ba.csv files
    repaired_l1ba_pc/         call_surface_PAIR_l1bapc.csv files
```

The annual SPX files are the OptionMetrics quote exports used in the thesis.
`spx_forward_prices.csv` and the delta exclusions come from the corresponding
OptionMetrics tables.  The settlement file combines the S&P 500 close for PM
contracts and CBOE SET for AM contracts.  The FX surface files are the outputs
of the earlier MATH70128 surface-construction project.

The raw files are kept outside Git because of GitHub's file-size limit.
