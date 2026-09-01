# Data

The shared Google Drive folder contains:

- `data/`: the raw files required to rerun the analysis;
- `sources/`: the papers and reports cited in the thesis.

The notebooks and saved results can be viewed without downloading either folder.

## Using the data

To rerun the analysis:

1. Download the `data` folder from Google Drive.
2. Copy it into the repository root, merging it with the existing `data` folder.
3. Check that the resulting paths begin with `data/raw`, `data/auxiliary` and `data/fx`.
4. Run the commands under **Rerunning the analysis** in the main README.

The `sources` folder is for reference only and is not used by the code.

## Folder structure

```text
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
    original/          option_surface_PAIR.csv
    repaired_l1/       call_surface_PAIR_l1.csv
    repaired_l1ba/     call_surface_PAIR_l1ba.csv
    repaired_l1ba_pc/  call_surface_PAIR_l1bapc.csv
```

The SPX files are OptionMetrics exports used in the thesis. The settlement file contains SPX closes for PM-settled contracts and Cboe SET values for AM-settled contracts.

The FX files are the original and repaired surfaces from the earlier MATH70128 project.

These files are stored outside GitHub because they exceed GitHub's file-size limits.