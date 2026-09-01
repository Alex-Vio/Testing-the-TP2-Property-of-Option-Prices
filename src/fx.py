"""
fx.py

This file runs the TP2/RR2 tests on the constructed FX option surfaces used in
Chapter 4 of the thesis.

The FX files already contain option prices, strikes, expiries and forward
prices. The task is to put the original and repaired surfaces
into the same column format as the SPX data, then apply the same forward-
moneyness inequalities.

Main functions:
    load_original_surfaces
        Reads the original bid, mid and ask call/put surfaces.

    load_repaired_surfaces
        Reads the saved L1, L1BA or L1BA-PC repaired surfaces.

    same_dates
        Restricts the original surfaces to the pair-dates available in a
        repaired sample, so the comparison uses identical observations.

    run_tests
        Runs the rounded mid-price and bid-ask robust TP2/RR2 tests for every
        currency pair and date.

    summarise / summarise_moneyness
        Combine the daily counts into the tables reported in the thesis.

    combine_saved_runs
        Combines the original and three repair runs into the five notebook
        input tables.

Important conventions:
    - FX prices are undiscounted prices in forward units.
    - Forward moneyness is k = K/F.
    - Calls violate TP2 when the determinant is negative.
    - Puts violate RR2 when the determinant is positive.
    - Cross-maturity strikes are rounded upward to an available strike.
    - The FX rounding limit is 5% of the target strike because the currency
      pairs have very different price scales.
"""

from pathlib import Path
import argparse

import polars as pl

from .spx import detect_violations


# ============================================================
# Put FX surfaces into the common option format
# ============================================================

def _finish_surface(table):
    """
    Add the derived columns expected by the TP2/RR2 test.

    The input has already been reshaped to one row per quote. This function
    adds forward moneyness and spread measures, removes unusable prices and
    sorts the rows by surface, date, maturity and strike.
    """

    return table.with_columns(
        (pl.col("K") / pl.col("F")).alias("k"),
        (pl.col("best_offer") - pl.col("best_bid")).alias("spread"),
        pl.lit(1.0).alias("discount_factor"),
        pl.lit(1).alias("volume"),
    ).with_columns(
        (pl.col("spread") / pl.col("mid")).alias("relative_spread")
    ).filter(
        (pl.col("K") > 0)
        & (pl.col("F") > 0)
        & (pl.col("mid") > 0)
        & (pl.col("best_bid") >= 0)
        & (pl.col("best_offer") >= pl.col("best_bid"))
    ).sort(["method", "pair", "date", "T_sort_days", "cp_flag", "K"])


def _option_rows(
    table, pair, method, cp_flag, columns,
    strike_column="strike", forward_column="forward",
):
    """
    Convert one call or put surface into the common option-column format.

    Parameters
    ----------
    table : polars.DataFrame
        Original or repaired FX surface.
    pair, method, cp_flag : str
        Currency pair, surface method and option type.
    columns : tuple
        Names of the mid, bid and ask price columns.
    strike_column, forward_column : str
        Columns containing the strike and forward used for this method.

    Returns
    -------
    polars.DataFrame
        One row per option quote with the column names used by spx.py.
    """

    price, bid, offer = columns
    return table.select(
        pl.lit(pair).alias("pair"),
        pl.col("date").cast(pl.Date),
        pl.col("expiry").alias("exdate"),
        pl.col("expiry").alias("T"),
        (pl.col("expiry") * 365).alias("T_days"),
        (pl.col("expiry") * 365).alias("T_sort_days"),
        pl.lit(0).alias("am_settlement"),
        pl.col(strike_column).alias("K"),
        pl.col(forward_column).alias("F"),
        pl.col(price).alias("mid"),
        pl.col(bid).alias("best_bid"),
        pl.col(offer).alias("best_offer"),
        pl.lit(cp_flag).alias("cp_flag"),
        pl.lit(method).alias("method"),
    )


def load_original_surfaces(folder):
    """
    Read and combine the original FX option surfaces.

    Each file contains bid, mid and ask call/put prices for one currency pair.
    Calls and puts are stacked into one table before the common derived columns
    are added.
    """

    tables = []
    for file_name in sorted(Path(folder).rglob("option_surface_*.csv")):
        # The mid rows contain the common strike grid. Raw bid/ask prices are
        # retained for the bid-ask robust test.
        raw = pl.read_csv(file_name, try_parse_dates=True).filter(pl.col("quote") == "mid")
        pair = file_name.stem.replace("option_surface_", "")
        calls = _option_rows(
            raw, pair, "original", "C",
            ("call_fv", "call_fv_bid_raw", "call_fv_ask_raw"),
        )
        puts = _option_rows(
            raw, pair, "original", "P",
            ("put_fv", "put_fv_bid_raw", "put_fv_ask_raw"),
        )
        tables.extend([calls, puts])
    return _finish_surface(pl.concat(tables, how="diagonal_relaxed"))


REPAIRED_COLUMNS = {
    "L1": {
        "forward": "forward_mid",
        "call": ("call_fv_repaired", "call_fv_bid", "call_fv_ask"),
    },
    "L1BA": {
        "forward": "forward",
        "call": ("call_fv_repaired", "call_bid", "call_ask"),
    },
    "L1BA-PC": {
        "forward": "forward",
        "call": ("call_fv_repaired", "call_bid", "call_ask"),
        "put": ("put_fv_repaired", "put_bid", "put_ask"),
    },
}


def load_repaired_surfaces(folder, method):
    """
    Read one of the L1, L1BA or L1BA-PC repaired samples.

    The three repair methods use slightly different saved column names. The
    REPAIRED_COLUMNS map records those differences so the testing code below
    can work with one common table.
    """

    method = method.upper()
    if method not in REPAIRED_COLUMNS:
        raise ValueError("method must be L1, L1BA or L1BA-PC")

    settings = REPAIRED_COLUMNS[method]
    tables = []
    for file_name in sorted(Path(folder).rglob("call_surface_*.csv")):
        # Files ending in _info.csv contain daily repair diagnostics rather
        # than quote-level surfaces.
        if file_name.name.endswith("_info.csv"):
            continue
        pair = file_name.stem.replace("call_surface_", "")
        for suffix in ("_l1bapc", "_l1ba", "_l1"):
            pair = pair.replace(suffix, "")

        raw = pl.read_csv(file_name, try_parse_dates=True)
        tables.append(_option_rows(
            raw, pair, method, "C", settings["call"],
            strike_column="strike_repaired",
            forward_column=settings["forward"],
        ))
        if "put" in settings:
            tables.append(_option_rows(
                raw, pair, method, "P", settings["put"],
                strike_column="strike_repaired",
                forward_column=settings["forward"],
            ))

    return _finish_surface(pl.concat(tables, how="diagonal_relaxed"))


def same_dates(original, repaired):
    """
    Restrict the original sample to pair-dates that were repaired.

    Repair methods are only run on selected dates.
    """

    dates = repaired.select("pair", "date").unique()
    return original.join(dates, on=["pair", "date"], how="inner").with_columns(
        pl.lit("original_on_repaired_dates").alias("method")
    )


def run_tests(surfaces, include_moneyness=False):
    """
    Run the two FX tests used in the thesis.

    For every method, currency pair and date, the function applies:
        1. the rounded mid-price TP2/RR2 test;
        2. the rounded bid-ask robust test.

    It returns daily call/put counts and when requested, the mid-price counts
    split by forward-moneyness region.
    """

    summaries = []
    moneyness = []
    settings = [("rounded_mid", False), ("rounded_bid_ask_robust", True)]

    # Each pair-date is one independent option surface.
    groups = surfaces.partition_by(["method", "pair", "date"], maintain_order=True)
    for number, day in enumerate(groups, 1):
        method = day["method"][0]
        pair = day["pair"][0]
        trade_date = day["date"][0]
        if number == 1 or number % 500 == 0 or number == len(groups):
            print(f"FX surface {number} of {len(groups)}")

        for run_name, bid_ask in settings:
            # Reuse the SPX inequality code. The relative rounding limit is
            # scale-free and therefore comparable across currency pairs.
            result = detect_violations(
                day,
                bid_ask=bid_ask,
                method="rounded",
                keep_rows=False,
                maximum_round_distance=float("inf"),
                maximum_relative_round_distance=0.05,
            )
            counts = {row["cp_flag"]: row for row in result["summary"].to_dicts()}
            call = counts["C"]
            put = counts["P"]
            summaries.append({
                "run_name": run_name, "method": method, "pair": pair,
                "date": trade_date,
                "call_pairs_tested": call["pairs_tested"],
                "call_violations": call["violations"],
                "put_pairs_tested": put["pairs_tested"],
                "put_violations": put["violations"],
            })
            if include_moneyness and run_name == "rounded_mid":
                present_option_types = day["cp_flag"].unique().to_list()
                moneyness.append(
                    result["moneyness"].filter(
                        pl.col("cp_flag").is_in(present_option_types)
                    ).with_columns(
                        pl.lit(run_name).alias("run_name"),
                        pl.lit(method).alias("method"),
                        pl.lit(pair).alias("pair"),
                    )
                )

    # Rates are calculated only after the call/put counts have been stored.
    daily = pl.DataFrame(summaries).with_columns(
        pl.when(pl.col("call_pairs_tested") > 0)
        .then(pl.col("call_violations") / pl.col("call_pairs_tested"))
        .otherwise(0.0).alias("call_violation_rate"),
        pl.when(pl.col("put_pairs_tested") > 0)
        .then(pl.col("put_violations") / pl.col("put_pairs_tested"))
        .otherwise(0.0).alias("put_violation_rate"),
    )
    moneyness_table = (
        pl.concat(moneyness, how="diagonal_relaxed") if moneyness else None
    )
    return daily, moneyness_table


def summarise(daily, columns):
    """
    Aggregate daily counts before calculating violation rates.

    Summing counts first gives the rate across all tested inequalities.
    """

    return daily.group_by(columns).agg(
        pl.len().alias("number_of_surfaces"),
        pl.col("call_pairs_tested").sum(),
        pl.col("call_violations").sum(),
        pl.col("put_pairs_tested").sum(),
        pl.col("put_violations").sum(),
    ).with_columns(
        pl.when(pl.col("call_pairs_tested") > 0)
        .then(pl.col("call_violations") / pl.col("call_pairs_tested"))
        .otherwise(0.0).alias("call_violation_rate"),
        pl.when(pl.col("put_pairs_tested") > 0)
        .then(pl.col("put_violations") / pl.col("put_pairs_tested"))
        .otherwise(0.0).alias("put_violation_rate"),
    ).sort(columns)


def summarise_moneyness(table):
    """
    Combine daily counts for each forward-moneyness region.

    The three regions are both options in-the-money, both out-of-the-money
    and one option on each side of the forward.
    """

    keys = ["run_name", "method", "cp_flag", "moneyness_region"]
    return table.group_by(keys).agg(
        pl.col("pairs_tested").sum(),
        pl.col("violations").sum(),
    ).with_columns(
        pl.when(pl.col("pairs_tested") > 0)
        .then(pl.col("violations") / pl.col("pairs_tested"))
        .otherwise(0.0).alias("violation_rate")
    ).sort(keys)


def combine_saved_runs(original_folder, repair_folders, output_folder):
    """
    Combine the separately saved FX runs into the five notebook input tables.

    The three repair runs use the same selected dates, so their repeated
    `original_on_repaired_dates` rows are removed after concatenation.
    """

    original = Path(original_folder)
    repairs = [Path(folder) for folder in repair_folders]
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    names = {
        "daily_summary.csv": "fx_original_daily_summary.csv",
        "method_summary.csv": "fx_original_method_summary.csv",
        "moneyness_summary.csv": "fx_original_moneyness_summary.csv",
    }
    for source, target in names.items():
        pl.read_csv(original / source, try_parse_dates=True).write_csv(output / target)

    for source, target in [
        ("method_summary.csv", "fx_repair_comparison_method_summary.csv"),
        ("pair_summary.csv", "fx_repair_comparison_pair_summary.csv"),
    ]:
        pl.concat([pl.read_csv(folder / source) for folder in repairs]).unique().sort(
            ["run_name", "method"] + (["pair"] if source.startswith("pair") else [])
        ).write_csv(output / target)


def main():
    """Read command-line arguments, run the selected FX sample and save tables."""

    parser = argparse.ArgumentParser(description="Run the FX TP2/RR2 extension")
    parser.add_argument("surface_folder")
    parser.add_argument("output_folder")
    parser.add_argument(
        "--method", choices=["original", "L1", "L1BA", "L1BA-PC"],
        default="original",
    )
    parser.add_argument(
        "--original-folder",
        help="For a repair, also test the original surfaces on the same dates",
    )
    parser.add_argument(
        "--combine-repairs", nargs="+", metavar="FOLDER",
        help="Combine an original result folder and the listed repair result folders",
    )
    args = parser.parse_args()

    if args.combine_repairs:
        combine_saved_runs(args.surface_folder, args.combine_repairs, args.output_folder)
        return

    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)

    # Read either the original surfaces or one repaired sample. If an original
    # folder is supplied for a repair, compare both methods on the same dates.
    if args.method == "original":
        surfaces = load_original_surfaces(args.surface_folder)
    else:
        repaired = load_repaired_surfaces(args.surface_folder, args.method)
        surfaces = repaired
        if args.original_folder:
            original = load_original_surfaces(args.original_folder)
            surfaces = pl.concat(
                [same_dates(original, repaired), repaired], how="diagonal_relaxed"
            )
    # The moneyness table is reported only for the full original sample.
    include_moneyness = args.method == "original"
    daily, moneyness = run_tests(surfaces, include_moneyness)
    summarise(daily, ["run_name", "method"]).write_csv(output / "method_summary.csv")
    if args.method == "original":
        daily.filter(pl.col("run_name") == "rounded_mid").write_csv(
            output / "daily_summary.csv"
        )
        summarise_moneyness(moneyness).write_csv(output / "moneyness_summary.csv")
    else:
        midpoint = daily.filter(pl.col("run_name") == "rounded_mid")
        summarise(midpoint, ["run_name", "method", "pair"]).write_csv(
            output / "pair_summary.csv"
        )


if __name__ == "__main__":
    main()
