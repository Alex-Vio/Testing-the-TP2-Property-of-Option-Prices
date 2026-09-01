"""
spx.py

This file contains the raw-data calculation behind the SPX results in Chapter
3 of the thesis.

The code reads one annual OptionMetrics file at a time, cleans the quotes,
attaches the SPX forward for each expiry and tests the forward-adjusted TP2
inequality for calls and RR2 inequality for puts.

Main functions:
    load_spx_file
        Cleans one annual option file, applies the settlement and delta
        information and adds forwards and forward moneyness.

    estimate_forwards
        Optional put-call-parity fallback when a forward file is unavailable.

    detect_violations
        Tests one trading date and returns overall, moneyness and maturity
        counts together with the individual violations when requested.

    run_years
        Applies the tests to all dates in the selected annual files and writes
        every SPX table used by the notebook, including quote quality,
        calendar monotonicity and violation duration.

Important conventions:
    - Strike is converted from OptionMetrics units by dividing by 1,000.
    - Forward moneyness is k = K/F.
    - The two maturities are ordered using settlement-adjusted maturity days.
    - Adjusted strikes are rounded upward to the next traded strike.
    - Calls violate when the determinant is negative.
    - Puts violate when the determinant is positive.
    - The bid-ask robust test values purchases at the offer and sales at the bid.
"""

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import argparse

import polars as pl


# ============================================================
# Load and clean the annual SPX files
# ============================================================

def _parse_dates(table, names=("date", "exdate")):
    """
    Parse the two date formats found in the annual OptionMetrics files.

    Some saved files use YYYY-MM-DD and others use YYYYMMDD. Existing Polars
    Date columns are left unchanged.
    """

    for name in names:
        if name in table.columns and table.schema[name] != pl.Date:
            table = table.with_columns(
                pl.coalesce(
                    pl.col(name).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                    pl.col(name).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
                ).alias(name)
            )
    return table


def estimate_forwards(table, minimum_pairs=5):
    """
    Estimate the forward and discount factor from put-call parity.

    For each date and expiry, regress the matched call-minus-put prices on
    strike using

        C(K,T) - P(K,T) = D(T)F(T) - D(T)K.

    The negative slope is the discount factor and the intercept divided by
    that discount factor is the forward. This is only a fallback and the thesis
    results use the supplied OptionMetrics forward file.
    """

    keys = ["date", "exdate", "am_settlement", "T_days", "T_sort_days"]
    calls = table.filter(pl.col("cp_flag") == "C").select(
        keys + ["K", pl.col("mid").alias("call")]
    )
    puts = table.filter(pl.col("cp_flag") == "P").select(
        keys + ["K", pl.col("mid").alias("put")]
    )
    pairs = calls.join(puts, on=keys + ["K"], how="inner").with_columns(
        (pl.col("call") - pl.col("put")).alias("y")
    )

    # Estimate the slope and intercept separately for every date-expiry.
    fit = pairs.group_by(keys).agg(
        pl.len().alias("n"),
        pl.col("K").mean().alias("x_bar"),
        pl.col("y").mean().alias("y_bar"),
        (pl.col("K") * pl.col("y")).mean().alias("xy_bar"),
        (pl.col("K") ** 2).mean().alias("x2_bar"),
    ).with_columns(
        (pl.col("xy_bar") - pl.col("x_bar") * pl.col("y_bar")).alias("cov"),
        (pl.col("x2_bar") - pl.col("x_bar") ** 2).alias("var"),
    ).with_columns(
        (pl.col("cov") / pl.col("var")).alias("slope")
    ).with_columns(
        (-pl.col("slope")).alias("discount_factor"),
        (pl.col("y_bar") - pl.col("slope") * pl.col("x_bar")).alias("intercept"),
    ).with_columns(
        (pl.col("intercept") / pl.col("discount_factor")).alias("F")
    ).filter(
        (pl.col("n") >= minimum_pairs)
        & (pl.col("var") > 0)
        & (pl.col("discount_factor") > 0)
        & (pl.col("F") > 0)
    )
    return fit.select(keys + ["F", "discount_factor"])


def load_spx_file(
    file_name,
    forward_file=None,
    settlement_file=None,
    delta_exclusions_file=None,
    estimate_discount=False,
):
    """
    Load and clean one annual OptionMetrics file.

    Parameters
    ----------
    file_name : path-like
        Annual raw option file.
    forward_file : path-like, optional
         OptionMetrics forwards by date and expiry.
    settlement_file : path-like, optional
         AM/PM settlement flag by option identifier.
    delta_exclusions_file : path-like, optional
        Contracts outside the paper's 0.01 to 0.99 absolute-delta range.
    estimate_discount : bool
        Estimate parity discount factors for the calendar check. Trading and
        the TP2/RR2 determinant itself do not need this extra calculation.

    Returns
    -------
    polars.DataFrame
        Clean quotes with strike K, mid-price, spread, maturity, forward F and
        forward moneyness k=K/F.
    """

    table = pl.read_csv(
        file_name,
        null_values=["", "NA", "NaN"],
        try_parse_dates=True,
        infer_schema_length=10_000,
    )
    table = _parse_dates(table)

    # Cast only the fields used later. Invalid values become null and are
    # removed by the quote filters below.
    numeric = [
        "strike_price", "best_bid", "best_offer", "volume", "open_interest",
        "optionid", "am_settlement",
    ]
    table = table.with_columns(
        [pl.col(name).cast(pl.Float64, strict=False) for name in numeric if name in table.columns]
    )

    # AM- and PM-settled options use different settlement times and prices.
    # Since some option IDs have inconsistent flags across the annual files,
    # replace the raw flag with the separately checked optionid mapping.
    if settlement_file is not None:
        settlement = pl.read_csv(settlement_file).select(
            pl.col("optionid").cast(pl.Int64),
            pl.col("am_settlement").cast(pl.Int64).alias("checked_am"),
        )
        table = table.with_columns(pl.col("optionid").cast(pl.Int64))
        table = table.join(settlement, on="optionid", how="left")
        table = table.drop("am_settlement").rename({"checked_am": "am_settlement"})
    elif "am_settlement" not in table.columns:
        raise ValueError("An AM/PM settlement flag is required")
    else:
        table = table.with_columns(pl.col("am_settlement").cast(pl.Int64))

    # Convert strike units and construct the quote and maturity variables.
    table = table.with_columns(
        (pl.col("strike_price") / 1000).alias("K"),
        ((pl.col("best_bid") + pl.col("best_offer")) / 2).alias("mid"),
        (pl.col("best_offer") - pl.col("best_bid")).alias("spread"),
        (pl.col("exdate") - pl.col("date")).dt.total_days().alias("T_days"),
    ).with_columns(
        pl.when(pl.col("am_settlement") == 0)
        .then(pl.col("T_days") + 0.5)
        .otherwise(pl.col("T_days"))
        .alias("T_sort_days"),
        pl.when(pl.col("mid") > 0)
        .then(pl.col("spread") / pl.col("mid"))
        .otherwise(None)
        .alias("relative_spread"),
    ).filter(
        (pl.col("T_days") > 0)
        & (pl.col("K") > 0)
        & (pl.col("best_bid") >= 0)
        & (pl.col("best_offer") >= pl.col("best_bid"))
        & (pl.col("mid") > 0)
    ).unique(maintain_order=True)

    # Remove contracts that failed the checked absolute-delta screen.
    if delta_exclusions_file is not None:
        excluded = pl.read_parquet(delta_exclusions_file).select(
            pl.col("date").cast(pl.Date), pl.col("optionid").cast(pl.Int64)
        )
        table = table.with_columns(pl.col("optionid").cast(pl.Int64))
        table = table.join(excluded, on=["date", "optionid"], how="anti")

    # Attach one forward to each date, expiry and settlement convention.
    keys = ["date", "exdate", "am_settlement"]
    if forward_file is not None:
        forwards = _parse_dates(pl.read_csv(forward_file, try_parse_dates=True)).select(
            keys + [pl.col("forward_price").cast(pl.Float64).alias("F")]
        )
        table = table.join(forwards, on=keys, how="left")
        if estimate_discount:
            # OptionMetrics supplies F but not the discount factor used by the
            # separate calendar-monotonicity check. Estimate only D from parity.
            discounts = estimate_forwards(table).select(
                keys + ["T_days", "T_sort_days", "discount_factor"]
            )
            table = table.join(
                discounts, on=keys + ["T_days", "T_sort_days"], how="left"
            )
        else:
            table = table.with_columns(pl.lit(1.0).alias("discount_factor"))
    else:
        forwards = estimate_forwards(table)
        table = table.join(
            forwards,
            on=keys + ["T_days", "T_sort_days"],
            how="left",
        )

    return table.filter(pl.col("F") > 0).with_columns(
        (pl.col("K") / pl.col("F")).alias("k")
    ).sort(["date", "T_sort_days", "cp_flag", "K"])


def round_up(
    strikes,
    target,
    maximum_distance: float | None = 50.0,
    maximum_relative_distance: float | None = None,
):
    """
    Return the first traded strike above a forward-adjusted target.

    The candidate is discarded if no higher strike exists or if the rounding
    distance exceeds either supplied limit. Upward rounding matches the rule
    used in the paper.
    """

    place = bisect_left(strikes, target)
    if place == len(strikes):
        return None
    strike = strikes[place]
    distance = strike - target
    if maximum_distance is not None and distance > maximum_distance:
        return None
    if maximum_relative_distance is not None and distance / target > maximum_relative_distance:
        return None
    return strike


def _region(cp_flag, k1, k2):
    """Classify an option pair relative to the forward for its option type."""

    if cp_flag == "C":
        if k1 >= 1 and k2 >= 1:
            return "both_otm"
        if k1 < 1 and k2 < 1:
            return "both_itm"
    else:
        if k1 <= 1 and k2 <= 1:
            return "both_otm"
        if k1 > 1 and k2 > 1:
            return "both_itm"
    return "crosses_forward"


# ============================================================
# TP2/RR2 test for one trading date
# ============================================================

def detect_violations(
    day,
    bid_ask=False,
    method="rounded",
    keep_rows=True,
    maximum_round_distance: float | None = 50.0,
    maximum_relative_round_distance: float | None = None,
    minimum_candidate_volume: float | None = None,
    maximum_candidate_relative_spread: float | None = None,
):
    """
    Test all eligible maturity-strike pairs on one trading date.

    The rounded test starts from k1 < k2 at maturities T1 < T2. Each strike is
    carried to the other maturity at the same forward moneyness and rounded up
    to an available quote. The four prices form the determinant

        price(K1,T1) price(K2,T2)
        - price(K1_cross,T2) price(K2_cross,T1).

    Mid-prices give the main test. With bid_ask=True, call
    purchases use offers and sales use bids. The directions are reversed for
    puts. Optional liquidity limits are applied to the four selected contracts,
    after the crossed strikes have been rounded on the full traded strike grid.
    The function returns aggregate counts and optionally one row for each
    violating quadruple.
    """

    if method not in ("rounded", "two_strike"):
        raise ValueError("method must be rounded or two_strike")

    all_violations = []
    summaries = []
    region_rows = []
    maturity_rows = []

    # Calls and puts have opposite determinant signs, so test them separately.
    for cp_flag in ("C", "P"):
        rows = day.filter(pl.col("cp_flag") == cp_flag).to_dicts()
        groups = defaultdict(list)
        for row in rows:
            groups[(row["exdate"], int(row["am_settlement"]), row["T_sort_days"])].append(row)

        # Build one strike grid and one quote lookup for each maturity.
        maturities = sorted(groups, key=lambda item: item[2])
        strikes = {m: sorted({row["K"] for row in groups[m]}) for m in maturities}
        quote = {(m, row["K"]): row for m in maturities for row in groups[m]}

        def eligible(row):
            """Check a selected contract without changing the strike grid."""

            enough_volume = (
                minimum_candidate_volume is None
                or row.get("volume") is not None
                and row["volume"] >= minimum_candidate_volume
            )
            narrow_enough = (
                maximum_candidate_relative_spread is None
                or row.get("relative_spread") is not None
                and row["relative_spread"] <= maximum_candidate_relative_spread
            )
            return enough_volume and narrow_enough

        candidates = {
            maturity: [row for row in groups[maturity] if eligible(row)]
            for maturity in maturities
        }
        regions = defaultdict(lambda: [0, 0])
        maturity_counts = defaultdict(lambda: [0, 0])
        tested = 0
        failed = 0

        # Form every ordered maturity pair and every eligible strike pair.
        for number, t1 in enumerate(maturities):
            for t2 in maturities[number + 1:]:
                for first in candidates[t1]:
                    for second in candidates[t2]:
                        if method == "rounded":
                            if first["k"] >= second["k"]:
                                continue
                            # Carry each strike to the other maturity at the
                            # same forward moneyness k=K/F, then round upward.
                            k1_target = first["K"] * second["F"] / first["F"]
                            k2_target = second["K"] * first["F"] / second["F"]
                            k1_cross = round_up(
                                strikes[t2], k1_target, maximum_round_distance,
                                maximum_relative_round_distance,
                            )
                            k2_cross = round_up(
                                strikes[t1], k2_target, maximum_round_distance,
                                maximum_relative_round_distance,
                            )
                        else:
                            if first["K"] >= second["K"]:
                                continue
                            k1_target = k1_cross = first["K"]
                            k2_target = k2_cross = second["K"]

                        crossed1 = quote.get((t2, k1_cross))
                        crossed2 = quote.get((t1, k2_cross))
                        if crossed1 is None or crossed2 is None:
                            continue
                        if not eligible(crossed1) or not eligible(crossed2):
                            continue

                        region = _region(cp_flag, first["k"], second["k"])
                        week_pair = (int(first["T_sort_days"] // 7) * 7,
                                     int(second["T_sort_days"] // 7) * 7)
                        tested += 1
                        regions[region][0] += 1
                        maturity_counts[week_pair][0] += 1

                        # Use either mid-price products or the executable sides
                        # required by the bid-ask robust definition.
                        if bid_ask and cp_flag == "C":
                            left = first["best_offer"] * second["best_offer"]
                            right = crossed1["best_bid"] * crossed2["best_bid"]
                        elif bid_ask:
                            left = first["best_bid"] * second["best_bid"]
                            right = crossed1["best_offer"] * crossed2["best_offer"]
                        else:
                            left = first["mid"] * second["mid"]
                            right = crossed1["mid"] * crossed2["mid"]

                        stat = left - right
                        violation = stat < 0 if cp_flag == "C" else stat > 0
                        if not violation:
                            continue

                        failed += 1
                        regions[region][1] += 1
                        maturity_counts[week_pair][1] += 1
                        if not keep_rows:
                            continue

                        # Save the complete four-corner construction so an
                        # individual violation can be audited later.
                        size = abs(stat)
                        four = [first, second, crossed1, crossed2]
                        spreads = [row.get("relative_spread") for row in four]
                        spreads = [value for value in spreads if value is not None]
                        volumes = [row.get("volume") for row in four]
                        volumes = [value for value in volumes if value is not None]
                        interests = [row.get("open_interest") for row in four]
                        interests = [value for value in interests if value is not None]
                        all_violations.append({
                            "date": first["date"], "cp_flag": cp_flag,
                            "exdate1": first["exdate"], "exdate2": second["exdate"],
                            "am_settlement1": first["am_settlement"],
                            "am_settlement2": second["am_settlement"],
                            "T1_days": first["T_days"], "T2_days": second["T_days"],
                            "T1_sort_days": first["T_sort_days"],
                            "T2_sort_days": second["T_sort_days"],
                            "K1": first["K"], "K2": second["K"],
                            "F1": first["F"], "F2": second["F"],
                            "discount_factor1": first.get("discount_factor"),
                            "discount_factor2": second.get("discount_factor"),
                            "k1": first["k"], "k2": second["k"],
                            "K1_adjusted": k1_target, "K2_adjusted": k2_target,
                            "K1_rounded": k1_cross, "K2_rounded": k2_cross,
                            "price_K1_T1": first["mid"],
                            "price_K2_T2": second["mid"],
                            "price_K1rounded_T2": crossed1["mid"],
                            "price_K2rounded_T1": crossed2["mid"],
                            "bid_K1_T1": first["best_bid"],
                            "ask_K1_T1": first["best_offer"],
                            "bid_K2_T2": second["best_bid"],
                            "ask_K2_T2": second["best_offer"],
                            "bid_K1rounded_T2": crossed1["best_bid"],
                            "ask_K1rounded_T2": crossed1["best_offer"],
                            "bid_K2rounded_T1": crossed2["best_bid"],
                            "ask_K2rounded_T1": crossed2["best_offer"],
                            "left_side": left, "right_side": right,
                            "violation_size": size,
                            "relative_violation_size": size / max(abs(left), 1e-12),
                            "moneyness_region": region,
                            "max_relative_spread": max(spreads) if spreads else None,
                            "mean_relative_spread": (
                                sum(spreads) / len(spreads) if spreads else None
                            ),
                            "volume_K1_T1": first["volume"],
                            "volume_K2_T2": second["volume"],
                            "volume_K1rounded_T2": crossed1["volume"],
                            "volume_K2rounded_T1": crossed2["volume"],
                            "min_leg_volume": min(volumes) if volumes else None,
                            "total_leg_volume": sum(volumes) if volumes else None,
                            "min_leg_open_interest": (
                                min(interests) if interests else None
                            ),
                        })

        summaries.append({
            "cp_flag": cp_flag, "pairs_tested": tested,
            "violations": failed,
            "violation_rate": failed / tested if tested else 0.0,
        })
        for name in ("both_itm", "crosses_forward", "both_otm"):
            n, v = regions[name]
            region_rows.append({
                "cp_flag": cp_flag, "moneyness_region": name,
                "pairs_tested": n, "violations": v,
                "violation_rate": v / n if n else 0.0,
            })
        for (t1_week, t2_week), (n, v) in maturity_counts.items():
            maturity_rows.append({
                "cp_flag": cp_flag, "T1_week_start": t1_week,
                "T2_week_start": t2_week, "pairs_tested": n,
                "violations": v, "violation_rate": v / n if n else 0.0,
            })

    return {
        "summary": pl.DataFrame(summaries),
        "violations": pl.DataFrame(all_violations) if all_violations else pl.DataFrame(),
        "moneyness": pl.DataFrame(region_rows),
        "maturity": pl.DataFrame(maturity_rows) if maturity_rows else pl.DataFrame(),
    }


def _add_counts(target, table, keys):
    """Add one daily count table to a running dictionary of totals."""

    for row in table.to_dicts():
        key = tuple(row[name] for name in keys)
        target[key][0] += row["pairs_tested"]
        target[key][1] += row["violations"]


def _quote_quality(table):
    """Summarise positive-bid, positive-volume quotes in one annual file."""

    liquid = table.filter((pl.col("best_bid") > 0) & (pl.col("volume") > 0))
    return liquid.group_by("cp_flag").agg(
        pl.len().alias("number_of_quotes"),
        pl.col("volume").median().alias("median_volume"),
        pl.col("volume").mean().alias("mean_volume"),
        pl.col("open_interest").median().alias("median_open_interest"),
        pl.col("open_interest").mean().alias("mean_open_interest"),
        pl.col("relative_spread").median().alias("median_relative_spread"),
        pl.col("relative_spread").mean().alias("mean_relative_spread"),
        pl.col("relative_spread").quantile(0.9).alias("q90_relative_spread"),
    )


def _add_violation_quality(totals, violations):
    """Add spread, volume and open-interest totals from one trading date."""

    if violations.is_empty():
        return
    columns = [
        "sum_max_relative_spread", "sum_mean_relative_spread",
        "sum_min_leg_volume", "sum_total_leg_volume",
        "open_interest_observations", "sum_min_leg_open_interest",
    ]
    summary = violations.group_by(["cp_flag", "moneyness_region"]).agg(
        pl.len().alias("violations"),
        pl.col("max_relative_spread").sum().alias(columns[0]),
        pl.col("mean_relative_spread").sum().alias(columns[1]),
        pl.col("min_leg_volume").sum().alias(columns[2]),
        pl.col("total_leg_volume").sum().alias(columns[3]),
        pl.col("min_leg_open_interest").count().alias(columns[4]),
        pl.col("min_leg_open_interest").sum().alias(columns[5]),
    )
    for row in summary.to_dicts():
        values = totals[(row["cp_flag"], row["moneyness_region"])]
        values[0] += row["violations"]
        for position, name in enumerate(columns, 1):
            values[position] += row[name] or 0


def _interpolate(curve, target):
    """Linearly interpolate inside a sorted strike-price curve."""

    place = bisect_left(curve[0], target)
    if place < len(curve[0]) and abs(curve[0][place] - target) < 1e-10:
        return curve[1][place]
    if place == 0 or place == len(curve[0]):
        return None
    lower, upper = curve[0][place - 1], curve[0][place]
    weight = (target - lower) / (upper - lower)
    return curve[1][place - 1] + weight * (curve[1][place] - curve[1][place - 1])


CALENDAR_COUNTS = [
    "total_violations", "calendar_testable", "calendar_not_testable",
    "calendar_violations", "calendar_consistent_violations",
    "fails_at_k1", "fails_at_k2",
]


def _add_calendar_counts(totals, day, violations, year):
    """Check whether each TP2/RR2 violation also fails calendar monotonicity."""

    if violations.is_empty():
        return
    curves = defaultdict(list)
    for row in day.to_dicts():
        key = (row["cp_flag"], row["exdate"], row["am_settlement"], row["T_sort_days"])
        curves[key].append((row["K"], row["mid"]))
    curves = {
        key: ([point[0] for point in sorted(points)],
              [point[1] for point in sorted(points)])
        for key, points in curves.items()
    }

    for row in violations.to_dicts():
        key = (year, row["cp_flag"], row["moneyness_region"])
        count = totals[key]
        count[0] += 1
        early = (
            row["cp_flag"], row["exdate1"], row["am_settlement1"],
            row["T1_sort_days"],
        )
        late = (
            row["cp_flag"], row["exdate2"], row["am_settlement2"],
            row["T2_sort_days"],
        )
        price_k1_late = _interpolate(curves[late], row["K1_adjusted"])
        price_k2_early = _interpolate(curves[early], row["K2_adjusted"])
        d1, d2 = row["discount_factor1"], row["discount_factor2"]
        if price_k1_late is None or price_k2_early is None or d1 is None or d2 is None:
            count[2] += 1
            continue

        scale1, scale2 = d1 * row["F1"], d2 * row["F2"]
        fail1 = price_k1_late / scale2 + 1e-12 < row["price_K1_T1"] / scale1
        fail2 = row["price_K2_T2"] / scale2 + 1e-12 < price_k2_early / scale1
        count[1] += 1
        count[5] += int(fail1)
        count[6] += int(fail2)
        count[3 if fail1 or fail2 else 4] += 1


def _option_key(cp_flag, expiry, am_settlement, strike):
    """Identify one option contract while allowing for harmless float noise."""

    return cp_flag, expiry, int(am_settlement), round(float(strike), 6)


def _pair_key(row):
    """Identify the two original options that start a violation episode."""

    return (
        row["cp_flag"], row["exdate1"], row["exdate2"],
        int(row["am_settlement1"]), int(row["am_settlement2"]),
        round(float(row["K1"]), 6), round(float(row["K2"]), 6),
    )


def _duration_state():
    """Create the small dictionaries used to follow open violation episodes."""

    return {
        "active": {}, "violating": set(), "expiry": defaultdict(set),
        "gaps": defaultdict(set), "outcomes": defaultdict(int),
        "durations": defaultdict(int), "gap_outcomes": defaultdict(int),
        "index": -1, "last_date": None,
    }


def _duration_lookup(day):
    """Build today's option and strike lookups for duration retesting."""

    rows, strikes = {}, defaultdict(list)
    for row in day.to_dicts():
        option = _option_key(row["cp_flag"], row["exdate"], row["am_settlement"], row["K"])
        maturity = (row["cp_flag"], row["exdate"], row["am_settlement"], row["T_sort_days"])
        rows[option] = row
        strikes[maturity].append(row["K"])
    return rows, {key: sorted(set(values)) for key, values in strikes.items()}


def _pair_testable(key, rows, strikes):
    """Return True when the same original comparison can be priced today."""

    cp, ex1, ex2, am1, am2, k1, k2 = key
    first = rows.get(_option_key(cp, ex1, am1, k1))
    second = rows.get(_option_key(cp, ex2, am2, k2))
    if first is None or second is None or first["k"] >= second["k"]:
        return False
    maturity1 = (cp, ex1, am1, first["T_sort_days"])
    maturity2 = (cp, ex2, am2, second["T_sort_days"])
    cross1 = round_up(strikes[maturity2], first["K"] * second["F"] / first["F"])
    cross2 = round_up(strikes[maturity1], second["K"] * first["F"] / second["F"])
    return cross1 is not None and cross2 is not None


def _remove_gap(state, key):
    """Remove one episode from the two contract-to-gap indexes."""

    cp, ex1, ex2, am1, am2, k1, k2 = key
    for option in [_option_key(cp, ex1, am1, k1), _option_key(cp, ex2, am2, k2)]:
        state["gaps"][option].discard(key)
        if not state["gaps"][option]:
            del state["gaps"][option]


def _finish_episode(state, key, outcome):
    """Record one episode outcome in the full and short-maturity samples."""

    episode = state["active"][key]
    samples = ["full"] + (["T2_le_60"] if episode["T2"] <= 60 else [])
    duration = state["index"] - episode["start"]
    for sample in samples:
        base = (sample, key[0], "all")
        state["outcomes"][base + (outcome,)] += 1
        state["durations"][base + (outcome, duration)] += 1
        if episode["had_gap"]:
            state["gap_outcomes"][base + (episode["reappeared"], outcome)] += 1


def _drop_episode(state, key):
    """Remove a finished episode from every active index."""

    episode = state["active"].pop(key)
    state["violating"].discard(key)
    state["expiry"][key[1]].discard(key)
    if not state["expiry"][key[1]]:
        del state["expiry"][key[1]]
    if episode["state"] == "gap":
        _remove_gap(state, key)


def _update_duration(state, day, violations, trade_date):
    """Advance every open violation episode by one observed trading date."""

    # A long break in the input is censoring, not an observed correction.
    if state["last_date"] is not None and (trade_date - state["last_date"]).days > 10:
        for key in list(state["active"]):
            _finish_episode(state, key, "sample_gap")
            _drop_episode(state, key)
    state["index"] += 1
    state["last_date"] = trade_date

    # The shorter expiry ends the comparison before today's quote test.
    for expiry in [date for date in state["expiry"] if date <= trade_date]:
        for key in list(state["expiry"].get(expiry, ())):
            episode = state["active"][key]
            outcome = (
                "expired_unresolved_after_gap"
                if episode["state"] == "gap" else "expired_while_violating"
            )
            _finish_episode(state, key, outcome)
            _drop_episode(state, key)

    rows, strikes = _duration_lookup(day)
    today = {}
    for row in violations.to_dicts():
        key = _pair_key(row)
        if key not in today or row["violation_size"] > today[key]["violation_size"]:
            today[key] = row

    # Start new episodes and mark any gap that has reappeared as a violation.
    for key, row in today.items():
        if key not in state["active"]:
            state["active"][key] = {
                "start": state["index"], "T2": row["T2_days"],
                "state": "violating", "had_gap": False, "reappeared": False,
            }
            state["expiry"][key[1]].add(key)
        elif state["active"][key]["state"] == "gap":
            _remove_gap(state, key)
            state["active"][key]["state"] = "violating"
            state["active"][key]["reappeared"] = True

    # If a previously observed violation can be retested and now passes, it is
    # corrected. Missing required quotes create an observability gap instead.
    for key in list(state["violating"] - set(today)):
        if _pair_testable(key, rows, strikes):
            _finish_episode(state, key, "corrected")
            _drop_episode(state, key)
        else:
            state["active"][key]["state"] = "gap"
            state["active"][key]["had_gap"] = True
            cp, ex1, ex2, am1, am2, k1, k2 = key
            state["gaps"][_option_key(cp, ex1, am1, k1)].add(key)
            state["gaps"][_option_key(cp, ex2, am2, k2)].add(key)

    # A gap needs retesting only when one of its two original contracts returns.
    gap_candidates = set()
    for option in rows:
        gap_candidates.update(state["gaps"].get(option, ()))
    for key in gap_candidates - set(today):
        if key in state["active"] and _pair_testable(key, rows, strikes):
            _finish_episode(state, key, "corrected")
            _drop_episode(state, key)

    state["violating"] = {
        key for key in today
        if key in state["active"] and state["active"][key]["state"] == "violating"
    }


def _count_table(counts, names):
    """Convert a dictionary of grouped integer counts to a Polars table."""

    return pl.DataFrame([
        {**dict(zip(names, key)), "count": count}
        for key, count in counts.items()
    ])


def _write_duration_outputs(state, output_folder):
    """Finish open episodes and write the four duration tables used later."""

    for key in list(state["active"]):
        _finish_episode(state, key, "sample_ended")

    outcomes = _count_table(
        state["outcomes"], ["sample", "cp_flag", "liquidity", "outcome"]
    ).with_columns(
        (pl.col("count") / pl.col("count").sum().over(
            ["sample", "cp_flag", "liquidity"]
        )).alias("share")
    ).sort(["sample", "liquidity", "cp_flag", "outcome"])

    durations = _count_table(
        state["durations"],
        ["sample", "cp_flag", "liquidity", "outcome", "trading_days"],
    ).filter(pl.col("sample") == "full").with_columns(
        (pl.col("count") / pl.col("count").sum().over(
            ["sample", "cp_flag", "liquidity"]
        )).alias("share_of_all_episodes")
    ).sort(["sample", "liquidity", "cp_flag", "trading_days", "outcome"])

    gaps = _count_table(
        state["gap_outcomes"],
        ["sample", "cp_flag", "liquidity", "ever_reappeared", "outcome"],
    ).filter(pl.col("sample") == "full").with_columns(
        (pl.col("count") / pl.col("count").sum().over(
            ["sample", "cp_flag", "liquidity"]
        )).alias("share_of_gap_episodes")
    ).sort(["sample", "liquidity", "cp_flag", "ever_reappeared", "outcome"])

    # Competing-risk cumulative probabilities are calculated from the duration
    # counts, one trading-day horizon at a time.
    persistence_rows = []
    for cp_flag in ("C", "P"):
        group = durations.filter(pl.col("cp_flag") == cp_flag)
        by_age = defaultdict(lambda: defaultdict(int))
        for row in group.to_dicts():
            by_age[row["trading_days"]][row["outcome"]] += row["count"]
        at_risk = sum(row["count"] for row in group.to_dicts())
        unresolved, corrected_cif, expiry_cif, gap_cif = 1.0, 0.0, 0.0, 0.0
        for age in range(max(by_age, default=0) + 1):
            today = by_age[age]
            corrected = today["corrected"]
            expiry = today["expired_while_violating"]
            gap = today["expired_unresolved_after_gap"]
            censored = today["sample_ended"] + today["sample_gap"]
            if at_risk:
                corrected_cif += unresolved * corrected / at_risk
                expiry_cif += unresolved * expiry / at_risk
                gap_cif += unresolved * gap / at_risk
                unresolved *= 1 - (corrected + expiry + gap) / at_risk
            persistence_rows.append({
                "sample": "full", "cp_flag": cp_flag, "liquidity": "all",
                "trading_days_after_start": age, "at_risk": at_risk,
                "corrected_today": corrected,
                "expired_while_violating_today": expiry,
                "expired_unresolved_today": gap, "sample_censored_today": censored,
                "confirmed_correction_probability": corrected_cif,
                "expired_while_violating_probability": expiry_cif,
                "expired_unresolved_probability": gap_cif,
                "still_unresolved_probability": unresolved,
            })
            at_risk -= corrected + expiry + gap + censored

    output = Path(output_folder)
    outcomes.write_csv(output / "duration_final_outcomes.csv")
    durations.write_csv(output / "duration_distribution.csv")
    pl.DataFrame(persistence_rows).write_csv(output / "persistence_curve.csv")
    gaps.write_csv(output / "gap_outcomes.csv")


def run_years(data_folder, output_folder, years, forward_file, settlement_file, delta_file):
    """
    Rebuild the main empirical SPX tables from the annual raw files.

    Four tests are run for each trading date: rounded mid-price, rounded
    bid-ask robust, two-strike midprice and two-strike bid-ask robust. The rounded
    mid-price test also supplies the daily, annual, moneyness and maturity
    summaries used in Chapter 3.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    daily_rows = []
    yearly_rows = []
    moneyness = defaultdict(lambda: [0, 0])
    maturity = defaultdict(lambda: [0, 0])
    paper_checks = defaultdict(lambda: [0, 0])
    paper_checks_by_year = []
    coverage_rows = []
    quote_years = []
    violation_quality = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0, 0.0])
    calendar_counts = defaultdict(lambda: [0] * len(CALENDAR_COUNTS))
    duration = _duration_state()

    # These four checks differ only in quote side and strike construction.
    checks = [
        ("rounded_mid", False, "rounded"),
        ("rounded_bid_ask_robust", True, "rounded"),
        ("two_strike_mid", False, "two_strike"),
        ("two_strike_bid_ask_robust", True, "two_strike"),
    ]

    for year in years:
        print(f"SPX {year}")
        file_path = Path(data_folder) / f"optionMetricsSpx{year}.csv"
        table = load_spx_file(
            file_path,
            forward_file, settlement_file, delta_file,
            estimate_discount=True,
        )
        quote_years.append(_quote_quality(table).with_columns(pl.lit(year).alias("year")))
        with file_path.open("r", encoding="utf-8") as file:
            raw_columns = file.readline().strip().split(",")
        coverage_rows.append({
            "year": year, "file": file_path.name,
            "file_size_mb": file_path.stat().st_size / 1_000_000,
            "first_date": table["date"].min(), "last_date": table["date"].max(),
            "number_of_dates": table["date"].n_unique(),
            "has_am_settlement_column": "am_settlement" in raw_columns,
            "has_delta_column": "delta" in raw_columns,
        })
        year_counts = {"C": [0, 0], "P": [0, 0]}
        year_paper_checks = defaultdict(lambda: [0, 0])

        # Each date is tested independently.
        for day in table.partition_by("date", maintain_order=True):
            trade_date = day["date"][0]
            # This is the paper-style SPX quote screen.
            test_day = day.filter((pl.col("best_bid") > 0) & (pl.col("volume") > 0))
            results = {}
            for name, bid_ask, method in checks:
                results[name] = detect_violations(
                    test_day, bid_ask=bid_ask, method=method,
                    keep_rows=name == "rounded_mid",
                )
                for row in results[name]["summary"].to_dicts():
                    key = (name, row["cp_flag"])
                    paper_checks[key][0] += row["pairs_tested"]
                    paper_checks[key][1] += row["violations"]
                    year_paper_checks[key][0] += row["pairs_tested"]
                    year_paper_checks[key][1] += row["violations"]

            main = results["rounded_mid"]
            row = {"date": trade_date}
            for result in main["summary"].to_dicts():
                cp = result["cp_flag"]
                label = "call" if cp == "C" else "put"
                row[f"{label}_pairs_tested"] = result["pairs_tested"]
                row[f"{label}_violations"] = result["violations"]
                row[f"{label}_violation_rate"] = result["violation_rate"]
                year_counts[cp][0] += result["pairs_tested"]
                year_counts[cp][1] += result["violations"]
            daily_rows.append(row)
            _add_counts(moneyness, main["moneyness"], ["cp_flag", "moneyness_region"])
            _add_counts(maturity, main["maturity"], ["cp_flag", "T1_week_start", "T2_week_start"])
            _add_violation_quality(violation_quality, main["violations"])
            _add_calendar_counts(calendar_counts, test_day, main["violations"], year)
            _update_duration(duration, test_day, main["violations"], trade_date)

        yearly_rows.append({
            "year": year,
            "number_of_dates": len(table["date"].unique()),
            "call_pairs_tested": year_counts["C"][0],
            "call_violations": year_counts["C"][1],
            "call_violation_rate": year_counts["C"][1] / year_counts["C"][0],
            "put_pairs_tested": year_counts["P"][0],
            "put_violations": year_counts["P"][1],
            "put_violation_rate": year_counts["P"][1] / year_counts["P"][0],
        })
        for name, bid_ask, method in checks:
            call = year_paper_checks[(name, "C")]
            put = year_paper_checks[(name, "P")]
            paper_checks_by_year.append({
                "year": year, "run_name": name,
                "max_round_distance": 50.0,
                "max_relative_spread": None,
                "use_bid_ask": bid_ask, "test_method": method,
                "forward_source": "optionmetrics",
                "delta_filter": delta_file is not None,
                "verified_settlement_flags": settlement_file is not None,
                "number_of_dates": len(table["date"].unique()),
                "call_pairs_tested": call[0], "call_violations": call[1],
                "call_violation_rate": call[1] / call[0] if call[0] else 0.0,
                "put_pairs_tested": put[0], "put_violations": put[1],
                "put_violation_rate": put[1] / put[0] if put[0] else 0.0,
            })

    # Save daily and annual headline counts.
    daily = pl.DataFrame(daily_rows).sort("date").with_columns(
        pl.col("date").dt.year().alias("year")
    )
    yearly = pl.DataFrame(yearly_rows).sort("year")
    daily.write_csv(output_folder / "daily_summary.csv")
    yearly.write_csv(output_folder / "overall_by_year.csv")

    daily.with_columns(
        pl.col("call_violation_rate").rolling_mean(30).alias("call_rate_30d"),
        pl.col("put_violation_rate").rolling_mean(30).alias("put_rate_30d"),
    ).write_csv(output_folder / "rolling_30_day_rates.csv")

    def save_counts(counts, keys, file_name):
        """Convert one running count dictionary into a CSV table."""

        rows = []
        for key, (tested, failed) in counts.items():
            rows.append({
                **dict(zip(keys, key)), "pairs_tested": tested,
                "violations": failed,
                "violation_rate": failed / tested if tested else 0.0,
            })
        pl.DataFrame(rows).write_csv(output_folder / file_name)

    save_counts(moneyness, ["cp_flag", "moneyness_region"], "moneyness_violation_rates.csv")
    save_counts(maturity, ["cp_flag", "T1_week_start", "T2_week_start"], "maturity_violation_rates.csv")
    pl.DataFrame(paper_checks_by_year).sort(["year", "run_name"]).write_csv(
        output_folder / "paper_checks_by_year.csv"
    )

    paper_rows = []
    for name, bid_ask, method in checks:
        call = paper_checks[(name, "C")]
        put = paper_checks[(name, "P")]
        paper_rows.append({
            "run_name": name, "number_of_dates": daily.height,
            "call_pairs_tested": call[0], "call_violations": call[1],
            "put_pairs_tested": put[0], "put_violations": put[1],
            "use_bid_ask": bid_ask, "test_method": method,
            "forward_source": "optionmetrics",
            "delta_filter": delta_file is not None,
            "verified_settlement_flags": settlement_file is not None,
            "call_violation_rate": call[1] / call[0] if call[0] else 0.0,
            "put_violation_rate": put[1] / put[0] if put[0] else 0.0,
        })
    pl.DataFrame(paper_rows).sort("run_name").write_csv(
        output_folder / "paper_checks_overall.csv"
    )

    # Data coverage and quote quality are calculated from the same cleaned
    # annual files, so they require no separate exploratory notebook.
    pl.DataFrame(coverage_rows).sort("year").write_csv(
        output_folder / "data_coverage_by_year.csv"
    )
    quote_by_year = pl.concat(quote_years, how="diagonal_relaxed")
    quote_quality = quote_by_year.group_by("cp_flag").agg(
        pl.col("number_of_quotes").sum(),
        pl.col("median_volume").median(),
        ((pl.col("mean_volume") * pl.col("number_of_quotes")).sum()
         / pl.col("number_of_quotes").sum()).alias("mean_volume"),
        pl.col("median_open_interest").median(),
        ((pl.col("mean_open_interest") * pl.col("number_of_quotes")).sum()
         / pl.col("number_of_quotes").sum()).alias("mean_open_interest"),
        pl.col("median_relative_spread").median(),
        ((pl.col("mean_relative_spread") * pl.col("number_of_quotes")).sum()
         / pl.col("number_of_quotes").sum()).alias("mean_relative_spread"),
        pl.col("q90_relative_spread").median(),
    ).sort("cp_flag")
    quote_quality.write_csv(output_folder / "quote_quality_summary.csv")

    quality_rows = []
    for (cp_flag, region), values in violation_quality.items():
        n, max_spread, mean_spread, min_volume, total_volume, oi_n, min_oi = values
        quality_rows.append({
            "cp_flag": cp_flag, "moneyness_region": region, "violations": n,
            "sum_max_relative_spread": max_spread,
            "sum_mean_relative_spread": mean_spread,
            "sum_min_leg_volume": min_volume,
            "sum_total_leg_volume": total_volume,
            "open_interest_observations": oi_n,
            "sum_min_leg_open_interest": min_oi,
            "mean_max_relative_spread": max_spread / n,
            "mean_leg_relative_spread": mean_spread / n,
            "mean_min_leg_volume": min_volume / n,
            "mean_total_leg_volume": total_volume / n,
            "mean_min_leg_open_interest": min_oi / oi_n if oi_n else None,
        })
    pl.DataFrame(quality_rows).sort(["cp_flag", "moneyness_region"]).write_csv(
        output_folder / "violation_spread_summary.csv"
    )

    calendar_rows = [
        {"year": year, "cp_flag": cp_flag, "moneyness_region": region,
         **dict(zip(CALENDAR_COUNTS, values))}
        for (year, cp_flag, region), values in calendar_counts.items()
    ]
    calendar_by_region_year = pl.DataFrame(calendar_rows)

    def calendar_table(table, keys):
        """Pool calendar counts and add the three reported shares."""

        pooled = table.group_by(keys).agg(
            [pl.col(name).sum() for name in CALENDAR_COUNTS]
        )
        return pooled.with_columns(
            (pl.col("calendar_violations") / pl.col("calendar_testable"))
            .alias("calendar_violation_share"),
            (pl.col("calendar_consistent_violations") / pl.col("calendar_testable"))
            .alias("calendar_consistent_share"),
            (pl.col("calendar_not_testable") / pl.col("total_violations"))
            .alias("calendar_not_testable_share"),
        )

    calendar_table(calendar_by_region_year, ["year", "cp_flag"]).sort(
        ["year", "cp_flag"]
    ).write_csv(output_folder / "calendar_monotonicity_check_by_year.csv")
    calendar_table(calendar_by_region_year, ["cp_flag"]).sort("cp_flag").write_csv(
        output_folder / "calendar_monotonicity_check.csv"
    )
    calendar_table(
        calendar_by_region_year, ["cp_flag", "moneyness_region"]
    ).sort(["cp_flag", "moneyness_region"]).write_csv(
        output_folder / "calendar_monotonicity_by_moneyness.csv"
    )
    _write_duration_outputs(duration, output_folder)


def main():
    """Read command-line arguments and run the selected SPX years."""

    parser = argparse.ArgumentParser(description="Run the SPX TP2/RR2 tests")
    parser.add_argument("data_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--forwards", required=True)
    parser.add_argument("--settlement-flags", required=True)
    parser.add_argument("--delta-exclusions", required=True)
    args = parser.parse_args()
    run_years(
        args.data_folder, args.output_folder, range(args.start, args.end + 1),
        args.forwards, args.settlement_flags, args.delta_exclusions,
    )


if __name__ == "__main__":
    main()
