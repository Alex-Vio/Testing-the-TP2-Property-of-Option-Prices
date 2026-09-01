"""
trading.py

This file contains the trading calculations for Chapter 5 of the thesis.

Each TP2/RR2 violation is converted into the two-leg portfolios described by
Glasserman, Li and Pirjol. The option quantities are the prices from the other
side of the violated product inequality. Positions receive more weight when
the initial signal credit is large relative to gross option exposure and all
positions are held to settlement.

Five portfolio simulations produce the six backtests in the thesis:
    1. mid-price execution with cash-balance returns;
    2. the same mid-price trades measured using marked portfolio equity;
    3. same-day execution at bid and ask prices;
    4. next-day execution at bid and ask prices;
    5. mid-price signals with next-day executable and liquidity restrictions;
    6. bid-ask robust signals with the same restrictions.

The file also contains the paper-style same-day bid-ask robust signal trade test and
the monthly factor regressions reported after the main backtest table.

Main functions:
    make_trade
        Converts one detected violation into a two-leg strategy and computes
        its settlement payoff.

    run_backtest
        Runs one strategy specification over the annual SPX files and saves the daily cash balance,
        marked equity, trade records and counts of detected, filtered and executable signals.

    make_backtest_table
        Calculates return, Sharpe ratio, drawdown and daily-loss statistics for
        the six reported specifications.

    factor_regressions
        Runs the monthly market and market-volatility-liquidity regressions.

    make_paper_comparison
        Compares annual mid-price cash returns and Sharpes with the published
        annual series and rebuilds the replication table.

Important conventions:
    - Initial wealth is USD 1 million and SPX option prices are quoted in index points, with each point worth USD 100 per contract.
    - All specifications size new trades from cash using kappa = 100.
    - Cash and marked-equity mid-price returns use identical trades.
    - A next-day signal is detected on date t and traded using quotes on t+1.
    - Lagged volume contains only observations available before execution.
    - Open long options are marked at the bid and shorts at the offer.
    - If an open option has no valid quote, its most recent liquidation value is carried forward until a new quote becomes available or the option settles.
    - Positions are normally held to settlement; positions still open at a long data gap or sample end use their latest available liquidation marks.
"""

from collections import defaultdict, deque
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from .spx import detect_violations, load_spx_file


# ============================================================
# Strategy definitions
# ============================================================

HEADLINE = [("C", "T1"), ("C", "K2"), ("P", "T1"), ("P", "K1")]

COMMON = {
    "signal": "mid",
    "execution": "mid",
    "lag": 0,
    "minimum_volume": 10,
    "maximum_spread": None,
    "lagged_volume": None,
    "volume_fraction": None,
    "net": False,
    "whole_contracts": False,
    "commission": 0.0,
}

LIQUID = {
    **COMMON,
    "execution": "bid_ask",
    "lag": 1,
    "minimum_volume": 1,
    "maximum_spread": 0.25,
    "lagged_volume": 10,
    "volume_fraction": 0.10,
    "net": True,
    "whole_contracts": True,
    "commission": 1.0,
}

SPECS = {
    "midpoint": dict(COMMON),
    "same_day_bid_ask": {**COMMON, "execution": "bid_ask"},
    "next_day_bid_ask": {**COMMON, "execution": "bid_ask", "lag": 1},
    "mid_signal_liquid_next_day": dict(LIQUID),
    "strong_signal_liquid_next_day": {**LIQUID, "signal": "bid_ask"},
    # This is the paper-style Table 5.12 check.
    "strong_signal_same_day": {
        **COMMON, "signal": "bid_ask", "execution": "bid_ask",
        "minimum_volume": 1,
    },
}


# ============================================================
# Settlement, quote and signal helpers
# ============================================================

def read_settlements(file_name):
    """
    Read the checked SPX settlement prices into a lookup dictionary.

    The key contains the expiry date and AM/PM settlement flag because both
    conventions can occur on the same calendar date.
    """

    table = pl.read_csv(file_name, try_parse_dates=True).select(
        pl.col("date").cast(pl.Date),
        pl.col("am_settlement").cast(pl.Int64),
        pl.col("settlement_price").cast(pl.Float64),
    )
    return {
        (row["date"], row["am_settlement"]): row["settlement_price"]
        for row in table.to_dicts()
    }


def contract_key(cp_flag, expiry, am_settlement, strike):
    """Build the common identifier used for quotes, positions and marks."""

    return cp_flag, expiry, int(am_settlement), round(float(strike), 6)


def paper_filter(signal, minimum_volume):
    """
    Apply the paper's additional screening criteria to one detected violation.

    All four options must satisfy the daily-volume requirement. The function
    also removes simple vertical-price inconsistencies at each maturity and
    checks that forward-normalised prices are non-decreasing with maturity.
    These screens prevent a basic arbitrage error from being labelled as a
    TP2/RR2 trading signal.
    """

    volumes = [
        signal["volume_K1_T1"], signal["volume_K2_T2"],
        signal["volume_K1rounded_T2"], signal["volume_K2rounded_T1"],
    ]
    if any(value is None or value < minimum_volume for value in volumes):
        return False

    # Check monotonicity in strike separately at each maturity.
    same_maturity = [
        (signal["K1"], signal["price_K1_T1"],
         signal["K2_rounded"], signal["price_K2rounded_T1"]),
        (signal["K1_rounded"], signal["price_K1rounded_T2"],
         signal["K2"], signal["price_K2_T2"]),
    ]
    for strike1, price1, strike2, price2 in same_maturity:
        if strike1 < strike2:
            good = price1 >= price2 if signal["cp_flag"] == "C" else price1 <= price2
        elif strike1 > strike2:
            good = price1 <= price2 if signal["cp_flag"] == "C" else price1 >= price2
        else:
            good = abs(price1 - price2) < 1e-10
        if not good:
            return False

    # Approximate calendar checks in forward-normalised price units.
    first = signal["price_K1_T1"] / signal["F1"]
    first_later = signal["price_K1rounded_T2"] / signal["F2"]
    second_early = signal["price_K2rounded_T1"] / signal["F1"]
    second = signal["price_K2_T2"] / signal["F2"]
    return first <= first_later and second_early <= second


def _quotes(day, histories, lookback=5):
    """
    Build today's contract lookup and attach lagged average volume.

    histories contains only earlier dates. Today's volume is added after all
    orders have been processed, avoiding look-ahead in the liquidity screen.
    """

    quotes = {}
    for row in day.to_dicts():
        key = contract_key(row["cp_flag"], row["exdate"], row["am_settlement"], row["K"])
        row = dict(row)
        old = histories.get(key, ())
        row["lagged_average_volume"] = (
            sum(old) / len(old) if len(old) >= lookback else None
        )
        quotes[key] = row
    return quotes


def _signal_prices(signal, rule):
    """
    Return the four prices used to decide whether the signal is violated.

    Mid-price signals use all four mid prices. Bid-ask robust call signals value the two
    purchases at the offer and the two sales at the bid; puts reverse those
    directions because RR2 has the opposite determinant sign.
    """

    if rule == "mid":
        return {
            "K1_T1": signal["price_K1_T1"],
            "K2_T2": signal["price_K2_T2"],
            "K1rounded_T2": signal["price_K1rounded_T2"],
            "K2rounded_T1": signal["price_K2rounded_T1"],
        }
    if signal["cp_flag"] == "C":
        return {
            "K1_T1": signal["ask_K1_T1"],
            "K2_T2": signal["ask_K2_T2"],
            "K1rounded_T2": signal["bid_K1rounded_T2"],
            "K2rounded_T1": signal["bid_K2rounded_T1"],
        }
    return {
        "K1_T1": signal["bid_K1_T1"],
        "K2_T2": signal["bid_K2_T2"],
        "K1rounded_T2": signal["ask_K1rounded_T2"],
        "K2rounded_T1": signal["ask_K2rounded_T1"],
    }


def _option_payoff(cp_flag, settlement, strike):
    """Return the intrinsic value of one call or put at settlement."""

    if cp_flag == "C":
        return max(settlement - strike, 0.0)
    return max(strike - settlement, 0.0)


# ============================================================
# Construct one paper portfolio from one violation
# ============================================================

def make_trade(signal, denomination, quotes, settlements, spec, execution_date):
    """
    Turn one violation into one of the paper's two-leg portfolios.

    The four prices form two products, labelled left and right. A call
    violation buys the left product and sells the right product; a put
    violation reverses those positions. The T1, T2, K1 and K2 denominations
    select which two options are traded and which two prices determine
    their quantities.

    The returned trade stores executable entry prices, settlement profit,
    gross exposure and the initial-credit weight used for portfolio sizing.
    None is returned when a required quote, settlement or liquidity condition
    is unavailable.
    """

    prices = _signal_prices(signal, spec["signal"])
    left = prices["K1_T1"] * prices["K2_T2"]
    right = prices["K1rounded_T2"] * prices["K2rounded_T1"]
    credit = right - left if signal["cp_flag"] == "C" else left - right
    gross_signal = left + right
    if credit <= 0 or gross_signal <= 0:
        return None

    left_sign = 1 if signal["cp_flag"] == "C" else -1
    # Sign and contract details for the four corners of the determinant.
    contracts = {
        "K1_T1": (signal["exdate1"], signal["am_settlement1"], signal["K1"], left_sign),
        "K2_T2": (signal["exdate2"], signal["am_settlement2"], signal["K2"], left_sign),
        "K1rounded_T2": (
            signal["exdate2"], signal["am_settlement2"],
            signal["K1_rounded"], -left_sign,
        ),
        "K2rounded_T1": (
            signal["exdate1"], signal["am_settlement1"],
            signal["K2_rounded"], -left_sign,
        ),
    }
    # Each tuple defines one trade leg. The first item is the contract traded,
    # and the second is the contract whose signal-date price sets its quantity.
    # This expresses the two products in the TP2/RR2 inequality as a two-leg trade.
    definitions = {
        "T1": [("K1_T1", "K2_T2"), ("K2rounded_T1", "K1rounded_T2")],
        "T2": [("K2_T2", "K1_T1"), ("K1rounded_T2", "K2rounded_T1")],
        "K1": [("K1_T1", "K2_T2"), ("K1rounded_T2", "K2rounded_T1")],
        "K2": [("K2_T2", "K1_T1"), ("K2rounded_T1", "K1rounded_T2")],
    }

    # Retrieve the trade-date quote for each selected contract and apply the
    # spread and volume filters.
    legs = []
    for contract_name, quantity_name in definitions[denomination]:
        expiry, am, strike, direction = contracts[contract_name]
        key = contract_key(signal["cp_flag"], expiry, am, strike)
        quote = quotes.get(key)
        if quote is None or quote["best_bid"] <= 0 or quote["volume"] <= 0:
            return None
        spread = (quote["best_offer"] - quote["best_bid"]) / quote["mid"]
        if spec["maximum_spread"] is not None and spread > spec["maximum_spread"]:
            return None
        lagged_volume = quote["lagged_average_volume"]
        if spec["lagged_volume"] is not None:
            if lagged_volume is None or lagged_volume < spec["lagged_volume"]:
                return None

        quantity = direction * prices[quantity_name]
        if spec["execution"] == "mid":
            entry = quote["mid"]
        else:
            entry = quote["best_offer"] if quantity > 0 else quote["best_bid"]
        if (expiry, int(am)) not in settlements:
            return None
        legs.append({
            "key": key, "quantity": quantity, "entry": entry,
            "mid": quote["mid"], "lagged_volume": lagged_volume,
        })

    # Profit equals the entry credit plus the two settlement cashflows.
    entry_credit = -sum(leg["quantity"] * leg["entry"] for leg in legs)
    gross = sum(abs(leg["quantity"] * leg["entry"]) for leg in legs)
    payoff = 0.0
    for leg in legs:
        cp_flag, expiry, am, strike = leg["key"]
        settlement = settlements[(expiry, am)]
        option_payoff = _option_payoff(cp_flag, settlement, strike)
        payoff += leg["quantity"] * option_payoff

    return {
        "signal_date": signal["date"], "execution_date": execution_date,
        "cp_flag": signal["cp_flag"], "denomination": denomination,
        # This is the corrected paper weight: initial credit / gross exposure.
        "weight": credit / gross_signal,
        "gross": gross, "profit_on_gross": (entry_credit + payoff) / gross,
        "legs": legs,
    }


def _mark_position(quantity, quote, previous):
    """
    Mark one open position at its current liquidation price.

    Long options use the bid and short options use the offer. If today's quote
    is missing or invalid, the previous valid mark is retained.
    """

    if quote is None:
        return previous
    mark = quote["best_bid"] if quantity > 0 else quote["best_offer"]
    if mark is None or not math.isfinite(float(mark)) or mark < 0:
        return previous
    # A zero bid is a valid liquidation value for a long option.
    return float(mark)


def _liquidate(strategy, cash, positions, marks, daily_rows, last_row):
    """
    Close open positions after a long data gap or at the end of the sample.

    Each position is valued using its latest available liquidation mark. Its
    value is added to cash, the position is removed and the final marked
    equity is set equal to the final cash balance.
    """

    for key, quantity in positions[strategy].items():
        cash[strategy] += quantity * marks[strategy][key] * 100
    positions[strategy].clear()
    marks[strategy].clear()
    if strategy in last_row:
        row = daily_rows[last_row[strategy]]
        row["ending_cash"] = cash[strategy]
        row["ending_equity"] = cash[strategy]
        row["open_contracts"] = 0.0


def run_backtest(
    run_name,
    data_folder,
    output_folder,
    years,
    forward_file,
    settlement_flags,
    delta_exclusions,
    settlement_prices,
    initial_cash=1_000_000.0,
    kappa=100.0,
):
    """
    Run one trading specification and save its complete daily path.

    Parameters
    ----------
    run_name : str
        Name of one specification in SPECS.
    data_folder, output_folder : path-like
        Annual OptionMetrics input folder and destination for saved results.
    years : iterable of int
        Calendar years to process in order.
    forward_file, settlement_flags, delta_exclusions : path-like
        Checked SPX inputs used by load_spx_file.
    settlement_prices : path-like
        Realised SPX settlement levels for option payoffs.
    initial_cash : float
        Starting wealth for each headline strategy.
    kappa : float
        Gross-exposure divisor. With kappa=100, new trades receive total gross
        exposure equal to approximately 1% of available cash before capacity
        limits and whole-contract rounding.

    Returns
    -------
    polars.DataFrame
        Daily cash, marked equity, open-contract and commission history.
    """

    # Each headline strategy is simulated as a separate USD 1 million account.
    spec = SPECS[run_name]
    strategies = list(HEADLINE)
    cash = {strategy: initial_cash for strategy in strategies}
    positions = {strategy: defaultdict(float) for strategy in strategies}
    marks = {strategy: {} for strategy in strategies}
    histories = defaultdict(lambda: deque(maxlen=5))
    settlements = read_settlements(settlement_prices)
    pending = []
    daily_rows = []
    last_row = {}
    trade_rows = []
    survival = defaultdict(lambda: [0, 0, 0])
    previous_date = None

    # Process the annual files in calendar order so open positions and lagged
    # volume histories carry correctly from one year to the next.
    for year in years:
        print(f"{run_name}: {year}")
        table = load_spx_file(
            Path(data_folder) / f"optionMetricsSpx{year}.csv",
            forward_file, settlement_flags, delta_exclusions,
        )

        for day in table.partition_by("date", maintain_order=True):
            trade_date = day["date"][0]

            # A long calendar gap means the continuous daily quote history has
            # been broken. Close positions at their latest valid marks and
            # restart the lagged-volume history.
            if previous_date is not None and (trade_date - previous_date).days > 10:
                for strategy in strategies:
                    _liquidate(strategy, cash, positions, marks, daily_rows, last_row)
                histories.clear()
                pending = []

            quotes = _quotes(day, histories)
            cash_for_sizing = dict(cash)

            # Existing positions settle before today's new orders.  A payoff
            # due exactly today is not used to enlarge today's position sizes.
            for strategy in strategies:
                for key in list(positions[strategy]):
                    cp_flag, expiry, am, strike = key
                    if expiry > trade_date:
                        continue
                    settlement = settlements[(expiry, am)]
                    payoff = _option_payoff(cp_flag, settlement, strike)
                    amount = positions[strategy].pop(key) * payoff * 100
                    cash[strategy] += amount
                    if expiry < trade_date:
                        cash_for_sizing[strategy] += amount
                    marks[strategy].pop(key, None)

            # Mark positions carried into today before opening new trades.
            starting_equity = {}
            starting_cash = dict(cash)
            for strategy in strategies:
                market_value = 0.0
                for key, quantity in positions[strategy].items():
                    mark = _mark_position(quantity, quotes.get(key), marks[strategy].get(key))
                    if mark is not None:
                        marks[strategy][key] = mark
                        market_value += quantity * mark * 100
                starting_equity[strategy] = cash[strategy] + market_value

            # A lagged specification trades yesterday's saved signals using
            # today's quotes. Same-day specifications trade today's signals.
            signals_to_trade = pending if spec["lag"] else []
            pending = []

            # The paper defines the traded strike grid using positive bids and
            # volume. Choose crossed strikes on that grid first. The detector
            # then rejects the signal if any selected leg fails this strategy's
            # tighter volume or spread rule.
            signal_day = day.filter(
                (pl.col("best_bid") > 0)
                & (pl.col("volume") > 0)
            )
            signals = detect_violations(
                signal_day,
                bid_ask=spec["signal"] == "bid_ask",
                method="rounded",
                keep_rows=True,
                minimum_candidate_volume=spec["minimum_volume"],
                maximum_candidate_relative_spread=spec["maximum_spread"],
            )["violations"].to_dicts()
            if spec["lag"]:
                pending = signals
            else:
                signals_to_trade = signals

            # Apply the paper filters and build executable trades for the
            # headline denomination belonging to this option type.
            trades = defaultdict(list)
            for signal in signals_to_trade:
                relevant = [strategy for strategy in strategies if strategy[0] == signal["cp_flag"]]
                for strategy in relevant:
                    survival[strategy][0] += 1

                if not paper_filter(signal, spec["minimum_volume"]):
                    continue
                if spec["maximum_spread"] is not None:
                    if signal["max_relative_spread"] > spec["maximum_spread"]:
                        continue

                for strategy in relevant:
                    survival[strategy][1] += 1
                    trade = make_trade(
                        signal, strategy[1], quotes, settlements, spec, trade_date,
                    )
                    if trade is None:
                        continue
                    survival[strategy][2] += 1
                    trades[strategy].append(trade)
                    trade_rows.append({
                        "year": trade_date.year, "cp_flag": strategy[0],
                        "denomination": strategy[1],
                        "profit_on_gross": trade["profit_on_gross"],
                    })

            # Convert accepted trades into orders, then apply sizing, netting,
            # volume capacity, whole-contract rounding and commissions.
            for strategy in strategies:
                todays_trades = trades[strategy]
                orders = []
                if todays_trades and cash_for_sizing[strategy] > 0:
                    total_weight = sum(trade["weight"] for trade in todays_trades)
                    for trade in todays_trades:
                        portfolio_weight = trade["weight"] / total_weight
                        scale = (
                            cash_for_sizing[strategy] / kappa * portfolio_weight
                            / (100 * trade["gross"])
                        )
                        for leg in trade["legs"]:
                            orders.append({**leg, "quantity": leg["quantity"] * scale})

                # Offset orders in the same contract before applying capacity.
                if spec["net"]:
                    net = defaultdict(float)
                    for order in orders:
                        net[order["key"]] += order["quantity"]
                    orders = []
                    for key, quantity in net.items():
                        quote = quotes[key]
                        entry = quote["best_offer"] if quantity > 0 else quote["best_bid"]
                        orders.append({
                            "key": key, "quantity": quantity, "entry": entry,
                            "mid": quote["mid"],
                            "lagged_volume": quote["lagged_average_volume"],
                        })

                # The practical specifications cap each net order at 10% of
                # its lagged five-observation average volume.
                capacity = 1.0
                if spec["volume_fraction"] is not None and orders:
                    limits = []
                    for order in orders:
                        volume = order["lagged_volume"]
                        if volume is None or volume <= 0:
                            limits.append(0.0)
                        elif order["quantity"] != 0:
                            limits.append(
                                spec["volume_fraction"] * volume / abs(order["quantity"])
                            )
                    if limits:
                        capacity = min(1.0, min(limits))

                # Scale all legs by the tightest contract-level capacity limit.
                quantities = [order["quantity"] * capacity for order in orders]
                if spec["whole_contracts"]:
                    whole = []
                    for value in quantities:
                        if abs(value) <= 1e-12:
                            whole.append(0)
                        else:
                            contracts = int(abs(value) + 1e-12)
                            whole.append(contracts if value > 0 else -contracts)
                    quantities = whole
                    if not (any(value > 0 for value in quantities)
                            and any(value < 0 for value in quantities)):
                        quantities = [0 for _ in quantities]

                # Execute orders and add them to the open position book.
                commissions = 0.0
                for order, quantity in zip(orders, quantities):
                    if quantity == 0:
                        continue
                    cash[strategy] -= quantity * order["entry"] * 100
                    cost = abs(quantity) * spec["commission"]
                    cash[strategy] -= cost
                    commissions += cost
                    positions[strategy][order["key"]] += quantity
                    marks[strategy][order["key"]] = order["mid"]

                # End-of-day equity values every open long at the bid and every
                # open short at the offer.
                market_value = 0.0
                for key, quantity in positions[strategy].items():
                    mark = _mark_position(quantity, quotes.get(key), marks[strategy].get(key))
                    if mark is not None:
                        marks[strategy][key] = mark
                        market_value += quantity * mark * 100

                daily_rows.append({
                    "date": trade_date, "run_name": run_name,
                    "cp_flag": strategy[0], "denomination": strategy[1],
                    "starting_cash": starting_cash[strategy],
                    "ending_cash": cash[strategy],
                    "starting_equity": starting_equity[strategy],
                    "ending_equity": cash[strategy] + market_value,
                    "market_option_volume": day["volume"].sum(),
                    "market_median_relative_spread": day.filter(pl.col("best_bid") > 0)["relative_spread"].median(),
                    "open_contracts": sum(abs(value) for value in positions[strategy].values()),
                    "commissions": commissions,
                })
                last_row[strategy] = len(daily_rows) - 1

            # Today's volume only becomes available after today's orders.
            for key, quote in quotes.items():
                if quote["volume"] is not None:
                    histories[key].append(float(quote["volume"]))
            for key in list(histories):
                if key[1] <= trade_date:
                    del histories[key]
            previous_date = trade_date

    # Close any positions still open at the end of the selected sample.
    for strategy in strategies:
        _liquidate(strategy, cash, positions, marks, daily_rows, last_row)

    # Save the daily wealth path and the smaller trade/signal summaries.
    output = Path(output_folder) / run_name
    output.mkdir(parents=True, exist_ok=True)
    daily = pl.DataFrame(daily_rows)
    daily.write_csv(output / "daily.csv")

    if trade_rows:
        trades = pl.DataFrame(trade_rows).group_by(
            ["year", "cp_flag", "denomination"]
        ).agg(
            pl.len().alias("number_of_trades"),
            pl.col("profit_on_gross").mean().alias("mean_profit_on_gross"),
            (pl.col("profit_on_gross") > 0).mean().alias("hit_rate"),
        ).sort(["year", "cp_flag", "denomination"])
    else:
        trades = pl.DataFrame()
    trades.write_csv(output / "trade_summary.csv")

    survival_rows = []
    for strategy, values in survival.items():
        survival_rows.append({
            "cp_flag": strategy[0], "denomination": strategy[1],
            "signals_detected": values[0], "signals_after_filters": values[1],
            "signals_executable": values[2],
        })
    pl.DataFrame(survival_rows).write_csv(output / "signal_survival.csv")
    return daily


# ============================================================
# Return and risk tables
# ============================================================

def _statistics(daily, value, starting_value, start_year, end_year):
    """
    Calculate the return and risk measures for one accounting series.

    The mean return is the arithmetic mean of calendar-year returns. Sharpe is
    the ordinary annualised full-sample Sharpe ratio from monthly returns,

        sqrt(12) * mean(monthly return) / std(monthly return).

    Drawdown uses the running peak of the selected wealth measure. Maximum
    daily loss uses consecutive daily values, with the first day compared with
    its recorded starting value.
    """

    rows = []
    groups = daily.partition_by(["cp_flag", "denomination"], as_dict=True)
    for key, full_table in groups.items():
        cp_flag, denomination = key
        full_table = full_table.sort("date")
        table = full_table.filter(
            pl.col("date").dt.year().is_between(start_year, end_year)
        )
        if table.is_empty():
            continue
        values = table[value].to_numpy()
        dates = table["date"].to_list()
        first_start = float(table[starting_value][0])
        earlier = full_table.filter(pl.col("date") < dates[0])
        annual_start = (
            float(earlier[value][-1]) if not earlier.is_empty() else first_start
        )
        frame = pd.DataFrame({"date": dates, "value": values}).set_index("date")

        # Build non-overlapping month-end returns from the daily wealth path.
        monthly_returns = []
        # For a holdout beginning in 2023, January must start from the final
        # 2022 wealth rather than from January's first recorded intraday value.
        previous = annual_start
        for _, group in frame.groupby(pd.to_datetime(frame.index).to_period("M")):
            end = group["value"].iloc[-1]
            if previous > 0:
                monthly_returns.append(end / previous - 1)
            previous = end

        # Calendar-year returns use the final wealth from the preceding year.
        annual = []
        previous = annual_start
        for _, group in frame.groupby(pd.to_datetime(frame.index).year):
            end = group["value"].iloc[-1]
            if previous > 0:
                annual.append(end / previous - 1)
            previous = end

        # Daily returns and running-peak drawdowns use the same wealth series.
        previous_values = np.r_[first_start, values[:-1]]
        positive_start = previous_values > 0
        daily_returns = values[positive_start] / previous_values[positive_start] - 1
        peaks = np.maximum.accumulate(np.r_[first_start, values])[1:]
        drawdown = values / peaks - 1
        monthly_returns = np.asarray(monthly_returns)
        rows.append({
            "cp_flag": cp_flag, "denomination": denomination,
            "mean_return": float(np.mean(annual)),
            "sharpe": (
                float(np.sqrt(12) * np.mean(monthly_returns) / np.std(monthly_returns, ddof=1))
                if np.std(monthly_returns, ddof=1) > 0 else None
            ),
            "maximum_drawdown": float(-np.min(drawdown)),
            "maximum_daily_loss": float(-np.min(daily_returns)),
        })
    return pl.DataFrame(rows)


def make_backtest_table(output_folder, start_year, end_year):
    """
    Create the six-stage return and risk table used by the notebook.

    Stages 1 and 2 read the same mid-price simulation but use cash and marked
    equity respectively. Stages 3 to 6 use marked equity from progressively
    more realistic execution specifications.
    """

    descriptions = [
        (1, "Paper baseline", "midpoint", "ending_cash", "starting_cash"),
        (2, "Marked portfolio equity", "midpoint", "ending_equity", "starting_equity"),
        (3, "Same-day bid-ask", "same_day_bid_ask", "ending_equity", "starting_equity"),
        (4, "Next-day bid-ask", "next_day_bid_ask", "ending_equity", "starting_equity"),
        (5, "Mid-signal, liquid next-day", "mid_signal_liquid_next_day", "ending_equity", "starting_equity"),
        (6, "Bid-ask robust signal, liquid next-day", "strong_signal_liquid_next_day", "ending_equity", "starting_equity"),
    ]
    tables = []
    for stage, description, run_name, value, starting_value in descriptions:
        daily = pl.read_csv(
            Path(output_folder) / run_name / "daily.csv", try_parse_dates=True
        )
        tables.append(_statistics(
            daily, value, starting_value, start_year, end_year
        ).with_columns(
            pl.lit(stage).alias("stage"),
            pl.lit(description).alias("specification"),
        ))
    return pl.concat(tables).sort(["stage", "cp_flag", "denomination"])


def make_strong_signal_table(output_folder):
    """
    Aggregate the paper-style same-day bid-ask robust signal test over 2014-2022.

    Signals are detected and priced at the same close, positions are fractional
    and no capacity limit is imposed. The table reproduces the
    paper's trade-level diagnostic.
    """

    folder = Path(output_folder) / "strong_signal_same_day"
    trades = pl.read_csv(folder / "trade_summary.csv").filter(
        pl.col("denomination") == "T1"
    ).group_by("cp_flag").agg(
        pl.col("number_of_trades").sum().alias("priced_trades"),
        pl.col("mean_profit_on_gross").mean(),
        pl.col("hit_rate").mean(),
    )
    signals = pl.read_csv(folder / "signal_survival.csv").filter(
        pl.col("denomination") == "T1"
    ).select(
        "cp_flag", pl.col("signals_detected").alias("signals")
    )
    return signals.join(trades, on="cp_flag").sort("cp_flag")


def make_paper_comparison(daily_file, paper_file, output_folder):
    """
    Rebuild the two tables that compare the mid-price run with the paper.

    `paper_file` is the annual series transcribed from the published table.
    Returns and Sharpe ratios for our implementation are calculated directly
    from the mid-price daily cash path, using the same 2003-2022 years.
    """

    daily = pd.read_csv(daily_file, parse_dates=["date"])
    paper = pd.read_csv(paper_file)
    comparisons = []

    columns = {
        ("C", "T1"): "call_T1", ("C", "K2"): "call_K2",
        ("P", "T1"): "put_T1", ("P", "K1"): "put_K1",
    }
    for (cp_flag, denomination), prefix in columns.items():
        path = daily[
            (daily["cp_flag"] == cp_flag)
            & (daily["denomination"] == denomination)
        ].sort_values("date")

        # One month-end observation is enough to recover both annual returns
        # and each year's Sharpe ratio from the daily cash balance.
        month_end = path.set_index("date")["ending_cash"].resample("ME").last().dropna()
        previous = float(path["starting_cash"].iloc[0])
        monthly_rows = []
        for date, value in month_end.items():
            monthly_rows.append({"year": date.year, "return": value / previous - 1})
            previous = value
        monthly = pd.DataFrame(monthly_rows)

        our_rows = []
        previous = float(path["starting_cash"].iloc[0])
        for year, group in path.groupby(path["date"].dt.year):
            end = float(group["ending_cash"].iloc[-1])
            returns = monthly.loc[monthly["year"] == year, "return"]
            sharpe = (
                np.sqrt(12) * returns.mean() / returns.std(ddof=1)
                if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan
            )
            our_rows.append({
                "year": year, "our_return_percent": 100 * (end / previous - 1),
                "our_sharpe": sharpe,
            })
            previous = end

        joined = pd.DataFrame(our_rows).merge(
            paper[["year", f"{prefix}_return", f"{prefix}_sharpe"]], on="year"
        ).query("2003 <= year <= 2022")
        comparisons.append({
            "cp_flag": cp_flag, "denomination": denomination,
            "our_mean_return_percent": joined["our_return_percent"].mean(),
            "paper_mean_return_percent": joined[f"{prefix}_return"].mean(),
            "return_correlation": joined["our_return_percent"].corr(joined[f"{prefix}_return"]),
            "sharpe_correlation": joined["our_sharpe"].corr(joined[f"{prefix}_sharpe"]),
        })

    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(comparisons).sort(["cp_flag", "denomination"]).write_csv(
        output / "paper_replication_summary.csv"
    )

    # The paper headline table uses all published years.
    paper_rows = []
    for (cp_flag, denomination), prefix in columns.items():
        paper_rows.append({
            "option_type": "Calls" if cp_flag == "C" else "Puts",
            "denomination": denomination,
            "mean_annual_return_percent": paper[f"{prefix}_return"].mean(),
            "median_reported_sharpe": paper[f"{prefix}_sharpe"].median(),
        })
    pl.DataFrame(paper_rows).write_csv(output / "paper_reported_summary.csv")


def factor_regressions(daily_file, market_file, output_file):
    """
    Run the monthly factor regressions for the Stage 6 portfolio returns.

    The first model contains the monthly SPX log return. The second adds mean
    VIX, the log of aggregate monthly option volume and the mean quoted relative spread. Those
    three added variables are standardised within each strategy series.
    Coefficients are estimated by OLS and inference uses Newey-West standard
    errors with three monthly lags.
    """

    # Keep only the four headline portfolios reported in the thesis.
    daily = pd.read_csv(daily_file, parse_dates=["date"])
    run_label = (
        daily["run_name"].iloc[0]
        if "run_name" in daily.columns
        else Path(daily_file).parent.name
    )
    allowed = set(HEADLINE)
    daily = daily[[
        (cp_flag, denomination) in allowed
        for cp_flag, denomination in zip(daily["cp_flag"], daily["denomination"])
    ]]
    # Match each strategy date to the same day's SPX return and VIX close.
    market = pd.read_csv(market_file, parse_dates=["date"])
    daily = daily.merge(market[["date", "spx_return", "vix_close"]], on="date")
    rows = []

    for (cp_flag, denomination), group in daily.groupby(["cp_flag", "denomination"]):
        group = group.sort_values("date").set_index("date")
        # Convert daily strategy and market data to one observation per month.
        monthly = group.resample("MS").agg({
            "starting_equity": "first", "ending_equity": "last",
            "spx_return": lambda x: np.log1p(x.dropna()).sum(),
            "vix_close": "mean", "market_option_volume": "sum",
            "market_median_relative_spread": "mean",
        }).rename(columns={
            "spx_return": "spx_log_return",
            "vix_close": "mean_vix",
            "market_option_volume": "monthly_option_volume",
            "market_median_relative_spread": "mean_quoted_spread",
        }).dropna()
        # Use the previous month-end wealth, as in the thesis Sharpe calculation.
        previous_end = monthly["ending_equity"].shift(1)
        previous_end.iloc[0] = monthly["starting_equity"].iloc[0]
        monthly["previous_equity"] = previous_end
        monthly["monthly_return"] = monthly["ending_equity"] / previous_end - 1
        monthly["log_option_volume"] = np.log1p(monthly["monthly_option_volume"])
        # Standardise the non-return factors so coefficients are comparable.
        for name in ["mean_vix", "log_option_volume", "mean_quoted_spread"]:
            monthly[name + "_z"] = (monthly[name] - monthly[name].mean()) / monthly[name].std()

        models = {
            "market": ["spx_log_return"],
            "market_volatility_liquidity": [
                "spx_log_return", "mean_vix_z", "log_option_volume_z",
                "mean_quoted_spread_z",
            ],
        }
        # Estimate the market-only and full specifications separately.
        for model, factors in models.items():
            x = sm.add_constant(monthly[factors])
            fit = sm.OLS(monthly["monthly_return"], x).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True}
            )
            for factor in fit.params.index:
                factor_name = {
                    "const": "constant", "mean_vix_z": "vix_z",
                    "mean_quoted_spread_z": "quoted_spread_z",
                }.get(factor, factor)
                rows.append({
                    "run_name": run_label,
                    "cp_flag": cp_flag, "denomination": denomination,
                    "model": model, "factor": factor_name,
                    "coefficient": fit.params[factor],
                    "newey_west_standard_error": fit.bse[factor],
                    "t_statistic": fit.tvalues[factor],
                    "p_value": fit.pvalues[factor],
                    "observations": int(fit.nobs), "r_squared": fit.rsquared,
                    "newey_west_lag": 3,
                })
    pl.DataFrame(rows).write_csv(output_file)


def main():
    """Read command-line arguments, run selected specifications and save tables."""

    parser = argparse.ArgumentParser(description="Run the Chapter 5 backtests")
    parser.add_argument("data_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--spec", choices=list(SPECS) + ["all"], default="all")
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--forwards", required=True)
    parser.add_argument("--settlement-flags", required=True)
    parser.add_argument("--delta-exclusions", required=True)
    parser.add_argument("--settlement-prices", required=True)
    parser.add_argument(
        "--market-data",
        help="Optional daily SPX/VIX file used to rebuild the factor regressions",
    )
    parser.add_argument(
        "--paper-results",
        help="Annual results transcribed from the paper for the replication table",
    )
    args = parser.parse_args()

    # The strong same-day replication uses only the paper's 2014-2022 window.
    names = list(SPECS) if args.spec == "all" else [args.spec]
    for name in names:
        years = (
            range(max(args.start, 2014), min(args.end, 2022) + 1)
            if name == "strong_signal_same_day"
            else range(args.start, args.end + 1)
        )
        if not years:
            continue
        run_backtest(
            name, args.data_folder, args.output_folder, years,
            args.forwards, args.settlement_flags, args.delta_exclusions,
            args.settlement_prices,
        )

    # Once all simulations exist, build the compact tables used by the notebook.
    if args.spec == "all":
        output = Path(args.output_folder)
        if args.start <= 2003 and args.end >= 2022:
            make_backtest_table(output, 2003, 2022).write_csv(
                output / "backtest_2003_2022.csv"
            )
        if args.start <= 2023 and args.end >= 2025:
            make_backtest_table(output, 2023, 2025).write_csv(
                output / "backtest_2023_2025.csv"
            )
        if args.start <= 2022 and args.end >= 2014:
            make_strong_signal_table(output).write_csv(
                output / "strong_signal_same_day.csv"
            )
        if args.market_data:
            factor_regressions(
                output / "strong_signal_liquid_next_day" / "daily.csv",
                args.market_data,
                output / "factor_regressions.csv",
            )
        if args.paper_results:
            make_paper_comparison(
                output / "midpoint" / "daily.csv", args.paper_results, output,
            )


if __name__ == "__main__":
    main()
