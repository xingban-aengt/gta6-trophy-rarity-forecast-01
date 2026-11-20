"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Model (original work, standard library only):
  * Collapse the case history into distinct page states ("platforms"): Exophase
    refreshes a game's counts on its own schedule, so long stretches of reads
    show the identical page. One platform = one distinct state, dated at its
    first read. Fitting on platforms, not raw reads, stops a long frozen
    stretch from outweighing the states that actually moved.
  * For each trophy, fit logit(rate) against ln(1 + hours since
    2026-11-19T00:00:00Z) with a Theil-Sen line (median pairwise slope), which
    a single refresh jump cannot drag the way least squares can. When the
    last 24 h of platforms hold at least 3 states, average the full-history
    slope with the recent-window slope, so the launch-night regime does not
    dominate the forecast. Two states give a half-shrunk slope; one state
    gives a flat line at the last rate.
  * Shrink each trophy's slope a little toward its tier's median slope
    (weight falls as the trophy's own fit gains platforms), and cap the move
    from the last read, so a noisy fit cannot extrapolate wildly. The level
    stays anchored on the trophy's own last rate: a flat history forecasts
    flat, it does not drift toward its tier.
  * The trophy every tracked player holds (rate 100%) gets a tight band just
    under 100: it survived a sixfold player jump at 100%, so it is a
    play-the-game trophy, but a sliver of doubt is cheaper than certainty.
  * Spread on the logit scale is honest, not tuned: a random-walk term growing
    with the square root of the horizon, plus the fit's residual scale (with a
    leverage factor for how far the target sits past the data), plus a refresh
    term -- large when history shows a single frozen page (the next refresh is
    unseen and can move every rate at once), smaller once refreshes are in the
    data. No randomness anywhere: two runs write identical bytes.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out per case
to $MAKEFUN_OUTPUT_DIR (else /output): {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}, ...]} with quantiles in percent, non-decreasing, 0..100.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels, in order.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
# Launch instant the log-time axis is measured from (house curve baseline's).
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Random-walk spread, logit units per square-root day.
WALK_SIGMA = 0.5
# Extra spread for a refresh nobody has seen yet (one frozen page in history).
# Evidence: the house baseline's published practice numbers (loss ~1.51,
# typical miss ~0.68 pp) imply the first real refresh moved typical rates by
# about -1.8 logit -- near-pure dilution as the tracked base jumped 638 ->
# 3,868. A 1.2 logit sigma puts the 5% quantile just past that move.
UNSEEN_REFRESH_SIGMA = 1.2
# Extra spread once refreshes are already in the history.
SEEN_REFRESH_SIGMA = 0.35
# Hard cap on how far a forecast may move from the last read, in logit units.
MAX_MOVE = 3.0
# Platinum cannot be earned fast: cap its upward move tighter.
PLATINUM_MAX_UP = 1.5
# Theil-Sen pairs grow quadratically; thin the platform list past this many.
MAX_PLATFORMS = 100


def _parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _logit(pct):
    """logit of a percent, clamped to the 0.01..99.99 range the scorer uses."""
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(logit_value):
    """Inverse logit, as a percent."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def _round4(value):
    """Round to 4 decimals, halves away from zero (matches the house style)."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if float(rounded).is_integer() else rounded


def _median(values):
    """Median of a non-empty list of numbers (deterministic)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _platforms(case):
    """Collapse history into distinct page states.

    Returns a list of (x, players, rates) with x = ln(1 + hours since EPOCH)
    at the state's first read, in time order. Consecutive reads showing the
    identical page count once; a state that returns later counts again, since
    its reappearance is a fresh observation of the same numbers.
    """
    states = []
    for read in case.get("history") or []:
        rates = tuple(read.get("rates") or [])
        players = read.get("players_tracked")
        if states and states[-1][1] == players and states[-1][2] == rates:
            continue
        moment = _parse_time(read["at"])
        hours = (moment - EPOCH).total_seconds() / 3600.0
        states.append((math.log1p(max(hours, -0.999)), players, rates))
    return states


def _theil_sen(points):
    """Median pairwise slope and median intercept for [(x, y), ...]."""
    slopes = []
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if xj != xi:
                slopes.append((yj - yi) / (xj - xi))
    if not slopes:
        return 0.0, _median([y for _, y in points])
    slope = _median(slopes)
    intercept = _median([y - slope * x for x, y in points])
    return slope, intercept


def _fit_trophy(series, x_target, horizon_days):
    """Robust trend fit for one trophy's [(x, logit rate)] platform series.

    Returns (slope, spread): the Theil-Sen slope of logit rate against
    log-time, and the spread in logit units. The intercept is left to the
    caller, which may blend the slope with its tier's before anchoring.
    """
    last_y = series[-1][1]
    n = len(series)
    if n == 1:
        # One frozen page: no trend is visible; the spread must carry the
        # whole unseen-refresh risk.
        return 0.0, math.sqrt((WALK_SIGMA * math.sqrt(horizon_days)) ** 2
                              + UNSEEN_REFRESH_SIGMA ** 2)
    use = series
    if n > MAX_PLATFORMS:
        step = (n - 1) / (MAX_PLATFORMS - 1)
        use = [series[int(round(k * step))] for k in range(MAX_PLATFORMS)]
    slope, _ = _theil_sen(use)
    if n == 2:
        slope *= 0.5
    else:
        # Blend in the recent window's slope when it has enough states.
        # x is log-time, so compare in hours: a state is "recent" when its
        # hour count is within 24 of the last state's.
        last_hours = math.expm1(use[-1][0])
        recent = [pt for pt in use if math.expm1(pt[0]) >= last_hours - 24.0]
        if len(recent) >= 3:
            recent_slope, _ = _theil_sen(recent)
            slope = 0.5 * slope + 0.5 * recent_slope
    # Residual scale of the fit, with a leverage factor for extrapolating
    # past the data; floored so a perfectly collinear fit still admits doubt.
    intercept = _median([y - slope * x for x, y in use])
    residuals = [abs(y - (intercept + slope * x)) for x, y in use]
    mad = _median(residuals)
    x_bar = sum(x for x, _ in use) / len(use)
    denom = sum((x - x_bar) ** 2 for x, _ in use)
    leverage = 1.0
    if denom > 0:
        leverage = 1.0 + (x_target - x_bar) ** 2 / denom
    fit_sigma = max(0.15, 1.4826 * mad) * math.sqrt(min(leverage, 4.0))
    spread = math.sqrt((WALK_SIGMA * math.sqrt(horizon_days)) ** 2
                       + fit_sigma ** 2 + SEEN_REFRESH_SIGMA ** 2)
    return slope, min(spread, 2.5)


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    trophies = case["trophies"]
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    target = _parse_time(case["target"])
    x_target = math.log1p(max((target - EPOCH).total_seconds() / 3600.0, -0.999))
    states = _platforms(case)

    # Last non-null rate per trophy; 1% with a wide band when nothing was seen.
    last_rates = []
    for i in range(len(trophies)):
        seen = [rates[i] for _, _, rates in states
                if i < len(rates) and rates[i] is not None]
        last_rates.append(seen[-1] if seen else None)

    series_of = []
    slopes = []
    raw = []  # (predicted logit, spread) per trophy, before tier shrinkage.
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None or rate >= 100.0:
            series_of.append([])
            slopes.append(0.0)
            raw.append((_logit(99.99) if rate is not None else _logit(1.0),
                        0.0 if rate is not None else 2.0))
            continue
        series = []
        for x, _, rates in states:
            if i < len(rates) and rates[i] is not None:
                series.append((x, _logit(rates[i])))
        series_of.append(series)
        if not series:
            slopes.append(0.0)
            raw.append((_logit(1.0), 2.0))
            continue
        slope, spread = _fit_trophy(series, x_target, horizon_days)
        slopes.append(slope)
        raw.append((None, spread))

    # Gentle hierarchical shrinkage on the slope, not the level: trophies in
    # a tier share progression dynamics, so a trophy whose own fit saw few
    # states borrows its tier's median slope. The level stays anchored on
    # the trophy's own last rate -- a flat history forecasts flat.
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier")
        if tier not in tier_slopes:
            members = [j for j, t in enumerate(trophies)
                       if t.get("tier") == tier and series_of[j]]
            tier_slopes[tier] = _median([slopes[j] for j in members]) \
                if members else 0.0
    for i, trophy in enumerate(trophies):
        if raw[i][0] is not None:
            continue
        series = series_of[i]
        n_states = len(series)
        weight = 0.15 / (1.0 + n_states / 4.0)
        slope = slopes[i] * (1.0 - weight) + tier_slopes[trophy.get("tier")] * weight
        intercept = _median([y - slope * x for x, y in series])
        predicted = intercept + slope * x_target
        last_y = _logit(last_rates[i])
        move = predicted - last_y
        cap_up = PLATINUM_MAX_UP if trophy.get("tier") == "platinum" else MAX_MOVE
        move = min(max(move, -MAX_MOVE), cap_up)
        raw[i] = (last_y + move, raw[i][1])

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        pred, spread = raw[i]
        if rate is not None and rate >= 100.0:
            # Stays ~100: the band below costs little if it holds, and halves
            # the damage if a few tracked players turn out not to hold it.
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 99.99, 100, 100]})
            continue
        quantiles = []
        for z in Z_SCORES:
            q = _sigmoid(pred + z * spread)
            quantiles.append(_round4(min(99.99, max(0.01, q))))
        # Rounding can only tie neighbours, never invert them; enforce anyway.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
