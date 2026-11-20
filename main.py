"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model: a blend of two trend forecasts, both fitted on the case's own
history only, blended per quantile in logit space (70% structural, 30%
curve). The blend was chosen on simulation, not on the practice cases:
the practice histories are flat (one refresh), and the rules say one
refresh over 90 minutes is little guide to the final's week.

Why two models? An Exophase earn rate is a ratio, rate_i(t) = E_i(t)/P(t):
players tracked P(t) floods in after launch (diluting every rate) while
earners E_i(t) grow as cohorts age through the story. The two components
move in opposite directions for the rate, so:

1. Structural (70%): fit log E_i(t) and log P(t) separately against
   x = ln(1 + hours since release) over a recency window (last max(36h,
   half the history span)), slopes shrunk x0.8 and re-centred at the
   window's mean x. The forecast is LEVEL-ANCHORED: the median starts at
   the trophy's last read rate and moves by the fitted change in
   log(E/P) from the last read to the target. Anchoring matters because
   rates print to 2 decimals: for a rare trophy at 0.03%, rounding is
   ~17% relative noise, and a ratio of two extrapolated quantities drifts
   off its level; the last read is the most trustworthy level.
   Spread: s^2 = sd_E^2 * predvar_E + sd_P^2 * predvar_P
   + (0.15 * sqrt(horizon_days))^2, floored so s >= 0.2; the prediction
   variance widens honestly when the target extrapolates beyond the reads.

2. Curve (30%): a straight OLS line of logit(rate) against ln(1+hours)
   over the case's last 24 hours of reads — the same family as the house
   curve baseline, implemented here from scratch — with spread from its
   residual and the same random-walk floor. It is the steadier of the two
   when the earners/players split misbehaves, so it hedges the blend.

Quantiles are lognormal on the logit scale at the case's levels.
Fallbacks: fewer than 3 reads, or a trophy with fewer than 3 usable
pairs, falls back to the house persistence baseline (last rate,
random-walk spread 0.5 * sqrt(days)) for that trophy. Rates of 100%
(the first trophy everyone earns) are clamped to 99.99% before the
logit, as the grader does.

Validation (tools/backtest.py, synthetic launches calibrated to the real
anchors: 638 players tracked on launch night, 3,868 at the first refresh,
~63,000 a day later from the live Exophase page): across six simulated
worlds (slow/fast player growth, slow/fast earning, low eventual shares)
the blend scores a mean pinball loss of 0.253, against 0.324 for the
curve baseline alone and 1.198 for persistence — beating the curve in
5 of 6 worlds, with the worst world (0.391) no worse than the curve's
worst (0.399). The pure structural model was rejected: on 2-decimal
rates it is the most volatile (mean 0.392, worst 0.661).

Runs on Python 3.13, standard library only, no network, deterministic:
reads every *.in in MAKEFUN_INPUT_DIR (/input), writes one *.out per case
to MAKEFUN_OUTPUT_DIR (/output), byte-identical across runs.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Slope shrinkage for the log-time fits (tuned on simulation; 0.8 matches
# the damping that won our sibling GTA VI Twitch forecaster's backtest).
SHRINK = 0.8
# Random-walk spread floor, in logit units per square-root day.
RW_SIGMA = 0.15
# The house persistence baseline's spread, used for fallbacks.
BASE_SIGMA = 0.5
# Recency window for the structural fit: last 36h of reads, or the last
# half of the history, whichever is more.
MIN_WINDOW_HOURS = 36.0
# Recency window for the curve fit: the last 24h of reads.
CURVE_WINDOW_HOURS = 24.0
# Blend weight on the structural model (logit space, per quantile).
STRUCTURAL_WEIGHT = 0.7


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, n, mean_x, sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0, n, mx, sxx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, n, mx, sxx


def shrunk_fit(xs, ys, xt):
    """OLS with slope shrunk SHRINK-fold and re-centred at mean x.

    Returns (prediction at xt, prediction-variance multiplier, resid_sd, n).
    The multiplier is 1 + 1/n + (xt - mx)^2 / sxx: the OLS prediction variance.
    """
    a, b, sd, n, mx, sxx = ols(xs, ys)
    b_s = b * SHRINK
    a_s = (a + b * mx) - b_s * mx
    pred = a_s + b_s * xt
    pv = 1 + 1 / n + (xt - mx) ** 2 / sxx if sxx > 0 else 2.0
    return pred, pv, sd, n


def persistence_row(last_rate, horizon_days):
    """The house baseline row for one trophy: last rate, random-walk spread."""
    lg = logit(last_rate if last_rate is not None else 0.0)
    spread = BASE_SIGMA * math.sqrt(horizon_days)
    return [round4(inv_logit(lg + z * spread)) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    if levels != LEVELS:
        raise ValueError(f"{name} asks for levels {levels}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    # Release instant: the game's release date when given, else the first read.
    game = case.get("game") or {}
    rel_date = game.get("release_date")
    if rel_date:
        release = parse_time(rel_date + "T00:00:00Z")
    else:
        release = parse_time(history[0]["at"])
    xt = math.log1p(max(0.0, hours_between(release, target)))

    # Reads with usable times, oldest first.
    reads = []
    for r in history:
        try:
            t = parse_time(r["at"])
        except (KeyError, ValueError):
            continue
        players = r.get("players_tracked")
        rates = r.get("rates") or []
        if players is None or players <= 0 or len(rates) != len(trophies):
            continue
        reads.append((hours_between(release, t), float(players), rates))
    if not reads:
        raise ValueError(f"{name} has no usable reads")

    at_h = hours_between(release, at)
    last_rates = reads[-1][2]
    rw_var = (RW_SIGMA * math.sqrt(horizon_days)) ** 2

    # --- Structural fit: log E and log P on log time, recency window. ---
    span = at_h - reads[0][0]
    win_h = max(MIN_WINDOW_HOURS, 0.5 * span)
    win = [r for r in reads if at_h - r[0] <= win_h]
    if len(win) < 3:
        win = reads
    xs = [math.log1p(max(0.0, t)) for (t, _, _) in win]
    x_last = xs[-1]
    log_p_t, p_pv, p_sd, _ = shrunk_fit(xs, [math.log(p) for (_, p, _) in win], xt)
    log_p_last, _, _, _ = shrunk_fit(xs, [math.log(p) for (_, p, _) in win], x_last)

    # --- Curve fit: logit(rate) on log time, last 24h of reads. ---
    cwin = [r for r in reads if at_h - r[0] <= CURVE_WINDOW_HOURS]
    if len(cwin) < 3:
        cwin = reads
    cxs = [math.log1p(max(0.0, t)) for (t, _, _) in cwin]

    rows = []
    for i, trophy in enumerate(trophies):
        # Structural row (level-anchored).
        pairs = []
        for k, (t, players, rates) in enumerate(win):
            rate = rates[i]
            if rate is None:
                continue
            pairs.append((xs[k], math.log(max(0.5, rate / 100.0 * players))))
        if len(pairs) < 3:
            rows.append({"id": trophy["id"], "q": persistence_row(last_rates[i], horizon_days)})
            continue
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        log_e_t, e_pv, e_sd, _ = shrunk_fit(px, py, xt)
        log_e_last, _, _, _ = shrunk_fit(px, py, x_last)
        var = (e_sd * e_sd) * e_pv + (p_sd * p_sd) * p_pv + rw_var
        s_struct = math.sqrt(max(var, 0.04))
        change = (log_e_t - log_p_t) - (log_e_last - log_p_last)
        lg_struct = logit(last_rates[i] if last_rates[i] is not None else 0.0) + change
        q_struct = [inv_logit(lg_struct + z * s_struct) for z in Z]

        # Curve row.
        cpairs = []
        for k, (t, players, rates) in enumerate(cwin):
            rate = rates[i]
            if rate is None:
                continue
            cpairs.append((cxs[k], logit(rate)))
        if len(cpairs) < 3:
            q_curve = [inv_logit(logit(last_rates[i] if last_rates[i] is not None else 0.0)
                        + z * BASE_SIGMA * math.sqrt(horizon_days)) for z in Z]
        else:
            cx = [p[0] for p in cpairs]
            cy = [p[1] for p in cpairs]
            a_c, b_c, sd_c, n_c, mx_c, sxx_c = ols(cx, cy)
            pred_c = a_c + b_c * xt
            pv_c = 1 + 1 / n_c + (xt - mx_c) ** 2 / sxx_c if sxx_c > 0 else 2.0
            s_curve = math.sqrt((sd_c * sd_c) * pv_c + rw_var)
            q_curve = [inv_logit(pred_c + z * s_curve) for z in Z]

        # Blend in logit space, per quantile; enforce non-decreasing.
        qs = []
        for k in range(5):
            lg = (STRUCTURAL_WEIGHT * logit(q_struct[k])
                  + (1 - STRUCTURAL_WEIGHT) * logit(q_curve[k]))
            qs.append(round4(inv_logit(lg)))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
