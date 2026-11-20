"""Backtest for the trophy-rarity forecaster (synthetic launches only).

Generates simulated launches calibrated to the real anchors (638 players
tracked on launch night, 3,868 at the first refresh, ~63,000 a day later),
then scores main.py's forecast_case against the house persistence baseline
and the house curve baseline by mean pinball loss on the logit scale.

    python3 tools/backtest.py

World: players P(t) with log P quadratic in ln(1+t) fitted to the anchors
(tapered after 72h); earners E_i(t) by a cohort-delay convolution — buyers
arrive at rate P'(b) and earn trophy i after a LogNormal delay (median m_i,
sigma 0.9) with eventual share A_i. Six scenarios vary player growth,
earning speed and eventual shares. No live data is used here.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import forecast_case, logit

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2 * max(tau * d, (tau - 1) * d)


def score(forecast, answers):
    tot, n = 0.0, 0
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    for tid, y in answers.items():
        for tau, q in zip(LEVELS, rows[tid]):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def make_P(taper_f=0.5):
    a, b, c = math.log(638), 0.720, 0.198

    def logP(t_h):
        u = math.log1p(t_h)
        v = a + b * u + c * u * u
        if t_h > 72:
            u72 = math.log1p(72)
            v72 = a + b * u72 + c * u72 * u72
            v = v72 + taper_f * (b + 2 * c * u72) * (u - u72)
        return v

    return logP


def simulate(P_log, trophies, t_grid, dt_h=1.0):
    ts = [i * dt_h for i in range(int(t_grid[-1] / dt_h) + 1)]
    P = [math.exp(P_log(t)) for t in ts]
    dP = [P[0]] + [P[i] - P[i - 1] for i in range(1, len(ts))]
    reads = []
    for t in t_grid:
        E = []
        for (A, m) in trophies:
            s = 0.0
            for j, b in enumerate(ts):
                if b >= t:
                    break
                lag = t - b
                if lag <= 0:
                    continue
                s += dP[j] * phi((math.log(lag) - math.log(m)) / 0.9)
            E.append(A * s)
        Pt = math.exp(P_log(t))
        reads.append((t, Pt, [100.0 * e / Pt for e in E]))
    return reads


def to_case(reads, trophies_meta, at_h, target_h):
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    fmt = lambda h: (base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hist = []
    for (t, P, rates) in reads:
        if t <= at_h + 1e-9:
            hist.append({"slot": fmt(t), "at": fmt(t), "players_tracked": int(round(P)),
                         "trophies_earned": 0, "rates": [round(r, 2) for r in rates]})
    return {"schema": "makefun/gta6-trophies-case/v1", "id": "sim", "at": fmt(at_h),
            "horizon_minutes": (target_h - at_h) * 60, "target": fmt(target_h),
            "levels": LEVELS, "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
            "release_date": "2026-11-19", "source": ""},
            "trophies": trophies_meta, "history": hist, "notes": ""}


def persistence(case):
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        lg = logit(last["rates"][i] or 0.0)
        qs = []
        for z in Z:
            v = 100.0 / (1.0 + math.exp(-(lg + z * spread)))
            qs.append(round(v, 4))
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def curve(case):
    # House curve baseline: OLS logit(rate) ~ ln(1+hours), last 24h reads.
    from datetime import datetime, timezone
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    at = datetime.fromisoformat(case["at"].replace("Z", "+00:00"))
    at_h = (at - base).total_seconds() / 3600.0
    win = [r for r in case["history"]
           if at_h - (datetime.fromisoformat(r["at"].replace("Z", "+00:00")) - base).total_seconds() / 3600.0 <= 24.0]
    if len(win) < 3:
        win = case["history"]
    xs = [math.log1p(max(0.0, (datetime.fromisoformat(r["at"].replace("Z", "+00:00")) - base).total_seconds() / 3600.0)) for r in win]
    tgt = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
    xt = math.log1p(max(0.0, (tgt - base).total_seconds() / 3600.0))
    d = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        ys = [logit(r["rates"][i]) if r["rates"][i] is not None else None for r in win]
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if len(pairs) < 3:
            rows.append({"id": tr["id"], "q": persistence(case)["trophies"][i]["q"]})
            continue
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        sxx = sum((p[0] - mx) ** 2 for p in pairs)
        b = sum((p[0] - mx) * (p[1] - my) for p in pairs) / sxx if sxx else 0.0
        a = my - b * mx
        sd = math.sqrt(sum((p[1] - (a + b * p[0])) ** 2 for p in pairs) / max(1, n - 2)) if n > 2 else 0.0
        pv = 1 + 1 / n + (xt - mx) ** 2 / sxx if sxx else 2.0
        s = math.sqrt(sd * sd * pv + rw * rw)
        pred = a + b * xt
        qs = [round(100.0 / (1.0 + math.exp(-(pred + z * s))), 4) for z in Z]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    kit = Path(__file__).resolve().parent.parent.parent / "gta6-trophy-rarity-kit" / "gta6-trophy-rarity-forecast-01-kit"
    meta = json.loads((kit / "input" / "p00.in").read_text())
    tm = meta["trophies"]
    params = []
    for i, tr in enumerate(tm):
        if i == 1:
            params.append((0.999, 1.0))
        elif "Chapter 1" in tr["description"] or "Paradise Lost" in tr["name"]:
            params.append((0.45, 6.0))
        elif "Chapter 2" in tr["description"]:
            params.append((0.35, 14.0))
        elif "Chapter 3" in tr["description"]:
            params.append((0.28, 26.0))
        elif "Chapter 4" in tr["description"]:
            params.append((0.22, 40.0))
        elif "final mission" in tr["description"].lower():
            params.append((0.12, 90.0))
        elif tr["tier"] == "platinum":
            params.append((0.004, 400.0))
        elif tr["tier"] == "gold":
            params.append((0.01, 300.0))
        elif tr["tier"] == "silver":
            params.append((0.02, 200.0))
        elif "fishing" in tr["description"].lower() or "Beat Cal" in tr["description"]:
            params.append((0.25, 10.0))
        elif "arcade" in tr["description"].lower():
            params.append((0.15, 20.0))
        else:
            params.append((0.05, 120.0))
    scenarios = {
        "A base": (make_P(), params),
        "B slow P": (make_P(0.2), params),
        "C no taper": (make_P(1.0), params),
        "D fast earn": (make_P(), [(A, m / 2) for A, m in params]),
        "E slow earn": (make_P(), [(A, m * 2) for A, m in params]),
        "F low eventual": (make_P(), [(A / 2, m) for A, m in params]),
    }
    grid = [i / 2.0 for i in range(0, 121)]
    means = {"persistence": [], "curve": [], "structural": []}
    for name, (P_log, prm) in scenarios.items():
        reads = simulate(P_log, prm, grid)
        case = to_case(reads, tm, 60.0, 168.0)
        ans = simulate(P_log, prm, [168.0])
        answers = {tr["id"]: ans[0][2][i] for i, tr in enumerate(tm)}
        lp = score(persistence(case), answers)
        lc = score(curve(case), answers)
        ls = score(forecast_case(case, "sim"), answers)
        means["persistence"].append(lp)
        means["curve"].append(lc)
        means["structural"].append(ls)
        print(f"{name:14s} persistence={lp:.4f} curve={lc:.4f} structural={ls:.4f}")
    for k, v in means.items():
        print(f"mean {k:12s} {sum(v)/len(v):.4f}")


if __name__ == "__main__":
    main()
