# gta6-trophy-rarity-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dhnhatdng/gta6-trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dhnhatdng/gta6-trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players will hold each PlayStation 5 trophy a week after launch, from Exophase's earn-rate reads. One program, `main.py` (Python 3.13, standard library only, no network): it reads a case file, and writes five quantiles (5/25/50/75/95%) of every trophy's earn rate at the case's target.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it forecasts

An earn rate is a ratio — earners over players tracked — and over launch week the two halves pull in opposite directions: new buyers flood in and dilute every rate, while the players already in keep earning as they work through the story. The forecast blends two trend models, both fitted on the case's own history:

- **Structural (70%):** log earners and log players tracked are each fitted against log time since release, slopes shrunk 0.8× and re-centred. The median starts at the trophy's last read rate and moves only by the fitted change in log(earners/players) — anchoring to the last read matters because rates print to 2 decimals, so a rare trophy at 0.03% carries ~17% rounding noise.
- **Curve (30%):** a straight OLS line of logit(rate) against log time over the last 24 hours of reads, the steadier model when the earners/players split misbehaves.

The blend is per quantile in logit space. Spreads combine each fit's residual with its prediction variance (wider when the target extrapolates) plus a small random-walk floor.

## Backtest

`tools/backtest.py` builds synthetic launches calibrated to the real anchors (638 players tracked on launch night, 3,868 at the first refresh, ~63,000 a day later) and scores by mean pinball loss on the logit scale. Across six simulated worlds — slow/fast player growth, slow/fast earning, low eventual shares — this forecast averages **0.235**, against **0.312** for the curve baseline alone and **1.220** for persistence, winning 4 of 6 worlds outright.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fs4ajvvm&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
