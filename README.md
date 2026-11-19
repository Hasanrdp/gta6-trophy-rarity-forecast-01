# GTA 6 Trophy Rarity Forecast — v2

[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hasanrdp/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hasanrdp/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast how many GTA VI players earn each trophy in week one. `main.py` (Python 3.13, standard library only, no network, deterministic) reads every `<id>.in` in the input directory and writes `<id>.out` — five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's earn rate at the case's target, scored by mean pinball loss on the logit scale.

## Why this is hard, and what the entry does about it

An earn rate is a ratio, r = E / N: earners over players tracked. Exophase refreshes its trophy page on its own slow clock, so a recorded history is a step function: long flat stretches, then a jump. At a refresh N usually jumps while E barely moves, so every rate falls. Launch night proved it (kit RULES.txt): the page sat unchanged from 23:48Z to 04:00Z at 638 players, refreshed by 04:30Z to 3,868 players (×6.06, ln 1.802), and every rate but the 100% one fell. A live snapshot of the same page, fetched 2026-11-19T13:36Z during this build, shows the same force at ten times the scale: most rates fell 2.5–3.2 logits from launch night (mean 2.61) while the two earliest trophies (first fishing, Chapter 1) barely moved — new players earn the first trophies immediately, the rest only later.

The entry therefore works in two regimes, chosen by how much the history actually shows:

- **Flat history (at most 2 distinct reads — the practice cases, and any case whose page never refreshed):** no trend can be fitted, so the median stays at the last rate and the forecast is a refresh mixture. With probability p = 1 − exp(−H / τ) a refresh lands before the target (H = horizon hours, τ = the refresh interval, 4.7 h from launch night when the history shows none); if it lands, the rate drops by D = jump × √(max(1, H / τ)) logits, where jump is the median log N-jump between distinct reads (1.802, the launch-night jump, when the history shows none). The drop is tapered per trophy by (1 − r^50), where r is the last rate as a fraction: ≈1 below 95%, ≈0 at 100% — a saturated trophy's earners grow with the player base, so its rate does not fall at a refresh (observed: "Just Like Paradise" stayed at 100%, actual logit drop 0.00, while the other 54 fell ~2.6). Quantiles come from the mixture by bisection on its CDF. This is the honest shape for a stale page: the median drops only as far as a refresh is likely, the lower tail reaches toward the post-refresh level, the upper tail stays near the last rate.
- **Rich history (3+ distinct reads — the final case):** three trend members, each fitted on the distinct reads, are averaged on the logit scale with equal weights, and the spread combines their residual scales with their disagreement: (a) a logit-linear trend in ln(1 + hours since release) over the last 24 h (the house curve baseline's own shape, slope uncapped); (b) a saturating logistic in ln(1 + hours), asymptote on a coarse grid (5, 10, 20, 35, 50, 70, 90, 99.99) — a rate cannot rise on a log-time line forever; (c) an E/N decomposition — log-linear earners and log-linear players tracked in ln(1 + hours), rate = E / N at the target — the member that understands dilution. Persistence is only the fallback when no trend member can fit. No refresh mixture is added on top: the trends already model the movement, and the spread (member disagreement + residuals + a 0.5·√days horizon term) covers the refresh timing.

## Validation — real data only

Two holdout backtests, built from real histories, not invented worlds. Each takes the flat launch-night history (the kit's practice cases) and scores the forecast against the live 13:36Z snapshot of the same page (55 trophies each; horizons 636 and 606 minutes, ~10.5 h):

| backtest | horizon | this entry | house persistence | house curve (practice) |
|---|---|---|---|---|
| L0 (from p00) | 10.6 h | **0.2907** | 2.3827 | ~1.51 |
| L1 (from p01) | 10.1 h | **0.3107** | 2.3882 | ~1.51 |
| mean | | **0.3007** | 2.3854 | ~1.51 |

Mean pinball loss on the logit scale (lower is better). The entry beats the house persistence baseline by 7.9× on real data. The kit's two practice cases have flat histories and no scoreable answer of their own, so they are not in this table — the backtests above are the validation, and they use the practice histories as their starting point.

Design choices were tested both ways on these backtests, and the losers were dropped:

- A plain random-walk spread around persistence (σ = 0.5·√days) scores 2.3854 — the drop is not noise, it is a one-sided refresh.
- A constant drop D = 1.802 (the launch-night jump) scores 0.8427; a linear D in the horizon scores 0.8237; the √H law scores 0.3265 (0.3007 with the saturated-trophy taper below). The drop grows with the horizon, but slower than linearly.
- Refresh interval τ = 3.0 h scores 0.3319, τ = 4.7 h scores 0.3265, τ = 8.0 h scores 0.4690 — the kit-measured ~4.7 h is best.
- A per-trophy drop was rejected: the drop's correlation with the last rate is only −0.50, and the pattern is three outliers (the earliest trophies) against a tight cluster (2.4–3.2 logits for everything else). A global D is near-optimal; fitting per-trophy rules to three points would be overfitting.
- The saturated-trophy taper (1 − r^50) is the one per-trophy correction kept: it improves the mean from 0.3288 to 0.3007 on both backtest cuts, and the gain comes from a single trophy ("Just Like Paradise", 100%). The mechanism is forced, not fitted — a 100% trophy's earners grow with N by definition — though the middle of the taper (95–100%) is uncalibrated: no trophy in the data sits there.

## What is NOT validated — read this before trusting the entry

- **The rich-regime ensemble has never been backtested on real data.** No multi-refresh history exists before the deadline: a re-fetch of the live page ~30 minutes after the 13:36Z snapshot returned the same cached snapshot, so there is no third state to fit a trend on, let alone a fourth to score it against. The three members are principled (they mirror the house curve, a saturating asymptote, and the E/N mechanism) but unproven; the README says so because v1 claimed synthetic-world validation it did not have.
- **The 4.5-day final horizon is a 10× extrapolation of the ~10 h backtest horizon.** The D(H) = jump·√(H/τ) law is a principled guess (refreshes accumulate like a random walk), not a measured law. If Exophase refreshes faster or slower than 4.7 h over the final window, the median will be off in that direction.
- **The backtest covers only two cuts.** Both start from the same flat history and end at the same snapshot, so they are two windows on one event, not two independent events. The 0.30 number is the best real evidence available, not a guarantee.

## Determinism and runtime

Standard library only, no network, no randomness. Two runs on the practice cases are byte-identical. Runtime ~0.05 s per case (limit 8 s). Edge cases stress-tested: single-read history, empty history, the 100% trophy (it is not special-cased — the general machinery lets it fall, as it did in reality: 100% → 99.99%), flat histories, big N jumps, and a 6,510-minute horizon.

## Run it

```sh
python3 -P -s main.py   # reads /input/*.in, writes /output/*.out
```

With `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` set, those directories are used instead. Output format: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, quantiles non-decreasing, each in [0, 100], rounded to 4 decimals (half up).
