# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hasanrdp/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hasanrdp/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every GTA VI trophy on PlayStation 5, its earn rate on Exophase at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Built for the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01). Python 3.13, standard library only, no network, deterministic (two runs byte-identical).

## The problem, in one ratio

An earn rate is a ratio, `r = E / N`: the tracked players who earned the trophy (E) over the players Exophase tracks (N). Exophase refreshes its page on its own slow clock, so the recorded history is a step function: long flat stretches, then a jump. At a refresh N usually jumps a lot while E barely moves, so every rate falls by roughly `ln(N_new / N_old)` in logit terms. Launch night proved it: the page sat unchanged from 23:48Z to 04:00Z at 638 players, refreshed at 04:30Z to 3,868 players (x6.06), and every rate except the 100% one fell, by about 1.7 logits on average, while earner counts barely moved (most 4-8 -> 4-9).

Over the final's 4.5-day horizon the two forces reverse: N growth slows as the launch surge passes, while E growth accelerates as players work through the story. Early trophies' rates recover; late trophies' keep falling. A forecaster that only extrapolates the rate misses the saturation; one that only models E/N misses how slowly earners arrive.

## Method

The forecast is the equal-weight mean, on the logit scale, of four members with deliberately opposing biases, so their errors partly cancel:

1. **Saturating logistic in log-time**, `r = L / (1 + exp(-(a + b ln(1+t))))`, fitted on the last 24 hours of distinct reads, asymptote `L` on a coarse grid (5, 10, 20, 35, 50, 70, 90, 99.99). The asymptote is the point: a rate cannot keep rising on a log-time line forever, and the house curve baseline has no asymptote and overshoots. The coarse grid regularises -- a fine grid fits noise and does worse.
2. **The same saturating logistic on the last 12 hours with a fine grid**: a second opinion on the recent slope and asymptote.
3. **A point-lag earners-over-players model**, `E(t) = s * N(t - tau)`, with the share `s` capped at 1 and `N` from a power-law / saturating average. It is the member that understands dilution: when N jumps, rates fall. On its own it underforecasts (players earn on a wide spread of schedules, not one lag) -- which is why it belongs in a mean with member 4.
4. **The house curve baseline's own shape**, `logit(r)` linear in `ln(1+t)` over the last 24 hours, with no slope cap. On its own it overshoots -- its slope runs to 3 or 4 on story trophies -- which is why it belongs in a mean with member 3. Capping the slope was tried and made the ensemble much worse (the median's mean |logit error| went from 0.56 to 1.37), so the slope is left uncapped; the member's logit is only clamped to [-6, 6] as a safety net.

The spread is a split normal on the logit scale. Its width combines the saturating-logistic fit's residuals, the disagreement between the four members (divided by 4, not 2 -- the members' errors cancel, so the disagreement overstates the mean's error), the uncertainty of the N extrapolation, a horizon term (0.25 logit sd per root-day), a staleness term, and -- only on the downside -- a refresh-mixture term: while the page is stale a refresh may or may not land before the target, and a refresh can only push rates down. When the history is flat and the horizon is short (the practice cases) that term dominates the lower tail, which is the honest shape: the median stays at the last rate, the 5% quantile reaches down toward the post-refresh level. The 100% trophy ("Just Like Paradise") gets a fixed row near 100%.

## Local validation

Practice cases, scored against the live Exophase page after its 04:30Z refresh (N = 3,868):

| case | this entry | starter baseline |
|------|-----------|------------------|
| p00  | 1.0697    | 1.5133           |
| p01  | 1.0388    | 1.5133           |

(The practice histories are flat -- one distinct read -- so every member falls back to the last rate and only the spread speaks; the gain comes from the asymmetric lower tail.)

Final-like cases, scored on synthetic launch worlds (agent-based: players join on a growing curve, each with a log-normal pace, each trophy with a time-to-earn and an eventual share; the page frozen between Exophase refreshes; checkpoint at the entry deadline, target 4.5 days out):

| model | mean pinball loss, 6 worlds | mean pinball loss, 9 worlds |
|-------|----------------------------|-----------------------------|
| this entry | **0.36** | **0.38** |
| house curve baseline | 0.90 | -- |
| persistence (last rate) | 1.15 | -- |

The entry's median misses the answer by 0.50 logits on average across the 9 worlds, against 1.55 for the house curve baseline and 1.80 for persistence. Runtime is about 0.5-1.3 s per case, well inside the 8 s limit.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes `<id>.out` for each. No dependencies beyond Python 3.13's standard library.

## Files

- `main.py` -- the forecaster (the whole entry; see its header comment for the full method and the launch-night evidence behind the constants)
- `README.md` -- this file
- `LICENSE` -- MIT
- `.makefun/owner.json` -- payout wallet for the bounty
