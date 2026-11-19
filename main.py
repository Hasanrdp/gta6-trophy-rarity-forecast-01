"""gta6-trophy-rarity-forecast-01 entry (v2): GTA VI trophy earn rates on Exophase.

Forecasts, for every trophy, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its
earn rate at the case's target. Python 3.13, standard library only, no network,
deterministic (two runs byte-identical), ~1 s per case.

The rate is a ratio, r = E / N: earners over players tracked. Exophase refreshes
its page on its own slow clock, so a recorded history is a step function: long
flat stretches, then a jump. At a refresh N usually jumps while E barely moves,
so every rate falls. Launch night proved it (kit RULES.txt): the page sat
unchanged from 23:48Z to 04:00Z at 638 players, refreshed by 04:30Z to 3,868
players (x6.06, ln 1.802), and every rate but the 100% one fell. A second real
snapshot, the live page fetched 2026-11-19T13:36Z during this build, shows the
same force at 10x the scale: most rates fell 2.5-3.2 logits from launch night
(mean 2.61) while the two earliest trophies (first fishing, Chapter 1) barely
moved -- new players earn the first trophies immediately, the rest only later.

Method (two regimes, chosen by how much the history actually shows):

FLAT history (at most 2 distinct reads -- the practice cases, and any case
whose page never refreshed): no trend can be fitted, so the median stays at
the last rate and the forecast is a refresh mixture. With probability
p = 1 - exp(-H / tau) a refresh lands before the target (H = horizon hours,
tau = the refresh interval, 4.7 h from launch night when the history shows
none); if it lands, the rate drops by D = jump * sqrt(max(1, H / tau)) logits,
where jump is the median log N-jump between distinct reads (1.802, the
launch-night jump, when the history shows none). The quantiles are the
mixture's, by bisection on its CDF. This is the honest shape for a stale page:
the median drops only as far as a refresh is likely, the lower tail reaches
toward the post-refresh level, the upper tail stays near the last rate.

RICH history (3+ distinct reads -- the final case): three trend members, each
fitted on the distinct reads, are averaged on the logit scale with equal
weights, and the spread combines their residual scales with their
disagreement:
  - trend: logit(rate) linear in ln(1 + hours since release), last 24 h
    (the house curve baseline's own shape, slope uncapped);
  - satlog: a saturating logistic in ln(1 + hours), asymptote on a coarse
    grid (5, 10, 20, 35, 50, 70, 90, 99.99) -- a rate cannot rise on a
    log-time line forever;
  - E/N: log-linear earners and log-linear players tracked in ln(1 + hours),
    rate = E / N at the target -- the member that understands dilution.
  Persistence is only the fallback when no trend member can fit. No refresh
  mixture is added on top: the trends already model the movement, and the
  spread (member disagreement + residuals + a 0.5*sqrt(days) horizon term)
  covers the refresh timing.

What was actually validated, on REAL data (see README): the flat-regime
mixture, on two backtests (flat launch-night history -> the live 13:36Z
snapshot, 55 trophies each, ~10 h horizon): mean pinball 0.33, against 2.39
for the house persistence baseline. The rich-regime ensemble is principled
but NOT backtest-validated: no multi-refresh history exists before the
deadline, so no real backtest can exercise it; the README says so plainly.
No synthetic worlds were used for validation anywhere.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. It reads every <id>.in in the input
directory and writes <id>.out to the output directory:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an import of
# a module shipped beside this file would find it (this entry ships none).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_RW = 0.5          # logit sd per root-day of the house baseline's spread
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
JUMP_PRIOR = 1.802      # ln(3868 / 638): the launch-night refresh's N jump
TAU_PRIOR = 4.7         # hours: page unchanged ~4.2 h, refreshed by 04:30Z
SATLOG_GRID = [5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0, 99.99]
SIGMA_FLOOR = 0.25


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as an
    int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(ts):
    return (parse_ts(ts) - RELEASE).total_seconds() / 3600.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd


def distinct_states(case):
    """Collapse consecutive identical reads to distinct states. Returns a list
    of (hours since release, players tracked, rates), oldest first, one per
    distinct page state, timed at the first read that showed it."""
    out = []
    prev = None
    for rd in case.get("history", []):
        rates = rd.get("rates") or []
        key = (rd.get("players_tracked"), tuple(rates))
        if key == prev:
            continue
        prev = key
        out.append((hours_since_release(rd["slot"]), rd.get("players_tracked") or 0, rates))
    return out


def last_rates(case, n_trophies):
    """The last read's rate per trophy, forward-filled over nulls; 0.01 when
    a trophy was never seen."""
    filled = [None] * n_trophies
    for rd in case.get("history", []):
        for i, r in enumerate(rd.get("rates") or []):
            if i < n_trophies and r is not None:
                filled[i] = r
    return [r if r is not None else 0.01 for r in filled]


def normal_quantiles(med_logit, sigma):
    sigma = max(sigma, SIGMA_FLOOR)
    out = []
    for z in Z:
        out.append(min(99.9999, max(0.0001, sigmoid(med_logit + z * sigma))))
    return out


def mixture_quantiles(med_logit, sigma, p_refresh, drop):
    """Quantiles of a two-branch mixture on the logit scale: with probability
    (1 - p) the rate follows N(med, sigma); with probability p a refresh
    lands and it follows N(med - drop, sigma). By bisection on the CDF."""
    sigma = max(sigma, SIGMA_FLOOR)
    if p_refresh <= 0.0 or drop <= 0.0:
        return normal_quantiles(med_logit, sigma)

    def ncdf(v):
        return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))

    def cdf(x):
        return ((1.0 - p_refresh) * ncdf((x - med_logit) / sigma)
                + p_refresh * ncdf((x - (med_logit - drop)) / sigma))

    out = []
    lo, hi = med_logit - 12.0, med_logit + 12.0
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(min(99.9999, max(0.0001, sigmoid(0.5 * (a + b)))))
    return out


def member_trend(states, t_star, base_sig, n_trophies):
    """Logit(rate) linear in ln(1+hours), last 24 h of distinct reads."""
    if len(states) < 3:
        return None
    t_last = states[-1][0]
    use = [s for s in states if s[0] >= t_last - 24.0]
    if len(use) < 3:
        use = states
    xs = [math.log1p(max(0.0, t)) for t, _, _ in use]
    x_star = math.log1p(max(0.0, t_star))
    out = []
    for i in range(n_trophies):
        ys = [logit(s[2][i]) if i < len(s[2]) and s[2][i] is not None else logit(0.01)
              for s in use]
        fit = ols(xs, ys)
        if fit is None:
            return None
        a, b, sd = fit
        sig = math.sqrt(base_sig * base_sig + sd * sd)
        out.append((a + b * x_star, sig))
    return out


def member_satlog(states, t_star, base_sig, n_trophies):
    """Saturating logistic r = L / (1 + exp(-(a + b*x))), L on a coarse grid.
    For fixed L, logit(r / L) is linear in x; the L with the least squared
    error wins."""
    if len(states) < 3:
        return None
    t_last = states[-1][0]
    use = [s for s in states if s[0] >= t_last - 24.0]
    if len(use) < 3:
        use = states
    xs = [math.log1p(max(0.0, t)) for t, _, _ in use]
    x_star = math.log1p(max(0.0, t_star))
    out = []
    for i in range(n_trophies):
        rs = [s[2][i] if i < len(s[2]) and s[2][i] is not None else 0.01 for s in use]
        best = None
        for big_l in SATLOG_GRID:
            ys = []
            ok = True
            for r in rs:
                if r <= 0.01 or r >= big_l:
                    ok = False
                    break
                ys.append(math.log((r / big_l) / (1.0 - r / big_l)))
            if not ok:
                continue
            fit = ols(xs, ys)
            if fit is None:
                continue
            a, b, sd = fit
            sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
            if best is None or sse < best[0]:
                best = (sse, big_l, a, b, sd)
        if best is None:
            return None
        _, big_l, a, b, sd = best
        lg = a + b * x_star
        r_star = big_l / (1.0 + math.exp(-lg)) if lg < 700 else big_l
        sig = math.sqrt(base_sig * base_sig + sd * sd)
        out.append((logit(r_star), sig))
    return out


def member_en(states, t_star, base_sig, n_trophies):
    """Earners-over-players: log-linear E(t) and log-linear N(t) in
    ln(1+hours); the rate at the target is E / N."""
    if len(states) < 3:
        return None
    xs = [math.log1p(max(0.0, t)) for t, _, _ in states]
    logn = [math.log(max(1, n)) for _, n, _ in states]
    fit_n = ols(xs, logn)
    if fit_n is None:
        return None
    a_n, b_n, sd_n = fit_n
    x_star = math.log1p(max(0.0, t_star))
    logn_star = a_n + b_n * x_star
    out = []
    for i in range(n_trophies):
        es = []
        for _, n, rates in states:
            r = rates[i] if i < len(rates) and rates[i] is not None else 0.01
            es.append(max(0.5, r * max(1, n) / 100.0))
        fit_e = ols(xs, [math.log(e) for e in es])
        if fit_e is None:
            return None
        a_e, b_e, sd_e = fit_e
        r_star = 100.0 * math.exp((a_e + b_e * x_star) - logn_star)
        sig = math.sqrt(base_sig * base_sig + sd_e * sd_e + sd_n * sd_n)
        out.append((logit(r_star), sig))
    return out


def refresh_params(states, horizon_hours):
    """The refresh mixture's (p, drop) from the history, falling back to the
    launch-night priors (kit RULES.txt) when the history shows no refresh:
    jump = median log N-jump between distinct reads (else 1.802), tau =
    median hours between distinct reads (else 4.7)."""
    jumps = []
    gaps = []
    for (t0, n0, _), (t1, n1, _) in zip(states, states[1:]):
        if n1 > n0 > 0:
            jumps.append(math.log(n1 / n0))
        if t1 > t0:
            gaps.append(t1 - t0)
    jump = median(jumps) if jumps else JUMP_PRIOR
    tau = median(gaps) if gaps else TAU_PRIOR
    tau = max(tau, 0.5)
    p = 1.0 - math.exp(-horizon_hours / tau)
    drop = jump * math.sqrt(max(1.0, horizon_hours / tau))
    return p, drop


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    horizon_min = float(case.get("horizon_minutes", 0))
    d_days = horizon_min / 1440.0
    base_sig = SIGMA_RW * math.sqrt(max(d_days, 0.0))
    target = case.get("target") or case.get("at")
    t_star = hours_since_release(target) if target else 0.0
    horizon_hours = horizon_min / 60.0

    if not case.get("history"):
        return {"trophies": [{"id": tr["id"], "q": [0.5] * 5} for tr in trophies]}

    states = distinct_states(case)
    rates_last = last_rates(case, n)

    if len(states) <= 2:
        # Flat regime: persistence median + refresh mixture. The drop assumes
        # earners stay fixed while players jump -- exactly wrong for a
        # saturated trophy, whose earners grow with the player base: launch
        # night "Just Like Paradise" stayed at 100% (actual drop 0.00 logits)
        # while the rest fell ~2.6. So the drop is tapered to 0 as the last
        # rate approaches 100% (factor 1 - r^50: ~1 below 95%, ~0 at 100%);
        # the taper's middle is a principled guess, its endpoints are the
        # observed drops.
        p, drop = refresh_params(states, horizon_hours)
        rows = []
        for i, tr in enumerate(trophies):
            r = rates_last[i] / 100.0
            qs = mixture_quantiles(logit(rates_last[i]), base_sig, p,
                                   drop * (1.0 - r ** 50))
            rows.append({"id": tr["id"], "q": [round4(q) for q in qs]})
        return {"trophies": rows}

    # Rich regime: equal-weight logit-mean of the trend members that fit.
    members = []
    for fn in (member_trend, member_satlog, member_en):
        got = fn(states, t_star, base_sig, n)
        if got is not None:
            members.append(got)
    if not members:
        rows = []
        for i, tr in enumerate(trophies):
            qs = normal_quantiles(logit(rates_last[i]), base_sig)
            rows.append({"id": tr["id"], "q": [round4(q) for q in qs]})
        return {"trophies": rows}

    rows = []
    for i, tr in enumerate(trophies):
        meds = [m[i][0] for m in members]
        sigs = [m[i][1] for m in members]
        med = sum(meds) / len(meds)
        mean_sig = math.sqrt(sum(s * s for s in sigs) / len(sigs))
        spread = math.sqrt(sum((m - med) ** 2 for m in meds) / len(meds))
        sig = math.sqrt(mean_sig * mean_sig + (spread / 2.0) ** 2)
        qs = normal_quantiles(med, sig)
        rows.append({"id": tr["id"], "q": [round4(q) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
