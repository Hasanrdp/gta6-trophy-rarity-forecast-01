"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95). Standard library only, no network, deterministic.

What is being forecast, and why it is hard: an earn rate is a ratio, r = E/N,
of the tracked players who earned the trophy (E) over the players Exophase
tracks (N). Exophase refreshes its page on its own slow clock, so the history
the house records is a step function: long flat stretches, then a jump. At a
refresh N usually jumps a lot (launch night: 638 -> 3,868 tracked players,
x6.06) while E barely moves, so every rate falls by roughly ln(N_new / N_old)
in logit terms (about 1.7 logits at the first refresh, measured from the live
page). Between refreshes nothing moves at all. Over the final's 4.5-day horizon
N growth slows (the launch surge passes) while E growth accelerates (players
work through the story), so early trophies' rates recover and late trophies'
keep falling. A forecaster that only extrapolates the rate itself misses the
saturation; one that only models E/N misses how slowly earners arrive.

The forecast is therefore the equal-weight mean, on the logit scale, of four
members with deliberately opposing biases, so their errors partly cancel:

  1. A saturating logistic in log-time, r = L / (1 + exp(-(a + b ln(1+t)))),
     fitted on the last 24 hours of distinct reads, with the asymptote L on a
     coarse grid (5, 10, 20, 35, 50, 70, 90, 99.99). The asymptote is the
     whole point: a rate cannot keep rising on a log-time line forever, and
     the house curve baseline (member 4) has no asymptote and overshoots.
     The coarse grid regularises: a fine grid fits noise and does worse.
  2. The same saturating logistic on the last 12 hours with a fine grid: a
     second opinion on the recent slope and asymptote.
  3. A point-lag earners-over-players model, E(t) = s * N(t - tau), with s
     capped at 1 (a share cannot exceed 1) and N from a power-law /
     saturating average. It is the member that understands dilution: when N
     jumps, rates fall. On its own it underforecasts (players earn on a wide
     spread of schedules, not one lag), which is exactly why it belongs in a
     mean with member 4.
  4. The house curve baseline's own shape, logit(r) linear in ln(1+t) over
     the last 24 hours, with no slope cap. On its own it overshoots -- in the
     agent worlds its slope runs to 3 or 4 on story trophies -- which is
     exactly why it belongs in a mean with member 3: the overshoot and the
     underforecast cancel. Capping the slope was tried and made the ensemble
     much worse (the median's mean |logit error| went from 0.56 to 1.37),
     so the slope is left uncapped; the member's logit is only clamped to
     [-6, 6] as a safety net against a pathological single-slope fit.

The four were chosen by backtesting on synthetic launch worlds (agent-based:
players join on a growing curve, each with a log-normal pace, each trophy
with a time-to-earn and an eventual share; the page frozen between Exophase
refreshes). On those worlds the four-way mean's median misses the answer by
0.56 logits on average, against 1.55 for the house curve baseline and 1.80
for persistence. The members were not tuned to the practice cases: the
practice histories are flat (one distinct read), so every member falls back
to the last rate and only the spread speaks.

The spread is a split normal on the logit scale. Its width combines the
saturating-logistic fit's residuals, the disagreement between the four
members (as a standard error, since a mean of four is more precise than any
member), the uncertainty of the N extrapolation, a horizon term, a staleness
term (the answer is the page as of its last refresh before the target, which
may be hours old), and -- only on the downside -- a refresh-mixture term:
while the page is stale a refresh may or may not land before the target, and
a refresh can only push rates down. When the history is flat and the horizon
is short (the practice cases) that term dominates the lower tail, which is
the honest shape: the median stays at the last rate, the 5% quantile reaches
down toward the post-refresh level.

Constants that are not estimated from the case (refresh interval prior 6h,
refresh-drop prior 1.5 logits, horizon spread 0.30 per root-day) come from
the launch-night evidence in the kit: the page sat unchanged from 23:48Z to
04:00Z and refreshed at 04:30Z, and the live page still showed that
refresh's counts at 11:43Z.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_DOWN = [-1.645, -0.674]          # z for the 5% and 25% quantiles
Z_UP = [0.674, 1.645]              # z for the 75% and 95% quantiles
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

REFRESH_INTERVAL_PRIOR_H = 6.0    # Exophase refresh gap, launch-night evidence
REFRESH_DROP_PRIOR = 1.5          # typical |logit| fall at one refresh
HORIZON_SIGMA = 0.25              # logit sd per sqrt(day) of horizon
TAU_GRID = [0.0, 6.0, 12.0, 24.0, 48.0, 72.0, 120.0, 168.0, 240.0]
L_COARSE = [5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0, 99.99]
L_FINE = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
          60.0, 70.0, 80.0, 90.0, 95.0, 99.99]


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def hours_since_launch(iso):
    return (parse_time(iso) - T0).total_seconds() / 3600.0


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def sd(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def linfit(xs, ys):
    """Least squares y = a + b x. Returns (a, b, resid_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = mean(xs)
    my = mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if n < 2 or den <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rsd = math.sqrt(sum(e * e for e in resid) / max(1, n - 2)) if n > 2 else 0.0
    return a, b, rsd


def fit_origin(xs, ys):
    """Least squares y = s x through the origin; returns (s, sse)."""
    den = sum(x * x for x in xs)
    if den <= 0:
        return 0.0, sum(y * y for y in ys)
    s = sum(x * y for x, y in zip(xs, ys)) / den
    sse = sum((y - s * x) ** 2 for x, y in zip(xs, ys))
    return s, sse


def distinct_snapshots(case):
    """Collapse the history to the reads where the page actually changed."""
    snaps = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append((hours_since_launch(read["at"]), read["players_tracked"],
                      list(read["rates"])))
    return snaps


class NModel:
    """Growth of the tracked-player count.

    Two shapes are fitted to the distinct reads and averaged in log space: a
    power law, ln N = a + b ln(1+t), which never saturates, and a saturating
    Weibull, N = Ninf (1 - exp(-(t/tau)^beta)), which levels off as the launch
    surge passes. Over a 4.5-day horizon they can differ by nearly 2x, and the
    truth is usually between them. The gap between the two forecasts is
    carried as extra uncertainty in log_sd_at.
    """

    SAT_TAUS = [24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0]
    SAT_BETAS = [0.7, 1.0, 1.3]

    def __init__(self, snaps):
        self.n_last = snaps[-1][1] if snaps else 638.0
        pts = [(t, n) for t, n, _ in snaps if n and n > 0]
        self.n_pts = len(pts)
        if len(pts) < 2:
            self.a = math.log(max(1.0, self.n_last))
            self.b = 0.0
            self.resid_sd = 0.0
            self.x_mean = 0.0
            self.x_ss = 0.0
            self.sat = None
            self.flat = True
            return
        self.flat = sd([math.log(n) for _, n in pts]) == 0.0
        xs = [math.log1p(t) for t, _ in pts]
        ys = [math.log(n) for _, n in pts]
        a, b, rsd = linfit(xs, ys)
        self.a = a
        self.b = min(2.5, max(-0.2, b))
        self.resid_sd = rsd
        self.x_mean = mean(xs)
        self.x_ss = sum((x - self.x_mean) ** 2 for x in xs)
        best = None
        for tau in self.SAT_TAUS:
            for beta in self.SAT_BETAS:
                f = [1.0 - math.exp(-((max(t, 0.0) / tau) ** beta)) for t, _ in pts]
                den = sum(v * v for v in f)
                if den <= 0:
                    continue
                ninf = sum(v * n for v, (_, n) in zip(f, pts)) / den
                if ninf <= 0:
                    continue
                sse = sum((math.log(n) - math.log(max(1.0, ninf * v))) ** 2
                          for v, (_, n) in zip(f, pts) if v > 1e-9)
                cnt = sum(1 for v in f if v > 1e-9)
                if cnt >= 2 and (best is None or sse < best[0]):
                    best = (sse, ninf, tau, beta)
        self.sat = best

    def _power(self, t):
        return math.exp(self.a + self.b * math.log1p(max(0.0, t)))

    def _sat(self, t):
        if self.sat is None:
            return None
        _, ninf, tau, beta = self.sat
        return max(1.0, ninf * (1.0 - math.exp(-((max(t, 0.0) / tau) ** beta))))

    def predict(self, t):
        pl = self._power(t)
        st = self._sat(t)
        if st is None:
            val = pl
        else:
            val = math.exp(0.5 * (math.log(pl) + math.log(st)))
        return max(val, self.n_last) if t >= 0 else val

    def log_sd_at(self, t, horizon_days):
        if self.n_pts < 2:
            return 0.35 * math.sqrt(horizon_days)
        x = math.log1p(max(0.0, t))
        lever = 1.0 + ((x - self.x_mean) ** 2 / self.x_ss if self.x_ss > 0 else 1.0)
        base = self.resid_sd * math.sqrt(lever) + 0.05 * math.sqrt(horizon_days)
        st = self._sat(t)
        if st is not None:
            gap = abs(math.log(self._power(t)) - math.log(st)) / 2.0
            return math.sqrt(base * base + gap * gap)
        return base


def satlog_fit(pts, t_tgt, lgrid):
    """Saturating logistic in log-time, fitted by grid over the asymptote L.

    pts is a list of (t_hours, rate). Returns (forecast_rate, resid_sd) or
    (None, 0.0) when the points cannot identify a fit.
    """
    best = None
    for L in lgrid:
        xs = []
        ys = []
        for t, r in pts:
            if r >= L:
                continue
            xs.append(math.log1p(t))
            ys.append(math.log((r / L) / (1.0 - r / L)))
        if len(xs) < 2:
            continue
        a, b, rsd = linfit(xs, ys)
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, a, b, L, rsd)
    if best is None:
        return None, 0.0
    _, a, b, L, rsd = best
    lg = a + b * math.log1p(max(0.0, t_tgt))
    rate = L / (1.0 + math.exp(-lg)) if lg < 700 else L
    return clamp_rate(rate), rsd


def forecast_case(case):
    trophies = case["trophies"]
    snaps = distinct_snapshots(case)
    if not snaps:
        return {"trophies": [{"id": tr["id"], "q": [0.01, 0.5, 1.0, 2.0, 5.0]}
                             for tr in trophies]}
    t_last = snaps[-1][0]
    t_at = hours_since_launch(case["at"])
    t_tgt = hours_since_launch(case["target"])
    horizon_h = max(0.0, t_tgt - t_at)
    horizon_days = case["horizon_minutes"] / 1440.0
    n_tr = len(trophies)

    nmodel = NModel(snaps)
    n_tgt = nmodel.predict(t_tgt)

    # --- refresh structure, for the downside spread term ---
    gaps = [snaps[i + 1][0] - snaps[i][0] for i in range(len(snaps) - 1)
            if snaps[i + 1][0] > snaps[i][0]]
    interval = median(gaps) if gaps else REFRESH_INTERVAL_PRIOR_H
    interval = min(max(interval, 2.0), 48.0)
    age = max(0.0, t_at - t_last)
    mu_i = math.log(interval)
    sig_i = 0.4

    def phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if horizon_h <= 0:
        p_refresh = 0.0
    else:
        f_age = phi((math.log(max(age, 0.05)) - mu_i) / sig_i)
        f_age_h = phi((math.log(max(age + horizon_h, 0.05)) - mu_i) / sig_i)
        p_refresh = (f_age_h - f_age) / max(1.0 - f_age, 1e-9)
        p_refresh = min(max(p_refresh, 0.0), 0.999)
    jumps = []
    for i in range(len(snaps) - 1):
        for j in range(n_tr):
            r0, r1 = snaps[i][2][j], snaps[i + 1][2][j]
            if r0 is not None and r1 is not None and 0 < r0 < 100 and 0 < r1 < 100:
                jumps.append(abs(logit(r1) - logit(r0)))
    drop_prior = median(jumps) if jumps else REFRESH_DROP_PRIOR
    drop_prior = min(max(drop_prior, 0.3), 2.5)
    sigma_mix = math.sqrt(p_refresh * (1.0 - p_refresh)) * drop_prior

    rows = []
    for j, tr in enumerate(trophies):
        r_last = snaps[-1][2][j]
        if r_last is None:
            r_last = 1.0
        # The 100% trophy: everyone tracked has it; it stays (nearly) 100%.
        if r_last >= 99.99:
            rows.append({"id": tr["id"], "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue
        pts_all = [(t, r) for (t, n, rates) in snaps
                   for r in [rates[j]] if r is not None and 0 < r < 100]
        pts24 = [p for p in pts_all if p[0] >= t_last - 24.0]
        pts12 = [p for p in pts_all if p[0] >= t_last - 12.0]

        # Member 1: saturating logistic, last 24h, coarse asymptote grid.
        rate1, rsd1 = satlog_fit(pts24 if len(pts24) >= 2 else pts_all,
                                 t_tgt, L_COARSE)
        # Member 2: saturating logistic, last 12h, fine asymptote grid.
        rate2, rsd2 = satlog_fit(pts12 if len(pts12) >= 2 else pts_all,
                                 t_tgt, L_FINE)
        # Member 3: point-lag earners over players, share capped at 1.
        e_pts = [(t, r * n / 100.0) for (t, n, rates) in snaps
                 for r in [rates[j]] if r is not None and n]
        best_lag = None
        for tau in TAU_GRID:
            xs = [nmodel.predict(t - tau) for t, _ in e_pts]
            ys = [e for _, e in e_pts]
            s, sse = fit_origin(xs, ys)
            s = min(s, 1.0)
            sse = sum((y - s * x) ** 2 for x, y in zip(xs, ys))
            if best_lag is None or sse < best_lag[0]:
                best_lag = (sse, s, tau)
        if best_lag is None:
            rate3 = r_last
        else:
            _, s_lag, tau_lag = best_lag
            rate3 = clamp_rate(100.0 * s_lag * nmodel.predict(t_tgt - tau_lag) / n_tgt)
        # Member 4: the house curve shape, logit-linear in log-time over the
        # last 24h, uncapped. Fitted on every recorded read (not just the
        # distinct snapshots), exactly as the kit's house curve baseline does
        # it; its overshoot is what cancels member 3's underforecast.
        pairs_c = []
        for read in case["history"]:
            t_read = hours_since_launch(read["at"])
            r_read = read["rates"][j]
            if t_read >= t_at - 24.0 and r_read is not None and 0 < r_read < 100:
                pairs_c.append((math.log1p(t_read), logit(r_read)))
        if len(pairs_c) >= 3:
            a_c, b_c, rsd_c = linfit([p[0] for p in pairs_c],
                                    [p[1] for p in pairs_c])
        elif pairs_c:
            a_c, b_c, rsd_c = pairs_c[-1][1], 0.0, 0.0
        else:
            a_c, b_c, rsd_c = logit(r_last), 0.0, 0.0
        lg4 = min(6.0, max(-6.0, a_c + b_c * math.log1p(max(0.0, t_tgt))))
        rate4 = clamp_rate(sigmoid(lg4))

        members = []
        for rate in (rate1, rate2, rate3, rate4):
            members.append(logit(rate if rate is not None else r_last))
        mu = mean(members)

        # --- spread (logit units), split normal ---
        # The four members disagree by ~1.6 logits sd on final-like worlds,
        # but the mean's error sd is ~1.0 and the pinball-optimal fixed spread
        # is ~0.6: the members' errors cancel, so the disagreement enters
        # divided by 4, not by 2. The horizon term is the largest single
        # component for the final (0.25 * sqrt(4.52) = 0.53).
        sigma_fit = 0.35
        if rsd1 > 0:
            sigma_fit = min(max(rsd1, 0.15), 1.0)
        elif rsd2 > 0:
            sigma_fit = min(max(rsd2, 0.15), 1.0)
        sigma_dis = sd(members) / 4.0
        sigma_n = 0.5 * nmodel.log_sd_at(t_tgt, horizon_days)
        sigma_hor = HORIZON_SIGMA * math.sqrt(max(horizon_days, 0.0))
        slope = abs(b_c) / (1.0 + max(0.0, t_tgt))
        sigma_stale = slope * interval / 2.0
        base2 = (sigma_fit ** 2 + sigma_dis ** 2 + sigma_n ** 2
                 + sigma_hor ** 2 + sigma_stale ** 2)
        sigma_up = math.sqrt(base2)
        sigma_down = math.sqrt(base2 + sigma_mix ** 2)
        if all(e <= 0 for _, e in e_pts):
            sigma_up = max(sigma_up, 1.0)
            sigma_down = max(sigma_down, 1.0)
        sigma_up = min(max(sigma_up, 0.30), 2.0)
        sigma_down = min(max(sigma_down, 0.30), 2.5)
        q = [
            round4(sigmoid(mu + Z_DOWN[0] * sigma_down)),
            round4(sigmoid(mu + Z_DOWN[1] * sigma_down)),
            round4(sigmoid(mu)),
            round4(sigmoid(mu + Z_UP[0] * sigma_up)),
            round4(sigmoid(mu + Z_UP[1] * sigma_up)),
        ]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        q = [min(99.99, max(0.01, v)) for v in q]
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
