# FinOptix — Cross-Sectional Equity Signals on Indian Markets

Pick 30 Indian stocks a month, hold them, and find out whether the picking was worth
anything. That is the whole project. The interesting part turned out to be the second half
of that sentence.

## Where this started

FinOptix was a summer project with the Finance & Analytics Club at IIT Kanpur, May to
August 2025. That phase built the pipeline: an XGBoost model over thirteen technical
indicators to score stocks, a fundamental screen on P/E, debt-to-equity and market cap to
cut the list down, Black-Litterman to blend those views with market-implied returns, and
Markowitz mean-variance optimisation to set the weights. It is written up in the
[end-evaluation report](./FAC%20-%20FinOptix%20Summer%20Project'25%20End%20Evaluation%20Report.pdf)
and `FINOPTIX_END_IMPLEMENTATION.ipynb`.

That work gave me the shape of the problem — predict, select, weight — and working code
for the classical portfolio-theory pieces. What it reported was the optimiser's own view
of the portfolio it had solved for: expected return and volatility at the solution.

This repository is what happened when I came back to it and asked what the thing does out
of sample.

## The change that mattered

The original model scored each stock on its own. Forty-nine separate XGBoost models, each
trained on one company's history, each predicting that company's return. Then you line the
predictions up and take the best ten.

That last step is where it breaks. Two models trained on different stocks have no shared
scale — one might systematically predict 2% and another 0.5% for reasons that have nothing
to do with which stock is actually better. Ranking their outputs against each other is
comparing numbers that were never on the same axis. The features had the same problem:
`ma_50` for Nestlé is around ₹2,400 and for ONGC around ₹200, so a tree that learned to
split at ₹1,000 on one stock is learning nothing transferable about the other. Fifteen of
the fifty names spent most of the test period at prices above anything in their training
range, where a tree simply clamps to its last split and stops responding.

So I rebuilt the signal around the comparison the strategy actually makes. The model sees
every liquid stock on a given day at once — roughly 350 of them — as a set of tokens, and
runs self-attention across that set. Not across time within one stock; across stocks within
one day. There is no positional encoding, because the order of the stocks means nothing,
and a transformer without positions is permutation-equivariant, which is exactly the right
symmetry for a cross-section.

Every feature is a percentile rank computed fresh each day, so a value of 0.9 means the same
thing for Nestlé and ONGC, in 2015 and in 2026. The target is the 21-day forward return
minus that day's cross-sectional average, so the model is asked which stocks beat the field
rather than what the market will do — market direction is common to every name and useless
for choosing between them. Sector embeddings and sector-relative ranks let it separate a
stock that is rising from a stock whose whole industry is rising.

The last piece is the loss. Correlation across the full cross-section spends most of its
gradient on the two hundred-odd stocks in the middle that will never be bought. I weight
40% of it on the top quintile instead. That one change is worth more than the architecture:
the model's information coefficient gets worse, and its portfolio gets substantially better.

On the same universe and the same months, the transformer's thirty picks returned 1.75% a
month against momentum's 0.92%, and only nine of the thirty names overlapped. Blending the
two by rank beat either alone.

## Where it landed

Every model below picks 30 stocks a month from the same NIFTY 500 panel, holds them
equally weighted, and pays 20bps on turnover. January 2021 to January 2026, 58 rebalances.
Sharpe is annualised arithmetic excess return over a 6% risk-free rate, divided by
annualised volatility.

| Model | CAGR | Sharpe | Vol | Max DD | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| **Ridge + transformer** | **55.0%** | **1.70** | 24.0% | −28.0% | 86.2 | 29.0 | 119.3 | 74.0 | 0.9 |
| Ridge + transformer + momentum | 52.4% | 1.63 | 24.0% | −27.3% | 97.4 | 24.0 | 109.6 | 48.6 | 9.6 |
| Transformer + momentum | 51.5% | 1.58 | 24.5% | −27.0% | 103.7 | 20.8 | 106.7 | 42.0 | 12.4 |
| Transformer alone | 50.0% | 1.55 | 24.2% | −26.2% | 82.9 | 15.4 | 131.7 | 65.9 | −0.1 |
| Ridge alone, 15 coefficients | 47.4% | 1.55 | 22.8% | −29.1% | 73.4 | 35.5 | 98.8 | 63.2 | −3.6 |
| 12-1 momentum, no model | 45.8% | 1.45 | 23.9% | −27.5% | 79.5 | 20.1 | 99.6 | 39.7 | 12.9 |
| *Equal-weight NIFTY 500 — benchmark* | *29.3%* | *1.24* | *17.1%* | *−21.5%* | 53.4 | 10.5 | 57.7 | 33.0 | 5.9 |

Averaging the transformer's and ridge's percentile ranks beats either alone and beats
adding momentum on top. There is a reason for that ordering rather than luck: momentum is
already one of the fifteen features both models consume, so adding it as a third
equal-weighted signal simply over-weights a factor they both use. Ridge and the transformer
are genuinely different — a linear map over factor ranks against attention across the
cross-section — and they agree on only 56% of their picks.

The gap between the best and worst model in that table is 9 points of CAGR. The gap
between any of them and the benchmark is 16 to 26. Model choice matters far less than the
factor construction, the decision to hold thirty names, and equal weighting.

The one model that fails outright is the one denied the factors. An LSTM given sixty days
of raw price history and nothing else returns 18.7%, below the benchmark; the same
architecture given the cross-sectional factor ranks returns 37.7%. Whatever is predictable
here lives in the relative features, not the shape of the price path.

## What the numbers do not support

I ran an adversarial audit against this repository before publishing it, and it changed
several conclusions. What follows is what survived and what did not.

**Ridge + transformer is not significantly better than the simpler blend.** The Sharpe
advantage is about +0.12 with a bootstrap standard error near 0.15. It is the best of seven
signal combinations, tested after roughly thirteen earlier configuration choices. Expected
best-of-seven improvement from pure noise at this sample size is around +0.20 Sharpe, which
is larger than what was measured. It also flips sign if you hold forty names instead of
thirty, and a single month — June 2024 — accounts for about a third of the gap. Treat the
ordering in that table as indicative, not established.

**Two years carry the compound return.** 2021 and 2023 returned 86% and 119%. Excluding
both leaves roughly 22% CAGR against a benchmark that did about 21%. There is no 55%-a-year
strategy here; there is a strategy that was long Indian mid-caps through an exceptional
run.

**The edge has gone quiet.** Monthly excess over the benchmark carried t ≈ +4.0 across
2021–2023 and t ≈ +0.22 across 2025–2026, where 2025 excess was −0.4 points. Whether that
is factor cyclicality or decay, thirteen months cannot say.

**Survivorship is the largest unquantified bias.** The universe is the *current* NIFTY 500
membership backfilled to 2015, then filtered on full-sample data availability. Forty-four
of 385 surviving names have no price on the first test date, and no name delists across
five years of Indian mid-caps, which does not happen in reality. Momentum-style selection
is particularly exposed: a stock joins the index *because* it ran up. No point-in-time
membership file exists in this repository, so the bias cannot be measured with the data
present.

**Capacity is small.** Median daily traded value of held names is about ₹17.6 crore, so at
10% participation the strategy supports roughly ₹53 crore (~$6M).

**An earlier version of this README reported Sharpe using `(CAGR − rf) / vol`.** That is not
a Sharpe ratio, and it inflated every figure by roughly 0.2. The table above uses arithmetic
excess return. An earlier backtest engine also dropped the portfolio's return on each
rebalance day — 58 of 1,257 days, all at the turn of the month — which understated both the
strategies and the benchmark. Both are fixed here.

---

## 1. Architecture

```
STAGE 1  PREDICT   cross-sectional transformer + 12-1 momentum, blended by rank
STAGE 2  SELECT    top 30 of ~250 liquid names, monthly
STAGE 3  WEIGHT    equal weight (MVO and Black-Litterman tested, both rejected)
         AUDIT     Fama-French / Carhart attribution, placebo test
```

Each month: score every liquid stock with both signals, average their cross-sectional
percentile ranks, buy the top 30 equally weighted, hold, repeat. **Measured one-way
turnover is 0.509** — about 6.6 of 30 names change per rebalance (22%).

## 2. Data and protocol

| | |
|---|---|
| Training universe | 1,117 NSE equities, ≥1,500 trading days, ≥₹3cr median daily value |
| Deployment universe | NIFTY 500 constituents, ≥₹5cr median daily value (~250–330/day) |
| Price history | Jan 2010 – Aug 2026, daily, adjusted (Yahoo Finance) |
| Features | 15 cross-sectional factor ranks + 3 sector-relative + sector embedding |
| Target | Forward 21-day return, cross-sectionally demeaned |
| Sampling | Weekly for training, monthly for evaluation |
| Retraining | Annual, expanding window, 6 folds, 3-seed ensemble |

### Walk-forward folds

| Fold | Trained | Validated | **Tested** |
|---|---|---|---|
| 1 | Mar 2011 – Mar 2020 | Apr–Oct 2020 | Jan–Dec 2021 |
| 2 | Mar 2011 – Mar 2021 | Apr–Oct 2021 | Jan–Dec 2022 |
| 3 | Mar 2011 – Mar 2022 | Apr–Oct 2022 | Jan–Dec 2023 |
| 4 | Mar 2011 – Mar 2023 | Apr–Oct 2023 | Jan–Dec 2024 |
| 5 | Mar 2011 – Mar 2024 | Apr–Oct 2024 | Jan–Dec 2025 |
| 6 | Mar 2011 – Mar 2025 | Apr–Oct 2025 | Jan 2026 |

No test-period data reaches the model. Validation is used only for early stopping.
**Embargo caveat:** the gap is 88 *calendar* days while the longest auxiliary label is 63
*trading* days (~91 calendar). In 4 of 6 folds the final training snapshot's 63-day
label extends 1–8 days past the fold boundary. The 21-day main target — the one used for
all reported results — has ~58 days of slack and never overlaps.

### Statistics

Forward 21-day returns overlap almost completely on consecutive days, so naive daily
t-statistics overstate significance by roughly √21. All t-statistics here are
Newey-West (21 lags) or computed on non-overlapping monthly observations. An earlier
draft reported t = 2.02 on daily ICs; the Newey-West value for that series is 0.68.

## 3. Results

### 3.1 Which factors carry signal

Fifteen candidate factors were screened on the NIFTY 500 panel over 2016–2026, each tested
for whether it ranks next month's cross-sectional returns better than chance, with
Newey-West and non-overlapping t-statistics to account for the fact that overlapping
21-day labels make naive daily tests look about √21 times more significant than they are.

Four survived that screen: **12-1 momentum** (the strongest by some margin, and the only
one clearing significance on all three non-overlapping offsets), **price relative to its
200-day average**, **size** with small beating large, and **6-1 momentum**. Turnover and
Amihud illiquidity were marginal.

Short-term reversal, realised volatility, downside volatility, beta, idiosyncratic
volatility, skewness and the MAX effect showed nothing. Several of these are documented
anomalies in US equities, so their absence here is worth noting rather than glossing over
— it may be a genuine market difference, or an artefact of a large-cap Indian universe
over a single bull-market decade.

12-1 momentum's sign is *a priori* from Jegadeesh & Titman (1993), not fitted on this data.
These are full-sample descriptive statistics, not walk-forward, and are used to motivate
the feature set rather than to support any performance claim.

### 3.2 Where the signal lives

Ablation on the LSTM, same folds and evaluation dates: given only 60 days of raw price
history it returns 18.7% (below the benchmark); given only the cross-sectional
factor ranks it returns 37.7%. The sequence branch contributes nothing the factors do not
already carry.

### 3.3 Target and loss variants

Seven variants of the transformer, same architecture and folds, differing only in what
they predict and how they are scored:

| Target + loss | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Vol-scaled + top-weighted | 41.5% | 1.74 | −28.4% |
| Top-decile classification | 40.5% | 1.39 | −32.5% |
| Demean + top-weighted (headline) | 39.5% | 1.37 | −28.6% |
| Clipped + top-weighted | 39.4% | 1.39 | −29.2% |
| Rank + top-weighted | 39.3% | 1.39 | −31.0% |
| Soft top-k portfolio | 34.9% | 1.27 | −32.2% |
| Rank + plain correlation | 34.2% | 1.34 | −31.6% |

Weighting the loss toward the top of the distribution is worth roughly five points of CAGR
over plain cross-sectional correlation — the bottom row is the only variant that optimises
the full ranking, and it is the worst. Differences among the top five are within selection
noise across seven variants on 58 months, so no winner is claimed; the headline
configuration is not the best row in this table.

### 3.4 Portfolio construction

Momentum picks, 2016-2026 (`results/v6.json`):

| Weighting | CAGR | Max DD |
|---|---|---|
| MVO max-Sharpe | 31.5% | −38.6% |
| Equal weight | 29.2% | −38.4% |
| Black-Litterman then MVO | 24.0% | −36.0% |

Blend picks, 2021-2026 (`src/final.py`):

| Weighting | CAGR | Max DD | Effective holdings |
|---|---|---|---|
| Equal weight | best of the four | −27.4% | 30.0 |
| MVO + L2 (γ=5) | −6pp | −32.9% | 12.7 |
| MVO unconstrained | −7pp | −34.5% | 9.5 |
| MVO + L2 (γ=2) | −8pp | −33.8% | 10.9 |

Equal weighting won every comparison on blend picks. Once the selection stage works,
mean-variance has only a trailing 252-day average to estimate returns from — mostly noise —
and it concentrates thirty positions into about eleven chasing it. Black-Litterman was the
weakest scheme tested: it shrinks the view toward market equilibrium, and here the view is
the strategy. On momentum picks over the longer 2016-2026 window MVO did beat equal
weighting, so the benefit disappears as the selection stage improves (cf. DeMiguel,
Garlappi & Uppal 2009).

These weighting comparisons predate the engine fix described above, so treat the CAGR
figures as relative rather than absolute; the ordering is unaffected.

### 3.5 Annual returns and excess over benchmark

See the table in [Where it landed](#where-it-landed) for annual returns by model.

Ridge + transformer against the equal-weight benchmark, by year:

| | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| Strategy | 86.2 | 29.0 | 119.3 | 74.0 | 0.9 |
| Benchmark | 53.4 | 10.5 | 57.7 | 33.0 | 5.9 |
| Excess | +32.8 | +18.5 | +61.6 | +41.0 | **−5.0** |

Positive in four of five years, negative in the most recent complete one.

### 3.6 Factor attribution

Ridge + transformer, corrected engine, daily returns regressed on factors built from the
same universe:

| Regression | Alpha | t | MKT | SMB | WML |
|---|---|---|---|---|---|
| ~ MKT + SMB | +4.95%/yr | +1.20 | 1.32 | 0.42 | — |
| ~ MKT + SMB + WML | −0.11%/yr | −0.03 | 1.17 | 0.42 | **0.51** |

Against market and size alone, alpha is +4.95% a year — but t = 1.20, so it is not
statistically significant and would need roughly three times the sample to establish. Add
the momentum factor and alpha is exactly zero (−0.11%, t = −0.03). Whatever the strategy
earns beyond market and size is momentum exposure.

The ensemble carries less of it than the simpler blend — WML loading 0.51 against 0.65 —
which follows from dropping momentum as an explicit signal, but not enough to generate
unexplained return. This is efficient factor harvesting, not alpha.

### 3.7 Placebo test

Momentum's top-30 vs **120 random 30-name portfolios** from the same universe, same
schedule, **gross of costs**: random mean 17.5%, momentum 31.0% — **z = +6.90**.

Costs are deliberately excluded here: a randomly redrawn 30-name portfolio turns over
~2.0/month against momentum's ~0.5, so charging both the same rate would credit momentum
with ~3.3pp/yr of pure turnover advantage. An earlier draft reported z = +8.8 from the
cost-charged version; that figure overstates the result.

## Known defects

**1. The evaluation window has a four-month hole, and the headline is probably flattered by it.** The `amihud` feature goes NaN across
the entire cross-section when a market-wide stale-volume bar falls inside its 63-day
window, and `xattn.py` drops any snapshot with a non-finite feature. This silently
removed **9 of 67 evaluation months**. The portfolio held **unrebalanced from
2025-03-03 to 2025-07-01 — 120 days** — and evaluation ends Jan 2026 rather than Jul 2026,
discarding six months of available data. Fix: `min_periods` on the rolling window, or drop
non-finite features per-name rather than per-snapshot. **All results above are computed on
the 58 dates that survived**, so they are internally consistent but do not cover the period
a reader would assume. The missing months fall disproportionately in 2025, the weakest
stretch in the sample — 2025 excess return over the benchmark was −5 points — so the
headline CAGR is more likely overstated than understated. The fix is two lines
(`min_periods` on the rolling windows, and replacing zero traded value with NaN before
dividing); it was verified to eliminate every in-window NaN date but the corrected re-run
is not reflected in the figures above.

**2. Survivorship bias interacts with the selection rule.** Universes are built from
*current* index membership and a full-sample data-availability filter, applied
retroactively. This is not neutral for momentum: a stock joins the NIFTY 500 *because* it
ran up, and momentum selects stocks that ran up. The placebo does **not** neutralise
this — random draws inherit the bias uniformly while momentum exploits it. An earlier
draft claimed otherwise.

**3. No bear market in the test window.** Jan 2021 – Jan 2026 was a strong Indian bull
market.

**4. Train/deploy universe mismatch.** The transformer is trained with cross-sectional
attention over ~527 all-NSE names but deployed on ~330 NIFTY 500 names, so each score is
conditioned on a token set that never exists at deployment. Its IC is ~4× higher in the
less-liquid half of the universe (+0.0142 vs +0.0038), so this is unlikely to be benign.

**5. Multiple comparisons.** Roughly 30 configurations were run across universes, targets,
losses and weighting schemes. Limitations are disclosed and the full ladder published, but
the headline is one selection among many. `final.py` also picks its attribution target
post-hoc by realised return.

**6. Costs exclude market impact.** 20bps on turnover is reasonable for retail size in
liquid names and understated at institutional size in mid-caps.

## What was tried and rejected

| Approach | Result |
|---|---|
| XGBoost on raw technical indicators (original project) | IC ≈ 0 — same-bar leakage; features in price units |
| LSTM on raw price sequences | IC +0.0114 (t = 0.78) |
| XGBoost on cross-sectional factors | IC −0.0046 (t = −0.30) |
| MVO on blend picks | Worse than equal weight at every setting |
| Black-Litterman | Worst weighting scheme tested (Sharpe 0.69) |
| Expanding to all-NSE small caps | Momentum Sharpe 1.42 → 0.63 |
| Blending three signals | No better than the best two |

## Repository

| File | Purpose |
|---|---|
| `src/xattn.py` | Cross-sectional transformer (headline model) |
| `src/sweep.py` | Seven target/loss variants |
| `src/final.py` | Three-stage backtest + Fama-French attribution |
| `src/v3.py` | Factor construction, ridge/XGBoost/composite comparison |
| `src/v5.py` | Placebo test |
| `src/backtest.py` | Walk-forward harness with turnover costs |

```bash
pip install pandas numpy scipy scikit-learn xgboost torch yfinance
python src/dl_big.py       # download panel
python src/xattn.py raw_big.pkl
python src/final.py
```

`LTIM.NS` and `TATAMOTORS.NS` no longer resolve on Yahoo (the latter demerged to
`TMPV.NS` in 2025). Pin a static price CSV for exact reproducibility.

## References

Jegadeesh & Titman (1993) · Fama & French (1993) · Carhart (1997) ·
DeMiguel, Garlappi & Uppal (2009) · Gu, Kelly & Xiu (2020) · Moskowitz & Grinblatt (1999)
