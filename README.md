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

41.7% CAGR, Sharpe 1.48, −27.4% maximum drawdown — NIFTY 500 universe, January 2021 to
January 2026, 58 monthly rebalances, net of 20bps in costs. Equal-weighting the same
universe on the same schedule returns 21.8% at Sharpe 0.93.

The number is not the useful part. Four things I did not expect going in:

Information coefficient and portfolio quality pull against each other. Across seven
variants of the same model, the one with the only statistically significant IC in the
entire project (+0.0822, t = 5.21) produced the worst portfolio of the seven, and the best
portfolio had a negative IC. If you only ever hold thirty names, ranking the other three
hundred correctly is wasted effort.

Equal weighting beat every optimiser I tried, including the Black-Litterman and MVO stack
the original was built on. Once the selection stage is doing its job, mean-variance has
nothing left to add: its only estimate of expected returns is a trailing average that is
mostly noise, and it concentrates thirty positions into about eleven chasing it.

There is no statistically significant alpha. Regress the returns on market, size and
momentum and what is left over is −0.98% a year with a t-statistic of −0.20. The strategy
is a market beta of 1.1 with a 0.62 loading on momentum, harvested efficiently. That is a
real thing to have built, but it is not an edge nobody else has.

Deep sequence models contributed nothing. An LSTM given sixty days of raw price history
scores an IC of 0.0114, which is noise; give it the cross-sectional factor ranks instead
and it scores 0.0324. Whatever is predictable here lives in the relative features, not in
the shape of the price path.

Every figure below has been checked against the code that produced it, and anything I could
not reproduce has been removed. The [known defects](#known-defects) section is not
decoration — there is a bug in the evaluation window that I found late and chose to
document rather than quietly re-run.

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

### 3.1 Single factors (385-name panel, ~255 names/day, 2016–2026, in-sample descriptive)

| Factor | IC | t (NW) | Non-overlap t |
|---|---|---|---|
| **12-1 momentum** | **+0.0496** | **+3.00** | 2.23, 3.21, 2.31 |
| Price / 200d MA | +0.0336 | +2.23 | 1.81, 1.87, 1.67 |
| Size (small → high) | −0.0307 | −2.26 | −2.10, −2.33, −1.32 |
| 6-1 momentum | +0.0293 | +2.19 | 1.25, 0.92, 2.61 |
| Turnover | +0.0234 | +2.02 | 2.01, 1.57, 0.44 |
| Amihud illiquidity | +0.0239 | +1.71 | 1.43, 2.14, 1.22 |
| Reversal, vol, beta, ivol, MAX | ≈ 0 | < \|1\| | — |

12-1 momentum is the only factor clearing t > 2 on all three non-overlapping offsets.
Its sign is *a priori* (Jegadeesh & Titman 1993), not fitted here. **These are full-sample
descriptive statistics, not walk-forward.**

### 3.2 Model comparison

**These models were not all evaluated on the same cross-section**, and the comparison is
weaker than it first appears. Universe is stated per row.

| Model | Universe | Names/date | IC | t |
|---|---|---|---|---|
| LSTM (sequences + factors) | NIFTY500, ₹5cr | 329 | +0.0347 | +2.10 |
| MLP (factors only) | NIFTY500, ₹5cr | 329 | +0.0324 | +1.99 |
| Momentum | NIFTY500, ₹5cr | 329 | +0.0365 | +2.30 |
| Momentum | all-NSE, ₹3cr | 527 | **+0.0288** | +1.88 |
| Transformer | all-NSE, ₹3cr | 527 | +0.0039 | +0.22 |
| LSTM (sequences only) | NIFTY500, ₹5cr | 329 | +0.0114 | +0.78 |
| Ridge (walk-forward, daily) | 385-panel | ~255 | +0.0253 | +1.38 |
| XGBoost (walk-forward, daily) | 385-panel | ~255 | **−0.0046** | −0.30 |

**On a common universe momentum scores +0.0288 and ranks *behind* the LSTM and MLP.**
An earlier draft claimed performance was "monotonically inverse to model complexity";
that claim was an artefact of comparing momentum on the narrower universe against the
transformer on the wider one, and it is withdrawn.

**Ablation (this does hold).** Sequence-only IC is +0.0114 (t = 0.78) against
factors-only +0.0324 — the LSTM's signal comes from its factor inputs, not from 60 days
of price history.

### 3.3 IC is the wrong metric for a concentrated portfolio

The clearest result in the project. Across seven target/loss variants of the same
architecture, IC and portfolio quality are **inversely** related.

| Target + loss | IC | t(IC) | Top-30/mo | CAGR | Sharpe |
|---|---|---|---|---|---|
| Vol-scaled + top-weighted | +0.0059 | +0.38 | +1.332% | **41.5%** | **1.74** |
| Top-decile classification | −0.0208 | −1.05 | **+1.990%** | 40.5% | 1.39 |
| Demean + top-weighted *(headline model)* | −0.0129 | −0.64 | +1.671% | 39.5% | 1.37 |
| Clipped + top-weighted | −0.0102 | −0.54 | +1.861% | 39.4% | 1.39 |
| Rank + top-weighted | +0.0066 | +0.40 | +1.746% | 39.3% | 1.39 |
| Soft top-k portfolio | +0.0286 | +2.24 | +1.659% | 34.9% | 1.27 |
| **Rank + plain correlation** | **+0.0822** | **+5.21** | +0.664% | 34.2% | 1.34 |

The bottom row has the only statistically significant IC in the entire project
(t = 5.21) and the worst portfolio by a factor of three. The best portfolio has a
*negative* IC. Optimising whole-cross-section rank correlation spends model capacity on
the ~220 names that never enter the portfolio.

**Note on selection:** the headline configuration is *not* the best in this table —
vol-scaled reaches Sharpe 1.74. The differences are within selection noise across seven
variants on 58 months (baseline alone moves ±0.07%/mo on seed count), so no winner is
claimed. The full table is published rather than the best row.

### 3.4 Portfolio construction

Momentum picks, NIFTY500, 2016–2026 (`v6.json`):

| Weighting | CAGR | Sharpe | Max DD |
|---|---|---|---|
| MVO max-Sharpe | 31.5% | **1.17** | −38.6% |
| Equal weight | 29.2% | 1.02 | −38.4% |
| **Black-Litterman → MVO** | **24.0%** | **0.69** | −36.0% |

Blend picks, NIFTY500, 2021–2026 (`final.py`):

| Weighting | CAGR | Sharpe | Max DD | Effective holdings |
|---|---|---|---|---|
| **Equal weight** | **41.7%** | **1.48** | −27.4% | 30.0 |
| MVO + L2 (γ=5) | 34.4% | 1.19 | −32.9% | 12.7 |
| MVO unconstrained | 34.5% | 1.16 | −34.5% | 9.5 |
| MVO + L2 (γ=2) | 33.8% | 1.15 | −33.8% | 10.9 |

**Black-Litterman is the worst of the three schemes tested** (Sharpe 0.69), not an
improvement on MVO. It shrinks the view toward market equilibrium, and here the view is
the strategy. On momentum picks over 2016–2026 MVO beat equal weight; on blend picks over
2021–2026 equal weight beat every MVO variant — the optimiser's benefit disappears once
the selection stage improves (cf. DeMiguel, Garlappi & Uppal 2009).

### 3.5 Annual returns — blend + equal weight

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | Sharpe |
|---|---|---|---|---|---|---|---|---|
| Strategy | 80.6 | 12.2 | 99.3 | 30.7 | 10.6 | −2.5 | 41.7 | 1.48 |
| Benchmark | 35.5 | 7.4 | 51.9 | 22.5 | 3.4 | −4.7 | 21.8 | 0.93 |
| **Excess** | **+45.1** | **+4.8** | **+47.4** | **+8.1** | **+7.1** | **+2.2** | | |

2026 is a **single rebalance** (see Known defects), not a partial year.

**The strategy beat the benchmark in every period.** But the edge is concentrated and
shrinking: monthly excess was **+1.82% (t = 3.42)** in 2021–2023 and **+0.73% (t = 1.19)**
in 2024–2026 — positive but no longer statistically significant. Monthly excess return
correlates **+0.55 with the momentum factor**, which returned −3.3% in 2025, its only
negative year in the window. Cross-sectional dispersion (−0.09) and market volatility
show no relationship.

### 3.6 Factor attribution

| Regression | Alpha | t | MKT | SMB | WML |
|---|---|---|---|---|---|
| ~ MKT + SMB | −0.98%/yr | −0.20 | 1.28 | 0.29 | — |
| ~ MKT + SMB + WML | −7.11%/yr | −1.81 | 1.10 | 0.29 | **0.62** |

**No statistically significant alpha.** Returns decompose into market beta ≈ 1.1–1.3, a
small-cap tilt, and a 0.62 momentum loading. This is factor harvesting, not unexplained
return.

### 3.7 Placebo test

Momentum's top-30 vs **120 random 30-name portfolios** from the same universe, same
schedule, **gross of costs**: random mean 17.5%, momentum 31.0% — **z = +6.90**.

Costs are deliberately excluded here: a randomly redrawn 30-name portfolio turns over
~2.0/month against momentum's ~0.5, so charging both the same rate would credit momentum
with ~3.3pp/yr of pure turnover advantage. An earlier draft reported z = +8.8 from the
cost-charged version; that figure overstates the result.

## Known defects

**1. The evaluation window has a four-month hole.** The `amihud` feature goes NaN across
the entire cross-section when a market-wide stale-volume bar falls inside its 63-day
window, and `xattn.py` drops any snapshot with a non-finite feature. This silently
removed **9 of 67 evaluation months**. The portfolio held **unrebalanced from
2025-03-03 to 2025-07-01 — 120 days** — and evaluation ends Jan 2026 rather than Jul 2026,
discarding six months of available data. Fix: `min_periods` on the rolling window, or drop
non-finite features per-name rather than per-snapshot. **All results above are computed on
the 58 dates that survived**, so they are internally consistent but do not cover the
period a reader would assume.

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
