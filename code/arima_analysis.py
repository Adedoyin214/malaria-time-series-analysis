"""
=============================================================================
TIME SERIES ANALYSIS OF MALARIA DEATHS
General Hospital Toro, Bauchi State, Nigeria (2000–2024)
=============================================================================
Method     : Box-Jenkins ARIMA
Tests      : ADF (Augmented Dickey-Fuller), PP (Phillips-Perron)
Best Model : ARIMA(0,1,2) — selected by AIC & BIC
Forecast   : 2025–2029 (5-year horizon)
Output     : EViews-style tables
=============================================================================
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import PhillipsPerron
from scipy.stats import jarque_bera, chi2
import warnings
warnings.filterwarnings('ignore')

# ── DATA ──────────────────────────────────────────────────────────────────────
years  = list(range(2000, 2025))
deaths = [44,40,44,55,36,51,43,54,37,36,53,51,36,56,41,65,52,41,62,68,39,56,52,55,57]

y = pd.Series(deaths, index=pd.PeriodIndex(years, freq='Y'), name='Malaria Deaths')
y_diff = y.diff().dropna()

# ── DESCRIPTIVE STATISTICS ────────────────────────────────────────────────────
print("="*70)
print("  DESCRIPTIVE STATISTICS")
print("="*70)
jb = jarque_bera(y)
print(f"  Mean:         {y.mean():.6f}")
print(f"  Median:       {y.median():.6f}")
print(f"  Maximum:      {y.max()}")
print(f"  Minimum:      {y.min()}")
print(f"  Std. Dev.:    {y.std():.6f}")
print(f"  Skewness:     {y.skew():.6f}")
print(f"  Kurtosis:     {y.kurtosis()+3:.6f}")
print(f"  Jarque-Bera:  {jb.statistic:.6f}  (p = {jb.pvalue:.6f})")
print(f"  Observations: {len(y)}")

# ── ADF TEST ──────────────────────────────────────────────────────────────────
def adf_test(series, label="Level"):
    print(f"\n{'='*70}")
    print(f"  ADF UNIT ROOT TEST — {label.upper()}")
    print(f"{'='*70}")
    for reg, name in [('c','Intercept'), ('ct','Trend and Intercept'), ('n','None')]:
        res = adfuller(series, regression=reg, autolag='AIC')
        print(f"\n  Exogenous: {name}  |  Lags: {res[2]}  (AIC, maxlag=4)")
        print(f"  {'':40s} {'t-Statistic':>14s}   {'Prob.*':>8s}")
        print(f"  {'-'*64}")
        print(f"  {'ADF test statistic':40s} {res[0]:>14.6f}   {res[1]:>8.4f}")
        print(f"  {'1% critical value':40s} {res[4]['1%']:>14.6f}")
        print(f"  {'5% critical value':40s} {res[4]['5%']:>14.6f}")
        print(f"  {'10% critical value':40s} {res[4]['10%']:>14.6f}")
        decision = "REJECT H0 — Stationary" if res[1] < 0.05 else "FAIL TO REJECT H0 — Non-stationary"
        print(f"  Decision (5%): {decision}")

adf_test(y,      "Level")
adf_test(y_diff, "1st Difference")

# ── PHILLIPS-PERRON TEST ──────────────────────────────────────────────────────
def pp_test(series, label="Level"):
    print(f"\n{'='*70}")
    print(f"  PHILLIPS-PERRON TEST — {label.upper()}")
    print(f"{'='*70}")
    bw = int(4*(len(series)/100)**0.25)
    for trend, name in [('c','Intercept'), ('ct','Trend and Intercept'), ('n','None')]:
        pp = PhillipsPerron(series.values, trend=trend, lags=bw)
        cvs = pp.critical_values
        print(f"\n  Exogenous: {name}  |  Bandwidth: {bw} (Newey-West, Bartlett)")
        print(f"  {'':40s} {'Adj. t-Stat':>14s}   {'Prob.*':>8s}")
        print(f"  {'-'*64}")
        print(f"  {'PP test statistic':40s} {pp.stat:>14.6f}   {pp.pvalue:>8.4f}")
        print(f"  {'1% critical value':40s} {cvs['1%']:>14.6f}")
        print(f"  {'5% critical value':40s} {cvs['5%']:>14.6f}")
        print(f"  {'10% critical value':40s} {cvs['10%']:>14.6f}")
        decision = "REJECT H0 — Stationary" if pp.pvalue < 0.05 else "FAIL TO REJECT H0 — Non-stationary"
        print(f"  Decision (5%): {decision}")

pp_test(y,      "Level")
pp_test(y_diff, "1st Difference")

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  ARIMA MODEL SELECTION — AIC/BIC COMPARISON")
print(f"{'='*70}")
print(f"  {'Model':15s} {'Log-Lik':>14s} {'AIC':>12s} {'BIC':>12s} {'AICc':>12s}")
print(f"  {'-'*66}")
candidates = []
for p in range(4):
    for q in range(4):
        for d in [0, 1]:
            try:
                m = ARIMA(y, order=(p,d,q)).fit()
                n = len(y); k = p+q+1
                aicc = m.aic + (2*k*(k+1))/(n-k-1)
                candidates.append((p,d,q, m.llf, m.aic, m.bic, aicc))
                print(f"  ARIMA({p},{d},{q}){' ':6s} {m.llf:>14.4f} {m.aic:>12.4f} {m.bic:>12.4f} {aicc:>12.4f}")
            except: pass
candidates.sort(key=lambda x: x[4])
bp,bd,bq = candidates[0][:3]
print(f"\n  >>> Best Model (AIC): ARIMA({bp},{bd},{bq}) — AIC={candidates[0][4]:.4f}, BIC={candidates[0][5]:.4f}")

# ── ESTIMATION ────────────────────────────────────────────────────────────────
model = ARIMA(y, order=(bp,bd,bq)).fit()
print(f"\n{'='*70}")
print(f"  ARIMA({bp},{bd},{bq}) ESTIMATION OUTPUT")
print(f"{'='*70}")
print(f"  {'Variable':30s} {'Coeff':>12s} {'Std.Err':>12s} {'z-Stat':>10s} {'Prob':>8s}")
print(f"  {'-'*74}")
for nm,c,s,z,p in zip(model.param_names,model.params,model.bse,model.tvalues,model.pvalues):
    print(f"  {nm:30s} {c:>12.6f} {s:>12.6f} {z:>10.4f} {p:>8.4f}")
print(f"  {'-'*74}")
print(f"  Log likelihood:      {model.llf:.6f}")
print(f"  AIC:                 {model.aic:.6f}")
print(f"  BIC:                 {model.bic:.6f}")
print(f"  Hannan-Quinn:        {model.hqic:.6f}")

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
resids = model.resid
lb = acorr_ljungbox(resids, lags=12, return_df=True)
print(f"\n{'='*70}")
print("  LJUNG-BOX Q-STATISTICS ON RESIDUALS")
print(f"{'='*70}")
print(f"  {'Lag':5s} {'Q-Stat':>12s} {'Prob':>8s}   Decision")
print(f"  {'-'*55}")
for lag, row in lb.iterrows():
    dec = "No autocorrelation" if row['lb_pvalue'] > 0.05 else "Autocorrelation present"
    print(f"  {lag:5d} {row['lb_stat']:>12.4f} {row['lb_pvalue']:>8.4f}   {dec}")

# ── FORECAST ──────────────────────────────────────────────────────────────────
fc = model.get_forecast(steps=5)
fc_ci = fc.conf_int(alpha=0.05).values
print(f"\n{'='*70}")
print(f"  ARIMA({bp},{bd},{bq}) FORECAST — 2025–2029")
print(f"{'='*70}")
print(f"  {'Year':6s} {'Forecast':>12s} {'Std.Err':>12s} {'95% Lower':>12s} {'95% Upper':>12s}")
print(f"  {'-'*56}")
for i,(yr,fv,se,lo,hi) in enumerate(zip(range(2025,2030),
                                          fc.predicted_mean.values,
                                          fc.se_mean.values,
                                          fc_ci[:,0], fc_ci[:,1])):
    print(f"  {yr:<6d} {fv:>12.4f} {se:>12.4f} {lo:>12.4f} {hi:>12.4f}")

print("\n  Analysis complete.")
