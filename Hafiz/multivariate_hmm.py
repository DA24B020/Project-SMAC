import warnings
warnings.filterwarnings("ignore")

import os
from math import log, isfinite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

OUTDIR = "hmm_outputs_3tickers"
os.makedirs(OUTDIR, exist_ok=True)

def num_params(m, p):
    return m * (m - 1) + m * p + m * (p * (p + 1) // 2)

def stationary_from_transition(G):
    vals, vecs = np.linalg.eig(G.T)
    idx = np.argmin(np.abs(vals - 1.0))
    v = np.real(vecs[:, idx])
    pi = v / np.sum(v)
    if np.any(pi < -1e-8):
        pi = np.ones(G.shape[0]) / G.shape[0]
        for _ in range(200):
            pi = pi.dot(G)
        pi = pi / np.sum(pi)
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)
    return pi

def plot_hist_with_mixture(returns_df, mus, sigs, mix_weights, varname, outdir):
    x = returns_df[varname].values
    plt.figure(figsize=(7,4))
    plt.hist(x, bins=40, density=True, alpha=0.6, label='Empirical')
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax, 400)
    mixture_pdf = np.zeros_like(xs)
    for i, (mu, s, w) in enumerate(zip(mus, sigs, mix_weights)):
        mixture_pdf += w * norm.pdf(xs, loc=mu, scale=s)
        plt.plot(xs, w * norm.pdf(xs, loc=mu, scale=s), '--', lw=1, label=f'state {i+1} weighted')
    plt.plot(xs, mixture_pdf, lw=2, label='Mixture (model)')
    plt.title(f'Histogram + mixture overlay: {varname}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"hist_mixture_{varname}.png"))
    plt.close()

def plot_acf_compare(emp_series, sim_series, nlags, name, outdir):
    acf_emp = acf(emp_series, nlags=nlags, fft=False)
    acf_sim = acf(sim_series, nlags=nlags, fft=False)
    lags = np.arange(len(acf_emp))
    plt.figure(figsize=(7,4))
    # NOTE: removed use_line_collection argument for matplotlib compatibility
    plt.stem(lags, acf_emp, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='Empirical')
    plt.stem(lags, acf_sim, linefmt='C1--', markerfmt='C1s', basefmt='k-', label='Model (sim)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title(f'ACF comparison: {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"acf_compare_{name}.png"))
    plt.close()

def fit_gaussian_hmm(X, n_states, n_iter=200, n_restarts=3, random_state=42):
    best_model = None
    best_ll = -np.inf
    rng = np.random.RandomState(random_state)
    for r in range(n_restarts):
        seed = int(rng.randint(0, 2**30))
        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=n_iter, random_state=seed, verbose=False)
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue
    return best_model, best_ll

def main():
    tickers = ["ALV.DE", "DBK.DE", "SIE.DE"]
    start_date = "2003-03-04"
    end_date = "2005-02-17"

    print("Downloading tickers:", tickers)
    frames = []
    success = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=False)
            if df is None or df.shape[0] == 0:
                print(f"{t}: NO DATA")
                continue
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            s = df[col].rename(t)
            frames.append(s)
            success.append(t)
            print(f"{t}: OK rows={s.shape[0]} first={s.index[0].date()} last={s.index[-1].date()}")
        except Exception as e:
            print(f"{t}: ERROR {e}")

    if len(frames) < 2:
        raise SystemExit("Need at least 2 tickers with data. Exiting.")

    adj = pd.concat(frames, axis=1)
    adj = adj.dropna(axis=0, how='any')
    print(f"Aligned adjusted close matrix shape: {adj.shape}")
    adj.to_csv(os.path.join(OUTDIR, "adj_close_3tickers.csv"))

    returns = 100.0 * np.log(adj / adj.shift(1))
    returns = returns.dropna(how='any')
    n, p = returns.shape
    print(f"Returns computed: n={n}, p={p}")
    returns.to_csv(os.path.join(OUTDIR, "returns_3tickers.csv"))

    X = returns.values

    print("Fitting 2-state HMM...")
    model2, ll2 = fit_gaussian_hmm(X, n_states=2, n_restarts=4, n_iter=200)
    print("Fitting 3-state HMM...")
    model3, ll3 = fit_gaussian_hmm(X, n_states=3, n_restarts=3, n_iter=200)

    def safe_criteria(ll, m, p, n):
        if not isfinite(ll):
            return np.inf, np.inf
        k = num_params(m, p)
        AIC = -2 * ll + 2 * k
        BIC = -2 * ll + k * log(n)
        return AIC, BIC

    AIC2, BIC2 = safe_criteria(ll2, 2, p, n)
    AIC3, BIC3 = safe_criteria(ll3, 3, p, n)
    print("\nModel selection:")
    print(f"2-state: ll={ll2}, AIC={AIC2}, BIC={BIC2}")
    print(f"3-state: ll={ll3}, AIC={AIC3}, BIC={BIC3}")
    chosen = model2 if BIC2 < BIC3 else model3
    m = chosen.n_components
    print(f"Chosen by BIC: {m}-state")

    G = chosen.transmat_
    pi = stationary_from_transition(G)
    means = chosen.means_          # shape (m, p)
    covars = chosen.covars_       # shape (m, p, p)
    stds = np.array([np.sqrt(np.diag(covars[i])) for i in range(m)])

    np.set_printoptions(precision=4, suppress=True)
    print("\nTransition matrix (chosen):")
    print(G)
    print("Stationary distribution (pi):", pi)
    for i in range(m):
        print(f"\nState {i+1} mean: {means[i]}")
        print(f"State {i+1} std:  {stds[i]}")

    print("\nPer-state correlation matrices:")
    for i in range(m):
        cov = covars[i]
        D = np.sqrt(np.diag(cov))
        corr = cov / np.outer(D, D)
        print(f"\nState {i+1} correlation:\n{np.round(corr,3)}")

    tick_names = returns.columns.tolist()
    for j, var in enumerate(tick_names):
        mus = means[:, j]
        sigs = stds[:, j]
        plot_hist_with_mixture(returns, mus, sigs, pi, var, OUTDIR)
    print(f"Saved histograms to {OUTDIR}/")

    sim_len = max(2000, n * 5)
    sim_X, _ = chosen.sample(sim_len)
    nlags = 30
    for j, var in enumerate(tick_names):
        emp_sq = returns[var].values ** 2
        sim_sq = sim_X[:, j] ** 2
        plot_acf_compare(emp_sq, sim_sq, nlags, f"{var}_squared", OUTDIR)
    print("Saved ACF comparison plots.")

    pred = chosen.predict(X)
    dates = returns.index
    for j, var in enumerate(tick_names):
        plt.figure(figsize=(10,3))
        for s in range(m):
            mask = pred == s
            plt.plot(dates[mask], returns[var].values[mask], '.', label=f'state {s+1}', alpha=0.8)
        plt.title(f"{var} returns colored by Viterbi states")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"viterbi_{var}.png"))
        plt.close()
    print("Saved Viterbi plots.")


    B = 25
    print(f"\nRunning small parametric bootstrap (B={B}) - this may take a while...")
    boot_tr = []
    boot_means = []
    for b in range(B):
        sim_b_X, _ = chosen.sample(n)
        try:
            model_b, llb = fit_gaussian_hmm(sim_b_X, n_states=m, n_restarts=2, n_iter=120)
            if model_b is None:
                continue
            boot_tr.append(model_b.transmat_.flatten())
            boot_means.append(model_b.means_[0])
        except Exception:
            continue
    boot_tr = np.array(boot_tr)
    boot_means = np.array(boot_means)
    if boot_tr.size:
        print("Bootstrap mean transition matrix:\n", np.round(boot_tr.mean(axis=0).reshape(m, m), 4))
    if boot_means.size:
        print("Bootstrap mean of state-1 mean:", np.round(boot_means.mean(axis=0), 4))

    print("\nAll outputs saved into folder:", OUTDIR)
    print("Open the PNGs and CSVs in that folder for figures/tables.")

if __name__ == "__main__":
    main()
