import warnings
warnings.filterwarnings("ignore")

import os
from math import log, isfinite
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

# --- Helper Functions (largely unchanged, kept outside the class as they are general utilities) ---

def num_params(m: int, p: int) -> int:
    """Calculate the number of parameters in a Gaussian HMM."""
    return m * (m - 1) + m * p + m * (p * (p + 1) // 2)

def stationary_from_transition(G: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution from a transition matrix."""
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
    return pi / np.sum(pi)

def fit_gaussian_hmm(X: np.ndarray, n_states: int, n_iter: int, n_restarts: int, random_state: int) -> Tuple[GaussianHMM, float]:
    """Fit a GaussianHMM with multiple restarts to find the best model."""
    best_model = None
    best_ll = -np.inf
    rng = np.random.RandomState(random_state)
    for _ in range(n_restarts):
        seed = int(rng.randint(0, 2**30))
        model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=n_iter, random_state=seed, verbose=False)
        try:
            model.fit(X)
            ll = model.score(X)
            if isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue
    return best_model, best_ll

# --- Plotting Functions (unchanged logic) ---

def plot_hist_with_mixture(returns_df: pd.DataFrame, mus: np.ndarray, sigs: np.ndarray, mix_weights: np.ndarray, varname: str, outdir: Path):
    """Plot the empirical histogram overlaid with the fitted mixture model."""
    x = returns_df[varname].values
    plt.figure(figsize=(7,4))
    plt.hist(x, bins=40, density=True, alpha=0.6, label='Empirical')
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax, 400)
    mixture_pdf = sum(w * norm.pdf(xs, loc=mu, scale=s) for mu, s, w in zip(mus, sigs, mix_weights))

    for i, (mu, s, w) in enumerate(zip(mus, sigs, mix_weights)):
        plt.plot(xs, w * norm.pdf(xs, loc=mu, scale=s), '--', lw=1, label=f'state {i+1} weighted')

    plt.plot(xs, mixture_pdf, lw=2, label='Mixture (model)')
    plt.title(f'Histogram + mixture overlay: {varname}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"hist_mixture_{varname}.png")
    plt.close()

def plot_acf_compare(emp_series: np.ndarray, sim_series: np.ndarray, nlags: int, name: str, outdir: Path):
    """Plot a comparison of empirical and simulated autocorrelation functions."""
    acf_emp = acf(emp_series, nlags=nlags, fft=False)
    acf_sim = acf(sim_series, nlags=nlags, fft=False)
    lags = np.arange(len(acf_emp))

    plt.figure(figsize=(7,4))
    plt.stem(lags, acf_emp, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='Empirical')
    plt.stem(lags, acf_sim, linefmt='C1--', markerfmt='C1s', basefmt='k-', label='Model (sim)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title(f'ACF comparison: {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"acf_compare_{name}.png")
    plt.close()

# --- Main Analyzer Class ---

class HMMStockAnalyzer:
    """A class to perform HMM analysis on stock ticker data."""
    def __init__(self, tickers: List[str], start_date: str, end_date: str, outdir_name: str, states_to_test: List[int]):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(outdir_name)
        self.states_to_test = states_to_test

        self.returns_df = None
        self.returns_arr = None
        self.models: Dict[int, Tuple[GaussianHMM, float]] = {}
        self.criteria: Dict[int, Dict[str, float]] = {}
        self.best_model: GaussianHMM = None

        self.output_dir.mkdir(exist_ok=True)
        np.set_printoptions(precision=4, suppress=True)

    def _fetch_and_preprocess_data(self):
        """Downloads and processes stock data to compute log returns."""
        print("Downloading tickers:", self.tickers)
        frames = []
        for t in self.tickers:
            try:
                df = yf.Ticker(t).history(start=self.start_date, end=self.end_date, auto_adjust=False)
                if df.empty:
                    print(f"{t}: NO DATA")
                    continue
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                s = df[col].rename(t)
                frames.append(s)
                print(f"{t}: OK rows={s.shape[0]} first={s.index[0].date()} last={s.index[-1].date()}")
            except Exception as e:
                print(f"{t}: ERROR {e}")

        if len(frames) < 2:
            raise SystemExit("Need at least 2 tickers with data. Exiting.")

        adj = pd.concat(frames, axis=1).dropna(axis=0, how='any')
        print(f"Aligned adjusted close matrix shape: {adj.shape}")
        adj.to_csv(self.output_dir / "adj_close_3tickers.csv")

        self.returns_df = 100.0 * np.log(adj / adj.shift(1)).dropna(how='any')
        self.returns_arr = self.returns_df.values
        n, p = self.returns_df.shape
        print(f"Returns computed: n={n}, p={p}")
        self.returns_df.to_csv(self.output_dir / "returns_3tickers.csv")

    def _fit_and_select_model(self):
        """Fits HMMs for a range of states and selects the best one using BIC."""
        fit_params = {2: {'n_restarts': 4}, 3: {'n_restarts': 3}}

        for m in self.states_to_test:
            print(f"Fitting {m}-state HMM...")
            restarts = fit_params.get(m, {'n_restarts': 3})['n_restarts']
            model, ll = fit_gaussian_hmm(self.returns_arr, n_states=m, n_iter=200, n_restarts=restarts, random_state=42)
            self.models[m] = (model, ll)

            n, p = self.returns_arr.shape
            k = num_params(m, p)
            self.criteria[m] = {
                'll': ll,
                'AIC': -2 * ll + 2 * k if isfinite(ll) else np.inf,
                'BIC': -2 * ll + k * log(n) if isfinite(ll) else np.inf
            }

        print("\nModel selection:")
        for m in self.states_to_test:
            c = self.criteria[m]
            print(f"{m}-state: ll={c['ll']}, AIC={c['AIC']}, BIC={c['BIC']}")

        best_m = min(self.criteria, key=lambda m: self.criteria[m]['BIC'])
        self.best_model = self.models[best_m][0]
        print(f"Chosen by BIC: {best_m}-state")

    def _analyze_best_model(self):
        """Prints parameters and correlation matrices of the chosen model."""
        if not self.best_model: return
        m = self.best_model.n_components

        G = self.best_model.transmat_
        pi = stationary_from_transition(G)
        means = self.best_model.means_
        covars = self.best_model.covars_
        stds = np.array([np.sqrt(np.diag(cov)) for cov in covars])

        print("\nTransition matrix (chosen):\n", G)
        print("Stationary distribution (pi):", pi)
        for i in range(m):
            print(f"\nState {i+1} mean: {means[i]}")
            print(f"State {i+1} std:  {stds[i]}")

        print("\nPer-state correlation matrices:")
        for i in range(m):
            cov = covars[i]
            D = np.sqrt(np.diag(cov))
            corr = cov / np.outer(D, D)
            print(f"\nState {i+1} correlation:\n{np.round(corr, 3)}")

    def _generate_plots(self):
        """Generates and saves all diagnostic plots."""
        if not self.best_model: return

        # 1. Histogram and Mixture Plots
        tick_names = self.returns_df.columns
        m, p = self.best_model.means_.shape
        pi = stationary_from_transition(self.best_model.transmat_)
        for j, var in enumerate(tick_names):
            mus = self.best_model.means_[:, j]
            sigs = np.array([np.sqrt(self.best_model.covars_[i, j, j]) for i in range(m)])
            plot_hist_with_mixture(self.returns_df, mus, sigs, pi, var, self.output_dir)
        print(f"Saved histograms to {self.output_dir}/")

        # 2. ACF Comparison Plots
        sim_len = max(2000, self.returns_arr.shape[0] * 5)
        sim_X, _ = self.best_model.sample(sim_len)
        nlags = 30
        for j, var in enumerate(tick_names):
            emp_sq = self.returns_df[var].values ** 2
            sim_sq = sim_X[:, j] ** 2
            plot_acf_compare(emp_sq, sim_sq, nlags, f"{var}_squared", self.output_dir)
        print("Saved ACF comparison plots.")

        # 3. Viterbi State Plots
        pred = self.best_model.predict(self.returns_arr)
        dates = self.returns_df.index
        for j, var in enumerate(tick_names):
            plt.figure(figsize=(10, 3))
            for s in range(m):
                mask = pred == s
                plt.plot(dates[mask], self.returns_df[var].values[mask], '.', label=f'state {s+1}', alpha=0.8)
            plt.title(f"{var} returns colored by Viterbi states")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"viterbi_{var}.png")
            plt.close()
        print("Saved Viterbi plots.")

    def _run_parametric_bootstrap(self, B: int = 25):
        """Runs a small parametric bootstrap to check parameter stability."""
        if not self.best_model: return

        print(f"\nRunning small parametric bootstrap (B={B}) - this may take a while...")
        m = self.best_model.n_components
        n = self.returns_arr.shape[0]
        boot_tr, boot_means = [], []

        for _ in range(B):
            sim_b_X, _ = self.best_model.sample(n)
            try:
                model_b, _ = fit_gaussian_hmm(sim_b_X, n_states=m, n_iter=120, n_restarts=2, random_state=42)
                if model_b:
                    boot_tr.append(model_b.transmat_.flatten())
                    boot_means.append(model_b.means_[0])
            except Exception:
                continue

        if boot_tr:
            boot_tr_arr = np.array(boot_tr)
            print("Bootstrap mean transition matrix:\n", np.round(boot_tr_arr.mean(axis=0).reshape(m, m), 4))
        if boot_means:
            boot_means_arr = np.array(boot_means)
            print("Bootstrap mean of state-1 mean:", np.round(boot_means_arr.mean(axis=0), 4))

    def run_full_analysis(self):
        """Executes the entire analysis pipeline."""
        self._fetch_and_preprocess_data()
        self._fit_and_select_model()
        self._analyze_best_model()
        self._generate_plots()
        self._run_parametric_bootstrap()

        print("\nAll outputs saved into folder:", self.output_dir)
        print("Open the PNGs and CSVs in that folder for figures/tables.")

if __name__ == "__main__":
    # --- Configuration ---
    TICKERS = ["ALV.DE", "DBK.DE", "SIE.DE"]
    START_DATE = "2003-03-04"
    END_DATE = "2005-02-17"
    OUTPUT_DIR = "hmm_outputs_3tickers" # Use same name to show output is identical
    STATES_TO_TEST = [2, 3]

    # --- Run Analysis ---
    analyzer = HMMStockAnalyzer(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        outdir_name=OUTPUT_DIR,
        states_to_test=STATES_TO_TEST
    )
    analyzer.run_full_analysis()
