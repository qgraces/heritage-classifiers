#!/usr/bin/env python3
"""
Figures 1 and 2 of "A Three-Child Case Study on Classifier Use in Heritage and Monolingual Mandarin",
made from data/children_tokens.csv and data/mothers_tokens.csv.

Usage:   python figures.py            (run from the repository root)
Output:  output/figure1.png, output/figure2.png

Figure 1  个 share of overt-noun tokens by age, overall and within DEM+CL and NUM+CL.
          Lines: quadratic binomial generalized linear model (logit link) fitted to session-level counts, bands are
          95% percentile intervals from 1,000 bootstrap resamples of sessions. Points: months with at least five
          overt-noun tokens in that construction (size = number of tokens). A band is omitted when three or fewer
          sessions fall below 100% (the model is then not identified).
Figure 2  share of tokens in each of the four constructions by age (same model and bands). Points: months with at
          least 10 tokens (size = number of tokens in the month, including tokens in other constructions).
"""
import os, json, zlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
SEED, NBOOT = 2026, 1000
CHILDREN = ["Avia", "Luna", "XueXue"]
COL = {"Avia": "#1b6ca8", "Luna": "#b02a30", "XueXue": "#2a9d8f"}
MINOV, MINTOK = 5, 10

with open(os.path.join(ROOT, "random_states.json")) as f:
    STATES = json.load(f)


def rng_for(key):
    """Generator state at the start of each bootstrap in the run that produced the manuscript figures."""
    if key in STATES:
        g = np.random.Generator(np.random.PCG64())
        g.bit_generator.state = STATES[key]
        return g
    return np.random.default_rng([SEED, zlib.crc32(key.encode())])


kids = pd.read_csv(os.path.join(ROOT, "data", "children_tokens.csv"), encoding="utf-8-sig", keep_default_na=False, na_values=[""])
kids["gen"] = (kids.classifier_type == "general").astype(int)
kids["month"] = kids.age_months
kids["child"] = kids.speaker


def design(age, mu, sd):
    z = (age - mu) / sd
    return np.column_stack([np.ones_like(z), z, z ** 2])


def logit_fit(X, k, n, it=50):
    b = np.zeros(X.shape[1])
    b[0] = np.log((k.sum() + .5) / (n.sum() - k.sum() + .5))
    for _ in range(it):                                              # iteratively reweighted least squares
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = n * p * (1 - p) + 1e-9
        z = eta + (k - n * p) / w
        bn = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-8 * np.eye(3), X.T @ (w * z))
        if np.max(np.abs(bn - b)) < 1e-8:
            b = bn
            break
        b = bn
    return b


def curve_binom(sess, grid, rng, B=NBOOT):
    """sess: one row per session with age_months, k (successes), n (tokens)"""
    mu, sd = sess.age_months.mean(), sess.age_months.std(ddof=1)
    Xg = design(grid, mu, sd)

    def fit(s):
        b = logit_fit(design(s.age_months.values.astype(float), mu, sd), s.k.values.astype(float), s.n.values.astype(float))
        return 100 / (1 + np.exp(-np.clip(Xg @ b, -30, 30)))

    est, boots = fit(sess), []
    for _ in range(B):
        s = sess.iloc[rng.integers(0, len(sess), len(sess))]
        try:
            boots.append(fit(s))
        except Exception:
            pass
    boots = np.array(boots)
    return est, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def sess_counts(s, flag):
    g = s.groupby("session").agg(age_months=("age_months", "first"), k=(flag, "sum"), n=(flag, "size"))
    return g.reset_index(drop=True)


def month_counts(s, flag):
    return s.groupby("month").agg(k=(flag, "sum"), n=(flag, "size"))


plt.rcParams.update({"font.family": ["DejaVu Sans", "Noto Sans CJK SC"], "font.size": 9, "axes.edgecolor": "#444", "axes.linewidth": .7})


def style(ax, c):
    ax.set_title(c, fontsize=10, fontweight="bold", color=COL[c])
    ax.grid(True, color="#e6e6e6", lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlim(18, 61)
    ax.set_xticks([24, 36, 48, 60])
    ax.set_xlabel("Age (months)")


# ---------------------------------------------------------------- Figure 1
ov = kids[kids.noun_status == "overt"]
LINES = [("DEM+CL", "#4c78a8", None), ("NUM+CL", "#f58518", None), ("Overall", "#333333", "--")]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for ax, c in zip(axs, CHILDREN):
    s_ = ov[ov.child == c]
    for lab, col, ls in LINES:
        t = s_ if lab == "Overall" else s_[s_.construction == lab]
        sc = sess_counts(t, "gen")
        grid = np.linspace(sc.age_months.min(), sc.age_months.max(), 100)
        est, lo, hi = curve_binom(sc, grid, rng_for(f"fig1|{c}|{lab}"))
        if (sc.k < sc.n).sum() > 3:
            ax.fill_between(grid, lo, hi, color=col, alpha=.13, lw=0)
        ax.plot(grid, est, color=col, lw=1.8, ls=ls or "-", label=lab)
        if lab != "Overall":
            mc = month_counts(t, "gen")
            mc = mc[mc.n >= MINOV]
            ax.scatter(mc.index, 100 * mc.k / mc.n, s=mc.n * 1.8, color=col, alpha=.65, edgecolor="white", lw=.4, zorder=3)
    style(ax, c)
    ax.set_ylim(30, 101)
axs[0].set_ylabel("Share of overt-noun tokens that are 个 gè (%)")
axs[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure1.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
CONS = [("DEM+CL", "#4c78a8"), ("NUM+CL", "#f58518"), ("CL+N", "#54a24b"), ("DEM+NUM+CL", "#9d755d")]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for ax, c in zip(axs, CHILDREN):
    s = kids[kids.child == c]
    for key, col in CONS:
        s2 = s.assign(f=(s.construction == key).astype(int))
        sc = sess_counts(s2, "f")
        grid = np.linspace(sc.age_months.min(), sc.age_months.max(), 100)
        est, lo, hi = curve_binom(sc, grid, rng_for(f"fig2|{c}|{key}"))
        ax.fill_between(grid, lo, hi, color=col, alpha=.15, lw=0)
        ax.plot(grid, est, color=col, lw=1.5, label=key)
        mc = month_counts(s2, "f")
        n_all = s.groupby("month").size()
        mc = mc[n_all.reindex(mc.index) >= MINTOK]
        ax.scatter(mc.index, 100 * mc.k / mc.n, s=n_all.reindex(mc.index) * .9, color=col, alpha=.65, edgecolor="white", lw=.3, zorder=3)
    style(ax, c)
    ax.set_ylim(-2, 100)
axs[0].set_ylabel("Share of classifier tokens (%)")
axs[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure2.png"), dpi=300)
plt.close(fig)
print("wrote output/figure1.png and output/figure2.png")
