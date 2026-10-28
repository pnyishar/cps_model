#!/usr/bin/env python3
"""
Reproducible pipeline for "Hybrid Mathematical Modeling of Cyber-Physical System
Vulnerabilities: A Water Treatment Case Study".

    .venv/bin/python run_all.py            # full pipeline -> results/
    .venv/bin/python run_all.py --upto 3   # stop after step 3 (identification + LQR checks)

Data: SWaT 2021 collection, session 2021-06-25 (see README.md for why).
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
CHECK = os.path.join(OUT, "checks")

# Configuration
SESSION_FILES = [  # SWaT 2021, session of 2021-06-25 (12:13 - 17:00)
    "SWaT/_20210625_121324/20210625_121324.csv",
    "SWaT/_20210625_121324/20210625_131629.csv",
    "SWaT/_20210625_121324/20210625_141937.csv",
    "SWaT/_20210625_121324/20210625_152242.csv",
    "SWaT/_20210625_121324/20210625_162535.csv",
]
STATES = ["FIT101", "LIT101", "FIT201", "LIT301"]
INPUTS = ["MV101", "P101", "MV201", "P301"]
# Status code -> actuator command (fraction open / pump on).
# SWaT convention: MV 0 = in transition, 1 = closed, 2 = open; P 1 = off, 2 = on.
MV_MAP = {0: 0.5, 1: 0.0, 2: 1.0}
P_MAP = {1: 0.0, 2: 1.0}
TRAIN_FRAC = 0.70
MAX_FILL_GAP_S = 3          # gaps <= 3 s are zero-order-held on the 1 s grid
Q_W, R_W = 1.0, 0.1         # LQR weights Q = Q_W*I, R = R_W*I

T_SIM, K_A0, K_A1 = 3600, 600, 1200
N_SEEDS = 20
SEEDS = list(range(1000, 1000 + N_SEEDS))
X0_STD = 0.05               # initial perturbation std (normalized units)
P_A = 120.0                 # FDI sinusoid period (s)
CASES = [  # key, label, magnitude
    ("dos_zero", "DoS (zero input)", 0.0),
    ("dos_hold", "DoS (hold last value)", 0.0),
    ("fdi_03", "Actuator FDI", 0.3),
    ("fdi_08", "Actuator FDI", 0.8),
    ("bias", "Bias injection", 0.5),
    ("spoof", "Sensor spoofing", 0.5),
]
ALARMS_MM = {"LIT101": (500.0, 800.0), "LIT301": (800.0, 1000.0)}  # (L, H)
DELTA_BARS = [0.1, 0.3, 0.5]
REDUCE_EVERY = 20
NOISE_SIGMAS = 3.0
BETA, N_PROP, AFFECT_THR = 0.5, 50, 0.5


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt_pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


# Data loading and preprocessing
def load_session():
    frames = [pd.read_csv(os.path.join(ROOT, f), low_memory=False) for f in SESSION_FILES]
    raw = pd.concat(frames, ignore_index=True)
    raw["t"] = pd.to_datetime(raw["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
    raw = raw[raw["t"].dt.year > 2000]                      # drop 1970 placeholder rows
    for c in STATES + INPUTS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")      # "Bad Input" objects -> NaN
    n_bad = int(raw[STATES + INPUTS].isna().any(axis=1).sum())
    raw = raw.dropna(subset=STATES + INPUTS).drop_duplicates("t").sort_values("t")
    raw = raw.set_index("t")[STATES + INPUTS]

    grid = pd.date_range(raw.index[0], raw.index[-1], freq="1s")
    df = raw.reindex(grid)
    df["observed"] = df[STATES[0]].notna()
    df = df.ffill(limit=MAX_FILL_GAP_S)
    if df[STATES + INPUTS].isna().any().any():
        raise RuntimeError("gap longer than MAX_FILL_GAP_S in session; segmenting needed")

    unmapped = {c: sorted(set(df[c].unique()) - set((MV_MAP if c.startswith("MV") else P_MAP)))
                for c in INPUTS}
    if any(unmapped.values()):
        raise RuntimeError(f"unmapped actuator codes: {unmapped}")
    for c in INPUTS:
        df[c] = df[c].map(MV_MAP if c.startswith("MV") else P_MAP).astype(float)

    info = dict(n_raw=len(raw), n_bad_rows=n_bad, n_grid=len(df),
                n_filled=int((~df["observed"]).sum()),
                start=str(df.index[0]), end=str(df.index[-1]))
    return df, info


def split_normalize(df):
    n_tr = int(round(TRAIN_FRAC * len(df)))
    tr, va = df.iloc[:n_tr], df.iloc[n_tr:]
    cols = STATES + INPUTS
    mu, sd = tr[cols].mean(), tr[cols].std(ddof=0)
    if (sd == 0).any():
        raise RuntimeError(f"zero-variance signal in training window: {list(sd[sd == 0].index)}")
    z = (df[cols] - mu) / sd
    z["observed"] = df["observed"]

    def win(w):
        return dict(start=str(w.index[0]), end=str(w.index[-1]), n=len(w),
                    duration_s=int((w.index[-1] - w.index[0]).total_seconds()) + 1)
    return z, n_tr, mu, sd, dict(train=win(tr), validation=win(va))


# System identification
def pairs(z, lo, hi):
    """Consecutive (k, k+1) pairs inside [lo, hi) where both samples were actually observed."""
    X, U, ok = z[STATES].values, z[INPUTS].values, z["observed"].values
    k = np.arange(lo, hi - 1)
    k = k[ok[k] & ok[k + 1]]
    return X[k], U[k], X[k + 1]


def identify(z, n_tr, offset):
    """x[k+1] = A x[k] + B u[k] (+ c if offset) by numpy.linalg.lstsq on the training window."""
    Xk, Uk, Xk1 = pairs(z, 0, n_tr)
    Phi = np.hstack([Xk, Uk] + ([np.ones((len(Xk), 1))] if offset else []))
    theta, *_ = np.linalg.lstsq(Phi, Xk1, rcond=None)
    A, B = theta[:4].T, theta[4:8].T
    c = theta[8] if offset else np.zeros(4)
    resid = Xk1 - Phi @ theta
    return A, B, c, resid, len(Xk)


def fit_pct(y, yhat):
    return 100.0 * (1.0 - np.linalg.norm(y - yhat, axis=0) / np.linalg.norm(y - y.mean(0), axis=0))


def validate(z, n_tr, A, B, c):
    Xk, Uk, Xk1 = pairs(z, n_tr, len(z))
    one = fit_pct(Xk1, Xk @ A.T + Uk @ B.T + c)
    X, U, ok = z[STATES].values[n_tr:], z[INPUTS].values[n_tr:], z["observed"].values[n_tr:]
    xs = np.zeros_like(X)
    xs[0] = X[0]
    for k in range(len(X) - 1):
        xs[k + 1] = A @ xs[k] + B @ U[k] + c
    free = fit_pct(X[ok], xs[ok])
    return one, free, xs


# LQR controller
def dlqr(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# Attack simulation and metrics
def simulate(A, B, K, Lw, case, seed, x0=None):
    """Closed-loop simulation with u = -K y; returns y and u, each (T x 4)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, X0_STD, 4) if x0 is None else x0.copy()
    W = rng.standard_normal((T_SIM, 4)) @ Lw.T
    Y, Uo = np.zeros((T_SIM, 4)), np.zeros((T_SIM, 4))
    u_hold = None
    for k in range(T_SIM):
        y = x
        att = (case is not None) and (K_A0 <= k < K_A1)
        y_ctrl = y + 0.5 if (att and case == "spoof") else y
        u = -K @ y_ctrl
        if k == K_A0 - 1:
            u_hold = u.copy()
        if att:
            if case == "dos_zero":
                u = np.zeros(4)
            elif case == "dos_hold":
                u = u_hold.copy()
            elif case in ("fdi_03", "fdi_08"):
                amp = 0.3 if case == "fdi_03" else 0.8
                u = u + amp * np.sin(2 * np.pi * k / P_A)
            elif case == "bias":
                u = u + 0.5
        Y[k], Uo[k] = y, u
        x = A @ x + B @ u + W[k]
    return Y, Uo


def metrics(Y, Yref, delta):
    e = np.linalg.norm(Y - Yref, axis=1)
    peak = e.max()
    rms = np.sqrt(np.mean(e[K_A0:K_A1] ** 2))
    thr = 0.05 * peak
    post = e[K_A1:]
    above = np.nonzero(post >= thr)[0]
    if len(above) == 0:
        rec = 0.0
    elif above[-1] == len(post) - 1:
        rec = np.nan                                  # not recovered
    else:
        rec = float(above[-1] + 1)                    # first k after which it stays below
    R = 1.0 - np.mean(np.minimum(1.0, e / delta))
    return dict(peak=peak, rms=rms, rec=rec, R=R)


# Zonotope reachability
class Zonotope:
    def __init__(self, c, G):
        self.c, self.G = np.asarray(c, float), np.asarray(G, float)

    def map(self, M):
        return Zonotope(M @ self.c, M @ self.G)

    def add_gens(self, Gn):
        return Zonotope(self.c, np.hstack([self.G, Gn]))

    def box(self):
        return Zonotope(self.c, np.diag(np.abs(self.G).sum(1)))

    def interval(self):
        r = np.abs(self.G).sum(1)
        return self.c - r, self.c + r

    def contains(self, x, tol=1e-9):
        """Exact membership: exists beta in [-1,1]^m with G beta = x - c (LP feasibility)."""
        from scipy.optimize import linprog
        m = self.G.shape[1]
        if m == 0:
            return bool(np.all(np.abs(x - self.c) <= tol))
        res = linprog(np.zeros(m), A_eq=self.G, b_eq=x - self.c, bounds=[(-1, 1)] * m,
                      method="highs")
        return res.status == 0

    def polygon2d(self, idx):
        """Vertices of the 2-D projection onto state indices idx (ordered, convex)."""
        c, G = self.c[list(idx)], self.G[list(idx)]
        G = G[:, np.linalg.norm(G, axis=0) > 1e-12]
        G = np.where(G[1] < 0, -G, G)  # all generators point to upper half-plane
        ang = np.arctan2(G[1], G[0])
        G = G[:, np.argsort(ang)]
        start = c - G.sum(1)
        pts = [start]
        for g in G.T:
            pts.append(pts[-1] + 2 * g)
        for g in G.T:
            pts.append(pts[-1] - 2 * g)
        return np.array(pts)


def reach(A_cl, B, w_half, Z0, dbar, k0=K_A0, k1=K_A1, reduce_every=REDUCE_EVERY):
    """Propagate from Z0 at k0 to k1. Returns list of zonotopes Z[k] for k=k0..k1 (pre-reduction)."""
    Bd = B @ np.diag(np.full(B.shape[1], dbar))
    Wg = np.diag(w_half)
    Z, out = Z0, [Z0]
    for i, k in enumerate(range(k0, k1)):
        Z = Z.map(A_cl).add_gens(Bd).add_gens(Wg)
        out.append(Z)
        if (i + 1) % reduce_every == 0:
            Z = Z.box()
    return out

# Dependency graph
STAGE = {"FIT101": 1, "LIT101": 1, "MV101": 1, "P101": 1, "PLC1": 1,
         "FIT201": 2, "MV201": 2, "PLC2": 2,
         "LIT301": 3, "P301": 3, "PLC3": 3, "SCADA": 0}
SENSORS = {"FIT101": "PLC1", "LIT101": "PLC1", "FIT201": "PLC2", "LIT301": "PLC3"}
ACTUATORS = {"PLC1": ["MV101", "P101"], "PLC2": ["MV201"], "PLC3": ["P301"]}
PHYS_PATH = ["MV101", "LIT101", "P101", "FIT201", "MV201", "LIT301", "P301"]


def build_graph(A_cl, B, K):
    """Directed weighted graph. Physical edges use normalized |B_d|, |K| or |A_cl| entries; cyber edges 1."""
    si, ui = {s: i for i, s in enumerate(STATES)}, {u: i for i, u in enumerate(INPUTS)}
    offA = np.abs(A_cl - np.diag(np.diag(A_cl))).max()
    G = nx.DiGraph()
    G.add_nodes_from(STAGE)
    src = {}
    for s, plc in SENSORS.items():
        G.add_edge(s, plc, weight=1.0, kind="cyber"); src[(s, plc)] = "sensor->PLC (1)"
    for plc, acts in ACTUATORS.items():
        for a in acts:
            G.add_edge(plc, a, weight=1.0, kind="cyber"); src[(plc, a)] = "PLC->actuator (1)"
    for plc in ACTUATORS:
        for e in [("SCADA", plc), (plc, "SCADA")]:
            G.add_edge(*e, weight=1.0, kind="cyber"); src[e] = "SCADA<->PLC (1)"
    for e in [("PLC1", "PLC2"), ("PLC2", "PLC1"), ("PLC2", "PLC3"), ("PLC3", "PLC2")]:
        G.add_edge(*e, weight=1.0, kind="cyber"); src[e] = "inter-PLC (1)"
    for a, b in zip(PHYS_PATH[:-1], PHYS_PATH[1:]):
        if a in ui and b in si:
            w = abs(B[si[b], ui[a]]) / np.abs(B).max(); how = f"|B_d[{b},{a}]|/max|B_d|"
        elif a in si and b in ui:
            w = abs(K[ui[b], si[a]]) / np.abs(K).max(); how = f"|K[{b},{a}]|/max|K|"
        else:
            w = abs(A_cl[si[b], si[a]]) / offA; how = f"|A_cl[{b},{a}]|/max offdiag|A_cl|"
        G.add_edge(a, b, weight=float(w), kind="physical"); src[(a, b)] = f"{how} = {w:.4g}"
    return G, src


def propagate(G, seed_node):
    nodes = list(G.nodes)
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight")   # W[i, j] = weight of i -> j
    s = np.zeros(len(nodes)); s[nodes.index(seed_node)] = 1.0
    for _ in range(N_PROP):
        s = np.minimum(1.0, s + BETA * W.T @ s)
    return dict(zip(nodes, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=7)
    ap.add_argument("--no-offset", action="store_true",
                    help="identify Eq. 8 exactly (no constant term) and use it downstream")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CHECK, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                         "mathtext.fontset": "dejavuserif"})
    S = {}

    # Step 1: preprocessing
    df, info = load_session()
    z, n_tr, mu, sd, windows = split_normalize(df)
    S.update(data=info, windows=windows)
    print("[1] data", json.dumps(info, indent=1))
    print("[1] windows", json.dumps(windows, indent=1))
    print("[1] means\n", mu.to_string(), "\n[1] stds\n", sd.to_string())

    # Step 2: identification, with and without offset
    ident = {}
    for name, off in [("eq8", False), ("eq8_offset", True)]:
        A_, B_, c_, res_, npairs = identify(z, n_tr, off)
        one_, free_, xs_ = validate(z, n_tr, A_, B_, c_)
        ident[name] = dict(A=A_, B=B_, c=c_, resid=res_, one=one_, free=free_, xs=xs_,
                           eig=np.linalg.eigvals(A_))
        print(f"[2] model {name}: lstsq on {npairs} pairs")
        for i, s in enumerate(STATES):
            print(f"[2]   {s:7s} one-step {one_[i]:7.2f}%  free-run {free_[i]:8.2f}%")
        print("[2]   |eig(A_d)|", np.round(np.abs(ident[name]["eig"]), 6))
    used = "eq8" if args.no_offset else "eq8_offset"
    M = ident[used]
    A, B, c, resid = M["A"], M["B"], M["c"], M["resid"]
    eig_A = M["eig"]
    if np.abs(eig_A).max() >= 1:
        raise RuntimeError(f"A_d not Schur stable: |λ| = {np.abs(eig_A)}")
    Sw = np.cov(resid.T)
    Lw = np.linalg.cholesky(Sw)
    print(f"[2] model used downstream: {used}")
    print("[2] A_d=\n", np.array2string(A, precision=6), "\n[2] B_d=\n", np.array2string(B, precision=6))
    print("[2] residual std", np.sqrt(np.diag(Sw)))

    fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    Xv = z[STATES].values[n_tr:]
    for i, s in enumerate(STATES):
        ax[i].plot(Xv[:, i], "k", lw=.7, label="measured")
        ax[i].plot(ident["eq8"]["xs"][:, i], lw=.7, label="free-run, Eq. 8")
        ax[i].plot(ident["eq8_offset"]["xs"][:, i], "--", lw=.7, label="free-run, Eq. 8 + offset")
        ax[i].set_ylabel(f"{s}\nnormalized deviation")
    ax[0].legend(fontsize=7); ax[-1].set_xlabel("time since start of validation window (s)")
    fig.savefig(os.path.join(CHECK, "check_freerun_validation.png"), dpi=110); plt.close(fig)

    # Step 3: LQR
    Q, R = Q_W * np.eye(4), R_W * np.eye(4)
    K, P = dlqr(A, B, Q, R)
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    print("[3] K=\n", np.array2string(K, precision=4))
    print("[3] eig(A_cl)", np.round(eig_cl, 6), "|λ|", np.round(np.abs(eig_cl), 6))
    if np.abs(eig_cl).max() >= 1:
        raise RuntimeError("closed loop not Schur stable with Q=I, R=0.1I")
    if args.upto <= 3:
        return

    # Normalized alarm limits and Delta
    lim = {t: tuple((v - mu[t]) / sd[t] for v in ALARMS_MM[t]) for t in ALARMS_MM}
    half = {t: (ALARMS_MM[t][1] - ALARMS_MM[t][0]) / 2 / sd[t] for t in ALARMS_MM}
    delta_tag = min(half, key=half.get)
    DELTA = half[delta_tag]
    obs_rng = {t: (float(df[t].min()), float(df[t].max())) for t in ALARMS_MM}
    print(f"[5] normalized alarm limits {lim}; half-widths {half}; Δ = {DELTA:.4f} ({delta_tag})")
    print(f"[5] observed range in session (mm): {obs_rng}")

    # Steps 4-5: attack simulations and metrics
    runs = {"nominal": []}
    for key, *_ in CASES:
        runs[key] = []
    for sd_ in SEEDS:
        runs["nominal"].append(simulate(A, B, K, Lw, None, sd_))
        for key, *_ in CASES:
            runs[key].append(simulate(A, B, K, Lw, key, sd_))
    met = {}
    for key, label, mag in CASES:
        ms = [metrics(runs[key][i][0], runs["nominal"][i][0], DELTA) for i in range(N_SEEDS)]
        met[key] = {m: np.array([x[m] for x in ms]) for m in ms[0]}
        r = met[key]
        print(f"[5] {key:9s} peak {fmt_pm(r['peak'].mean(), r['peak'].std())}  "
              f"rms {fmt_pm(r['rms'].mean(), r['rms'].std())}  rec {np.nanmean(r['rec']) if np.isfinite(r['rec']).any() else 'nan'} "
              f"(not rec: {np.isnan(r['rec']).sum()})  R {fmt_pm(r['R'].mean(), r['R'].std(), 4)}")
    # nominal runs: max |y| per channel
    nom = np.array([y for y, _ in runs["nominal"]])
    print("[4] nominal |y| max over seeds/time per channel:", np.abs(nom).max(axis=(0, 1)))

    # check plot: LIT101, seed 0, all cases
    fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for a, (key, label, mag) in zip(ax.flat, CASES):
        a.plot(runs["nominal"][0][0][:, 1], lw=.6); a.plot(runs[key][0][0][:, 1], lw=.6); a.set_title(key)
    fig.savefig(os.path.join(CHECK, "check_cases_seed0.png"), dpi=100); plt.close(fig)

    # Step 6: zonotope reachability
    iL1, iL3 = STATES.index("LIT101"), STATES.index("LIT301")
    w_half = NOISE_SIGMAS * np.sqrt(np.diag(Sw))
    Zinit = Zonotope(np.zeros(4), np.diag(np.full(4, NOISE_SIGMAS * X0_STD)))
    Z600 = reach(A_cl, B, w_half, Zinit, 0.0, k0=0, k1=K_A0)[-1].box()   # noise-only set at k=600

    def crossing(Zs):
        lo = np.array([z_.interval()[0] for z_ in Zs]); hi = np.array([z_.interval()[1] for z_ in Zs])
        out = {}
        for t, i in [("LIT101", iL1), ("LIT301", iL3)]:
            under = np.nonzero(lo[:, i] < lim[t][0])[0]
            over = np.nonzero(hi[:, i] > lim[t][1])[0]
            out[t] = dict(underflow=bool(len(under)), overflow=bool(len(over)),
                          first_over_k=int(K_A0 + over[0]) if len(over) else None,
                          first_under_k=int(K_A0 + under[0]) if len(under) else None,
                          lo_1200=float(lo[-1, i]), hi_1200=float(hi[-1, i]),
                          lo_min=float(lo[:, i].min()), hi_max=float(hi[:, i].max()))
        return out, any(v["underflow"] or v["overflow"] for v in out.values())

    def analyze(Z0, wh, tag):
        sets, info = {}, {}
        for db in DELTA_BARS + [0.8]:
            sets[db] = reach(A_cl, B, wh, Z0, db)
            info[db], crossed = crossing(sets[db])
            print(f"[6:{tag}] δ̄={db}: crosses={crossed} " + json.dumps(
                {t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for t, d in info[db].items()}))
        crossed0 = crossing(reach(A_cl, B, wh, Z0, 0.0))[1]
        if crossed0:
            dmin = 0.0
        else:
            lo_b, hi_b = 0.0, 1.0
            while not crossing(reach(A_cl, B, wh, Z0, hi_b))[1]:
                lo_b, hi_b = hi_b, hi_b * 2
                if hi_b > 1e3:
                    raise RuntimeError("no crossing found for δ̄ up to 1000")
            while hi_b - lo_b > 1e-4:
                mid = 0.5 * (lo_b + hi_b)
                lo_b, hi_b = (lo_b, mid) if crossing(reach(A_cl, B, wh, Z0, mid))[1] else (mid, hi_b)
            dmin = hi_b
        crit = crossing(reach(A_cl, B, wh, Z0, dmin))[0]
        print(f"[6:{tag}] set with δ̄=0 crosses: {crossed0};  minimal crossing δ̄ = {dmin:.4f}  {crit}")
        return dict(sets=sets, info=info, crossed0=crossed0, dbar_min=dmin, crit=crit)

    # (a) absolute state, 3-sigma noise box per step, starting from the noise-only set at k=600
    RA = analyze(Z600, w_half, "spec")
    # (b) attack-induced deviation e = x - x_ref, e[600] = 0, no noise, at the operating point
    RB = analyze(Zonotope(np.zeros(4), np.zeros((4, 0))), np.zeros(4), "deviation")

    # Monte Carlo containment (LP membership every 5 steps and at k=1200)
    check_ks = sorted(set(range(K_A0, K_A1 + 1, 5)) | {K_A1})
    contain = {}
    for which, RR in [("spec", RA), ("deviation", RB)]:
        for case, db in [("nominal", 0.1), ("fdi_03", 0.3), ("bias", 0.5), ("fdi_08", 0.8), ("fdi_08", 0.5)]:
            if which == "deviation" and case == "nominal":
                continue
            Zs = RR["sets"][db]
            n_out, n_tot, seeds_out = 0, 0, set()
            for i in range(N_SEEDS):
                Y = runs[case][i][0] - (runs["nominal"][i][0] if which == "deviation" else 0.0)
                for k in check_ks:
                    n_tot += 1
                    if not Zs[k - K_A0].contains(Y[k]):
                        n_out += 1; seeds_out.add(i)
            contain[f"{which}:{case}@{db}"] = dict(points_checked=n_tot, points_outside=n_out,
                                                   seeds_with_violation=len(seeds_out))
            print(f"[6:{which}] containment {case} in δ̄={db}: {n_out}/{n_tot} points outside, "
                  f"{len(seeds_out)}/{N_SEEDS} seeds with any violation")

    # Step 7: dependency graph
    G, edge_src = build_graph(A_cl, B, K)
    affected, final_s = {}, {}
    for n in G.nodes:
        s = propagate(G, n)
        final_s[n] = s
        affected[n] = sum(1 for m, v in s.items() if m != n and v > AFFECT_THR)
    btw = nx.betweenness_centrality(G, normalized=True)          # unweighted (hop) paths
    rank = sorted(G.nodes, key=lambda n: (-affected[n], -btw[n], n))
    print("[7] affected:", affected); print("[7] betweenness:", {k: round(v, 4) for k, v in btw.items()})
    print("[7] ranking:", rank[:5])

    # Tables
    eA = sorted(eig_A, key=lambda v: -abs(v)); eC = sorted(eig_cl, key=lambda v: -abs(v))
    cfmt = lambda v: (f"{v.real:.6f}{v.imag:+.6f}j" if abs(v.imag) > 1e-12 else f"{v.real:.6f}") + f" (|λ|={abs(v):.6f})"
    t1 = pd.DataFrame({
        "Output": STATES,
        "Validation fit % (one-step / free-run)":
            [f"{ident[used]['one'][i]:.2f} / {ident[used]['free'][i]:.2f}" for i in range(4)],
        "Eq.8 without offset: fit % (one-step / free-run)":
            [f"{ident['eq8']['one'][i]:.2f} / {ident['eq8']['free'][i]:.2f}" for i in range(4)],
        "Open-loop eigenvalue (A_d)": [cfmt(v) for v in eA],
        "Closed-loop eigenvalue (A_d - B_d K)": [cfmt(v) for v in eC],
    })
    t1.to_csv(os.path.join(OUT, "table1_identification.csv"), index=False)

    def rec_str(r):
        nrec = np.isnan(r).sum()
        if nrec == len(r):
            return "not recovered"
        s = fmt_pm(np.nanmean(r), np.nanstd(r), 1)
        return s + (f" ({nrec}/{len(r)} not recovered)" if nrec else "")
    t2 = pd.DataFrame([{
        "Attack": label, "Magnitude": ("u = 0" if key == "dos_zero" else "u = u[599]" if key == "dos_hold"
                                       else f"A = {mag}, P_a = {P_A:g} s" if key.startswith("fdi")
                                       else f"b = {mag}" if key == "bias" else f"+{mag} on all y"),
        "Peak deviation": fmt_pm(met[key]["peak"].mean(), met[key]["peak"].std()),
        "RMS deviation": fmt_pm(met[key]["rms"].mean(), met[key]["rms"].std()),
        "Recovery time (s)": rec_str(met[key]["rec"]),
        "R": fmt_pm(met[key]["R"].mean(), met[key]["R"].std(), 4)} for key, label, mag in CASES])
    t2.to_csv(os.path.join(OUT, "table2_attack_impact.csv"), index=False)

    t3 = pd.DataFrame([{"Rank": i + 1, "Node": n, "Components affected": affected[n],
                        "Betweenness": round(btw[n], 4)} for i, n in enumerate(rank)])
    t3.head(5).to_csv(os.path.join(OUT, "table3_graph.csv"), index=False)
    t3.to_csv(os.path.join(OUT, "table3_graph_all_nodes.csv"), index=False)

    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("# Tables\n\n## Table 1. Identification (validation window)\n\n")
        f.write(t1.to_markdown(index=False) if hasattr(t1, "to_markdown") else t1.to_string())
        f.write("\n\nEigenvalues are listed once each (sorted by |λ|); a row's eigenvalue is not tied to that row's output.\n")
        f.write("\n## Table 2. Attack impact (mean ± std over 20 seeds)\n\n")
        f.write(t2.to_markdown(index=False))
        f.write(f"\n\nΔ = {DELTA:.4f} (normalized). Deviation = ‖y − y_ref‖₂ over the 4 outputs.\n")
        f.write("\n## Table 3. Dependency graph, top 5 nodes\n\n")
        f.write(t3.head(5).to_markdown(index=False))
        f.write("\n\nAll nodes:\n\n" + t3.to_markdown(index=False) + "\n")

    # Figures
    lab = {s: f"{s} normalized deviation" for s in STATES}
    shade = dict(color="0.85", zorder=0)
    t = np.arange(T_SIM)
    # Fig. 3 seed: median FDI-0.8 peak deviation
    pk = met["fdi_08"]["peak"]
    rep = int(np.argsort(pk)[len(pk) // 2])
    Yn, Ya = runs["nominal"][rep][0], runs["fdi_08"][rep][0]
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.75)
    for i, s in enumerate(STATES):
        a = fig.add_subplot(gs[0, i])
        a.axvspan(K_A0, K_A1, **shade, label="attack window")
        a.plot(t, Ya[:, i], color="C3", lw=.7, label="FDI attack")
        a.plot(t, Yn[:, i], color="C0", lw=.5, label="normal")
        a.set_xlabel("time (s)"); a.set_ylabel(f"{s}\nnormalized deviation"); a.set_title(s)
        if i == 0:
            a.legend(fontsize=7, loc="best")
    for j, (p, q) in enumerate([("FIT101", "FIT201"), ("LIT101", "LIT301")]):
        a = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        ip, iq = STATES.index(p), STATES.index(q)
        a.plot(Ya[:, ip], Ya[:, iq], color="C3", lw=.6, alpha=.8, label="FDI attack")
        a.plot(Yn[:, ip], Yn[:, iq], color="C0", lw=.5, alpha=.8, label="normal")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        a.set_xlabel(lab[p]); a.set_ylabel(lab[q]); a.legend(fontsize=7)
    fig.suptitle(f"Actuator FDI, $A$ = 0.8, $P_a$ = {P_A:g} s (seed {SEEDS[rep]})", y=0.98)
    save_fig(fig, "fig3_fdi_normal_vs_attack")

    fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for a, (key, label, mag) in zip(ax.flat, CASES):
        Ln = np.array([y[:, 1] for y, _ in runs["nominal"]]); La = np.array([y[:, 1] for y, _ in runs[key]])
        a.axvspan(K_A0, K_A1, **shade)
        for L_, col, nm in [(Ln, "C0", "normal"), (La, "C3", "attack")]:
            m_, s_ = L_.mean(0), L_.std(0)
            a.fill_between(t, m_ - s_, m_ + s_, color=col, alpha=.25, lw=0)
            a.plot(t, m_, color=col, lw=.8, label=f"{nm} (mean ± std)")
        ttl = {"dos_zero": "DoS (zero input)", "dos_hold": "DoS (hold last value)",
               "fdi_03": "Actuator FDI, A = 0.3", "fdi_08": "Actuator FDI, A = 0.8",
               "bias": "Bias injection, b = 0.5", "spoof": "Sensor spoofing, +0.5"}[key]
        a.set_title(ttl); a.set_ylabel(lab["LIT101"])
    for a in ax[1]:
        a.set_xlabel("time (s)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig4_attack_scenarios")

    fig, axs = plt.subplots(2, 3, figsize=(12, 8.4))
    overlay = {0.1: ("nominal", "no attack"), 0.3: ("fdi_03", "FDI A = 0.3"), 0.5: ("bias", "bias b = 0.5")}
    for row, (RR, rowname) in enumerate([(RA, "(a) Absolute state, 3σ noise bound"),
                                         (RB, "(b) Attack-induced deviation x − x_ref at operating point")]):
        for a, db in zip(axs[row], DELTA_BARS):
            poly = RR["sets"][db][-1].polygon2d((iL1, iL3))
            a.fill(poly[:, 0], poly[:, 1], color="C2", alpha=.3, label=f"reach set, k = {K_A1}")
            if row == 0:
                p0 = RR["sets"][db][0].polygon2d((iL1, iL3))
                a.plot(p0[:, 0], p0[:, 1], color="0.4", lw=.7, ls=":", label=f"initial set, k = {K_A0}")
            case, nm = overlay[db]
            for i in range(N_SEEDS):
                Y = runs[case][i][0] - (runs["nominal"][i][0] if row == 1 else 0.0)
                a.plot(Y[K_A0:K_A1 + 1, iL1], Y[K_A0:K_A1 + 1, iL3], color="C3", lw=.3, alpha=.5,
                       label=(f"MC {nm}, k = 600–1200" if i == 0 else None))
                a.plot(Y[K_A1, iL1], Y[K_A1, iL3], "k.", ms=3, label=f"MC at k = {K_A1}" if i == 0 else None)
            for v in lim["LIT101"]:
                a.axvline(v, color="C3", ls="--", lw=.8)
            for v in lim["LIT301"]:
                a.axhline(v, color="C0", ls="--", lw=.8)
            a.set_title(f"{rowname}\n$\\bar{{\\delta}}$ = {db}", fontsize=8)
            a.set_xlabel(lab["LIT101"]); a.set_ylabel(lab["LIT301"])
            xs_ = np.r_[poly[:, 0], lim["LIT101"]]; ys_ = np.r_[poly[:, 1], lim["LIT301"]]
            a.set_xlim(xs_.min() - .3, xs_.max() + .3); a.set_ylim(ys_.min() - .3, ys_.max() + .3)
            a.legend(fontsize=6, loc="lower left")
    fig.text(0.5, -0.01, "Dashed lines: normalized alarm limits (LIT101 L = 500 / H = 800 mm, red; "
             "LIT301 L = 800 / H = 1000 mm, blue). Row (b) MC curves are y − y_ref of the same seed.",
             ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig5_reachability")

    fig, a = plt.subplots(figsize=(10, 6))
    pos = {"SCADA": (3, 3), "PLC1": (1, 2), "PLC2": (3, 2), "PLC3": (5, 2),
           "FIT101": (0.2, 1), "LIT101": (0.9, 0.2), "MV101": (0.2, 0.2), "P101": (1.6, 1),
           "FIT201": (2.6, 1), "MV201": (3.4, 0.2), "LIT301": (4.6, 1), "P301": (5.4, 0.2)}
    cols = {0: "0.6", 1: "C0", 2: "C1", 3: "C2"}
    phys = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "physical"]
    cyb = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "cyber"]
    sizes = [600 + 230 * affected[n] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=a, node_size=sizes, node_color=[cols[STAGE[n]] for n in G.nodes],
                           edgecolors="k", linewidths=.6)
    nx.draw_networkx_edges(G, pos, edgelist=cyb, ax=a, node_size=sizes, edge_color="0.45",
                           arrowsize=10, width=.9, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(G, pos, edgelist=phys, ax=a, node_size=sizes, edge_color="C3", style="dashed",
                           arrowsize=10, width=[.6 + 2.5 * G[u][v]["weight"] for u, v in phys],
                           connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_labels(G, pos, ax=a, font_size=8, font_family="serif",
                            labels={n: f"{n}\n({affected[n]})" for n in G.nodes})
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], marker="o", ls="", color=cols[i], mec="k", label=l)
          for i, l in [(1, "Stage 1"), (2, "Stage 2"), (3, "Stage 3"), (0, "Supervisory")]]
    hl += [Line2D([], [], color="0.45", label="cyber link (w = 1)"),
           Line2D([], [], color="C3", ls="--", label="physical link (w from B_d, K)")]
    a.legend(handles=hl, fontsize=7, loc="upper left")
    a.axis("off")
    save_fig(fig, "fig6_dependency_graph")

    # JSON outputs
    mat = dict(
        A_d=A.tolist(), B_d=B.tolist(), c_offset=c.tolist(), K=K.tolist(), A_cl=A_cl.tolist(),
        model_used=used, eig_A_d=[[v.real, v.imag, abs(v)] for v in eig_A],
        eig_A_cl=[[v.real, v.imag, abs(v)] for v in eig_cl],
        Q=Q.tolist(), R=R.tolist(), noise_cov=Sw.tolist(),
        state_tags=STATES, input_tags=INPUTS,
        norm_mean=mu.to_dict(), norm_std=sd.to_dict(),
        tag_mapping={"MV101/MV201": {"0 (transition)": 0.5, "1 (closed)": 0.0, "2 (open)": 1.0},
                     "P101/P301": {"1 (off)": 0.0, "2 (on)": 1.0},
                     "note": "mapped command is then z-scored with training mean/std"},
        Delta=DELTA, Delta_tag=delta_tag, alarm_levels_mm=ALARMS_MM, alarm_limits_normalized=lim,
        half_widths_normalized=half, windows=windows, data=info, seeds=SEEDS,
    )
    with open(os.path.join(OUT, "matrices.json"), "w") as f:
        json.dump(mat, f, indent=2)
    detail = dict(
        ident={k: dict(one=v["one"].tolist(), free=v["free"].tolist(),
                       eig_abs=np.abs(v["eig"]).tolist()) for k, v in ident.items()},
        metrics={k: {m: v.tolist() for m, v in r.items()} for k, r in met.items()},
        reach=dict(spec=dict(info={str(k): v for k, v in RA["info"].items()}, dbar_min=RA["dbar_min"],
                             zero_dbar_crosses=RA["crossed0"], crit=RA["crit"]),
                   deviation=dict(info={str(k): v for k, v in RB["info"].items()}, dbar_min=RB["dbar_min"],
                                  zero_dbar_crosses=RB["crossed0"], crit=RB["crit"]),
                   containment=contain,
                   w_half=w_half.tolist(), Z600_halfwidths=np.abs(Z600.G).sum(1).tolist()),
        graph=dict(affected=affected, betweenness=btw, ranking=rank,
                   edges={f"{u}->{v}": d["weight"] for u, v, d in G.edges(data=True)},
                   edge_weight_source={f"{u}->{v}": s for (u, v), s in edge_src.items()},
                   final_s={n: {m: round(x, 4) for m, x in s.items()} for n, s in final_s.items()}),
        fig3_seed=SEEDS[rep], observed_range_mm=obs_rng,
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                      pandas=pd.__version__, matplotlib=matplotlib.__version__, networkx=nx.__version__,
                      platform=platform.platform()),
    )
    with open(os.path.join(OUT, "run_details.json"), "w") as f:
        json.dump(detail, f, indent=2, default=float)
    print("[done] outputs written to", OUT)


if __name__ == "__main__":
    main()
