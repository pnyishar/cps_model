#!/usr/bin/env python3
"""
T8: SWaT.A12 session (11 March 2026), compared with the 2021 model.

    .venv/bin/python revision/run_t8.py      # -> revision/outputs/t8_results.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import run_all as RA  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
A12_DIR = "SWaT.A12_OTDataset_Mar_26"
A12_FILE = "11-Mar-2026_0900_1700.csv"
TAGMAP = {"FIT101": "FIT101.Pv", "LIT101": "LIT101.Pv", "FIT201": "FIT201.Pv", "LIT301": "LIT301.Pv",
          "MV101": "MV101.Status", "P101": "P101.Status", "MV201": "MV201.Status", "P301": "P301.Status"}
NORMAL_END = pd.Timestamp("2026-03-11 13:00:00")     # first 4 h = 09:00:00 .. 12:59:59
T, K0, K1 = RA.T_SIM, RA.K_A0, RA.K_A1
SEEDS, N = RA.SEEDS, len(RA.SEEDS)
CASES = [("dos_zero", None, "DoS zero"), ("dos_hold", None, "DoS hold"), ("fdi", 0.3, "FDI 0.3"),
         ("fdi", 0.8, "FDI 0.8"), ("bias", 0.5, "Bias 0.5"), ("spoof", 0.5, "Spoofing 0.5")]
LABELS = [c[2] for c in CASES]
R_ = {}

# Step 1: file inspection
path = os.path.join(ROOT, A12_DIR, A12_FILE)
raw0 = pd.read_csv(path, low_memory=False)
raw0["t"] = pd.to_datetime(raw0["t_stamp"], format="%d-%b-%Y %H:%M:%S")
grid_all = pd.date_range(raw0["t"].min(), raw0["t"].max(), freq="1s")
label_like = [c for c in raw0.columns if any(k in c.lower() for k in ["attack", "label", "normal", "annot"])]
bad_rows = raw0[TAGMAP.values()].astype(str).eq("Bad Input").any(axis=1)
R_["inspect"] = dict(
    file=f"{A12_DIR}/{A12_FILE}", n_rows=len(raw0), n_columns=raw0.shape[1] - 1,   # minus added "t"
    start=str(raw0["t"].min()), end=str(raw0["t"].max()), n_grid=len(grid_all),
    missing_seconds=int(len(grid_all.difference(raw0["t"]))), duplicate_ts=int(raw0["t"].duplicated().sum()),
    tag_names=TAGMAP, tags_missing=[k for k, v in TAGMAP.items() if v not in raw0.columns],
    label_like_columns=label_like, attack_labels="NO ATTACK LABELS" if not label_like else label_like,
    bad_input_rows_in_tags=[str(x) for x in raw0.loc[bad_rows, "t"]],
)

# Step 2: preprocessing and identification
raw = raw0[raw0["t"] < NORMAL_END].copy()
for k, v in TAGMAP.items():
    raw[k] = pd.to_numeric(raw[v], errors="coerce")
n_bad = int(raw[RA.STATES + RA.INPUTS].isna().any(axis=1).sum())
raw = raw.dropna(subset=RA.STATES + RA.INPUTS).drop_duplicates("t").sort_values("t").set_index("t")[RA.STATES + RA.INPUTS]
grid = pd.date_range(raw.index[0], raw.index[-1], freq="1s")
df = raw.reindex(grid)
df["observed"] = df[RA.STATES[0]].notna()
df = df.ffill(limit=RA.MAX_FILL_GAP_S)
n_unfilled = int(df[RA.STATES].isna().any(axis=1).sum())        # gaps > 3 s stay NaN
for c in RA.INPUTS:
    mp = RA.MV_MAP if c.startswith("MV") else RA.P_MAP
    bad = set(df[c].dropna().unique()) - set(mp)
    if bad:
        raise RuntimeError(f"unmapped codes in {c}: {bad}")
    df[c] = df[c].map(mp)
z, n_tr, mu, sd, windows = RA.split_normalize(df)
A, B, c, resid, n_pairs = RA.identify(z, n_tr, True)
one, free, _ = RA.validate(z, n_tr, A, B, c)
A0, B0, c0, _, _ = RA.identify(z, n_tr, False)
one0, free0, _ = RA.validate(z, n_tr, A0, B0, c0)
if np.isnan(free).any():
    raise RuntimeError("NaN in validation window")
Sw = np.cov(resid.T)
Lw = np.linalg.cholesky(Sw)
Q, Rw = RA.Q_W * np.eye(4), RA.R_W * np.eye(4)
K, _ = RA.dlqr(A, B, Q, Rw)
Acl = A - B @ K
eigA, eigCl = np.linalg.eigvals(A), np.linalg.eigvals(Acl)

# 2021 model
df21, _ = RA.load_session()
z21, n21, mu21, sd21, _ = RA.split_normalize(df21)
A21, B21, c21, res21, _ = RA.identify(z21, n21, True)
K21, _ = RA.dlqr(A21, B21, Q, Rw)
Acl21 = A21 - B21 @ K21
m = json.load(open(os.path.join(ROOT, "results", "matrices.json")))
assert np.allclose(A21, m["A_d"], atol=1e-12) and np.allclose(B21, m["B_d"], atol=1e-12)


def diff(X, Y):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    return dict(max_abs=float(np.abs(X - Y).max()), rel_fro=float(np.linalg.norm(X - Y) / np.linalg.norm(Y)))


# 2026 model in the 2021 normalization
Dx26, Du26 = np.diag(sd[RA.STATES].values), np.diag(sd[RA.INPUTS].values)
Dx21, Du21 = np.diag(sd21[RA.STATES].values), np.diag(sd21[RA.INPUTS].values)
mx26, mu26 = mu[RA.STATES].values, mu[RA.INPUTS].values
mx21, mu21u = mu21[RA.STATES].values, mu21[RA.INPUTS].values
iDx21 = np.linalg.inv(Dx21)
A_in21 = iDx21 @ Dx26 @ A @ np.linalg.inv(Dx26) @ Dx21
B_in21 = iDx21 @ Dx26 @ B @ np.linalg.inv(Du26) @ Du21
c_in21 = iDx21 @ (Dx26 @ (A @ np.linalg.inv(Dx26) @ (mx21 - mx26) + B @ np.linalg.inv(Du26) @ (mu21u - mu26) + c)
                  + mx26 - mx21)
R_["ident"] = dict(
    n_raw_normal=len(raw), n_bad_rows_dropped=n_bad, n_grid=len(df), n_filled=int((~df["observed"]).sum()) - n_unfilled,
    n_unfilled_gap=n_unfilled, n_pairs=n_pairs, windows=windows,
    mean=mu.to_dict(), std=sd.to_dict(), mean21=mu21.to_dict(), std21=sd21.to_dict(),
    fit_one=one.tolist(), fit_free=free.tolist(), fit_one_nooffset=one0.tolist(), fit_free_nooffset=free0.tolist(),
    eigA=[[v.real, v.imag, abs(v)] for v in sorted(eigA, key=lambda v: -abs(v))],
    eigAcl=[[v.real, v.imag, abs(v)] for v in sorted(eigCl, key=lambda v: -abs(v))],
    A=A.tolist(), B=B.tolist(), c=c.tolist(), K=K.tolist(),
    diff_own_units=dict(A=diff(A, A21), B=diff(B, B21), c=dict(max_abs=float(np.abs(c - c21).max())), K=diff(K, K21)),
    diff_in_2021_units=dict(A=diff(A_in21, A21), B=diff(B_in21, B21), c=dict(max_abs=float(np.abs(c_in21 - c21).max()))),
    residual_std=np.sqrt(np.diag(Sw)).tolist(),
)
print("[T8.2] fit one", np.round(one, 3), "free", np.round(free, 3), "| |eig A|", np.round(np.abs(eigA), 6),
      "| |eig Acl|", np.round(np.abs(eigCl), 6))
print("[T8.2] diff", json.dumps(R_["ident"]["diff_own_units"]), json.dumps(R_["ident"]["diff_in_2021_units"]))
if np.abs(eigCl).max() >= 1:
    raise RuntimeError("2026 closed loop not Schur stable")

# Step 3: attacks, dbar, graph
lim = {t: tuple((v - mu[t]) / sd[t] for v in RA.ALARMS_MM[t]) for t in RA.ALARMS_MM}
half = {t: (RA.ALARMS_MM[t][1] - RA.ALARMS_MM[t][0]) / 2 / sd[t] for t in RA.ALARMS_MM}
DELTA = min(half.values())
iL1, iL3 = RA.STATES.index("LIT101"), RA.STATES.index("LIT301")

X0, Zn = [], []
for s in SEEDS:
    rng = np.random.default_rng(s)
    X0.append(rng.normal(0.0, RA.X0_STD, 4))
    Zn.append(rng.standard_normal((T, 4)))
X0, Zn = np.array(X0), np.array(Zn)


def sim_batch(A_, B_, K_, Lw_, case, mag):
    """run_all.simulate batched over seeds."""
    W = Zn @ Lw_.T
    x = X0.copy()
    Y = np.empty((N, T, 4))
    u_hold = None
    for k in range(T):
        att = case is not None and K0 <= k < K1
        yc = x + mag if (att and case == "spoof") else x
        u = -yc @ K_.T
        if k == K0 - 1:
            u_hold = u.copy()
        if att:
            if case == "dos_zero":
                u = np.zeros_like(u)
            elif case == "dos_hold":
                u = u_hold
            elif case == "fdi":
                u = u + mag * np.sin(2 * np.pi * k / RA.P_A)
            elif case == "bias":
                u = u + mag
        Y[:, k] = x
        x = x @ A_.T + u @ B_.T + W[:, k]
    return Y


def metrics_batch(Y, Yref, delta):
    e = np.linalg.norm(Y - Yref, axis=2)
    peak = e.max(1)
    rms = np.sqrt(np.mean(e[:, K0:K1] ** 2, 1))
    rec = np.empty(N)
    for i in range(N):
        post = e[i, K1:]
        above = np.nonzero(post >= 0.05 * peak[i])[0]
        rec[i] = 0.0 if len(above) == 0 else (np.nan if above[-1] == len(post) - 1 else above[-1] + 1.0)
    return dict(peak=peak, rms=rms, rec=rec, R=1.0 - np.mean(np.minimum(1.0, e / delta), 1))


def dbar_min(Acl_, B_, lim_, horizon=K1 - K0):
    G = np.zeros((4, 0))
    r = [np.zeros(4)]
    for i in range(horizon):
        G = np.hstack([Acl_ @ G, B_])
        r.append(np.abs(G).sum(1))
        if (i + 1) % RA.REDUCE_EVERY == 0:
            G = np.diag(np.abs(G).sum(1))
    r = np.array(r)
    return min(min(abs(lim_[t][0]), abs(lim_[t][1])) / r[:, i].max() for t, i in [("LIT101", iL1), ("LIT301", iL3)])


def crosses(Acl_, B_, lim_, dbar):
    Zs = RA.reach(Acl_, B_, np.zeros(4), RA.Zonotope(np.zeros(4), np.zeros((4, 0))), dbar)
    return any((Z_.interval()[0][i] < lim_[t][0]) or (Z_.interval()[1][i] > lim_[t][1])
               for Z_ in Zs for t, i in [("LIT101", iL1), ("LIT301", iL3)])


def criticality(G):
    btw = nx.betweenness_centrality(G, normalized=True)
    aff = {n: sum(1 for mm, v in RA.propagate(G, n).items() if mm != n and v > RA.AFFECT_THR) for n in G.nodes}
    return sorted(G.nodes, key=lambda n: (-aff[n], -btw[n], n)), {n: aff[n] + 0.5 * btw[n] for n in G.nodes}, aff, btw


Yref = sim_batch(A, B, K, Lw, None, None)
met = {lab: metrics_batch(sim_batch(A, B, K, Lw, case, mg), Yref, DELTA) for case, mg, lab in CASES}
rank_peak = sorted(LABELS, key=lambda l: -met[l]["peak"].mean())
rank_R = sorted(LABELS, key=lambda l: met[l]["R"].mean())
db = dbar_min(Acl, B, lim)
db_chk = (crosses(Acl, B, lim, db + 1e-6), crosses(Acl, B, lim, db - 1e-6))
G26, w26 = RA.build_graph(Acl, B, K)
G21, w21 = RA.build_graph(Acl21, B21, K21)
rk26, sc26, aff26, bt26 = criticality(G26)
rk21, sc21, aff21, bt21 = criticality(G21)
nodes = list(G21.nodes)
tau, p = kendalltau([sc21[n] for n in nodes], [sc26[n] for n in nodes])
base = json.load(open(os.path.join(OUT, "revision_results.json")))["baseline"]
R_["sim"] = dict(
    Delta=DELTA, half_widths=half, lim=lim,
    metrics={l: {k: met[l][k].tolist() for k in ["peak", "rms", "rec", "R"]} for l in LABELS},
    rank_peak=rank_peak, rank_R=rank_R,
    rank_peak_same_as_2021=rank_peak == base["rank_peak"], rank_R_same_as_2021=rank_R == base["rank_R"],
    dbar=db, dbar_check=dict(crosses_above=db_chk[0], crosses_below=db_chk[1]), dbar_2021=base["dbar"],
    graph_rank=rk26, graph_rank_2021=rk21, affected=aff26, affected_2021=aff21, betweenness=bt26,
    tau_vs_2021=float(tau), tau_p=float(p), top5_overlap=len(set(rk26[:5]) & set(rk21[:5])),
    phys_weights_2026={f"{u}->{v}": s for (u, v), s in w26.items() if "(1)" not in s},
    phys_weights_2021={f"{u}->{v}": s for (u, v), s in w21.items() if "(1)" not in s},
)
print("[T8.3] Δ", round(DELTA, 3), "δ̄", round(db, 3), db_chk, "rank peak", rank_peak, "rank R", rank_R)
for l in LABELS:
    print(f"   {l:13s} peak {met[l]['peak'].mean():.3f}±{met[l]['peak'].std():.3f} R {met[l]['R'].mean():.3f}")
print("[T8.3] graph", rk26[:5], "τ", round(tau, 3), "overlap", R_["sim"]["top5_overlap"])

# LIT101 rising with no measured inflow
L = pd.to_numeric(raw0.set_index("t")["LIT101.Pv"], errors="coerce")
F = pd.to_numeric(raw0.set_index("t")["FIT101.Pv"], errors="coerce")
P101s = pd.to_numeric(raw0.set_index("t")["P101.Status"], errors="coerce")
MV101s = pd.to_numeric(raw0.set_index("t")["MV101.Status"], errors="coerce")
rise = L.diff(60)                                   # mm over the last 60 s
flag = (rise > 5) & (F.rolling(60).max() < 0.1)     # >5 mm/min rise while FIT101 < 0.1 m3/h for the whole minute
eps, cur = [], None
for t_, f_ in flag.items():
    if f_ and cur is None:
        cur = [t_ - pd.Timedelta(seconds=60), t_]
    elif f_:
        cur[1] = t_
    elif cur is not None and (t_ - cur[1]).total_seconds() > 60:
        eps.append(cur); cur = None
if cur is not None:
    eps.append(cur)
R_["evidence_lit101"] = [dict(start=str(a), end=str(b), duration_s=int((b - a).total_seconds()),
                              lit101_change_mm=float(L[b] - L[a]), fit101_max=float(F[a:b].max()),
                              p101_on_fraction=float((P101s[a:b] == 2).mean()),
                              mv101_open_fraction=float((MV101s[a:b] == 2).mean()),
                              window=("training" if b < pd.Timestamp(windows["validation"]["start"]) else
                                      "validation" if b < NORMAL_END else "attack period"))
                         for a, b in eps if F[a:b].max() < 0.1]   # drop valve-opening edges
# same detector on the 2021 session (comparison)
r21 = pd.concat([pd.read_csv(os.path.join(ROOT, f), low_memory=False) for f in RA.SESSION_FILES])
r21["t"] = pd.to_datetime(r21["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
r21 = r21[r21["t"].dt.year > 2000].drop_duplicates("t").set_index("t").sort_index()
L21 = pd.to_numeric(r21["LIT101"], errors="coerce").asfreq("1s").ffill()
F21 = pd.to_numeric(r21["FIT101"], errors="coerce").asfreq("1s").ffill()
R_["evidence_flagged_seconds"] = dict(
    s2021_session=int(((L21.diff(60) > 5) & (F21.rolling(60).max() < 0.1)).sum()), s2021_len=len(L21),
    s2026_normal=int(flag[:NORMAL_END - pd.Timedelta(seconds=1)].sum()), s2026_normal_len=int((L.index < NORMAL_END).sum()),
    s2026_last4h=int(flag[NORMAL_END:].sum()), s2026_last4h_len=int((L.index >= NORMAL_END).sum()))
# last 4 h: LIT101 outside its normal-window range
Ln = L[:NORMAL_END - pd.Timedelta(seconds=1)]
out = L[NORMAL_END:][(L[NORMAL_END:] < Ln.min()) | (L[NORMAL_END:] > Ln.max())]
R_["context_last4h"] = dict(normal_range_mm=[float(Ln.min()), float(Ln.max())], n_seconds_outside=int(len(out)),
                            first=str(out.index.min()) if len(out) else None, last=str(out.index.max()) if len(out) else None,
                            min_mm=float(out.min()) if len(out) else None, max_mm=float(out.max()) if len(out) else None)
print("[T8] LIT101 episodes:", json.dumps(R_["evidence_lit101"], indent=0)[:1500])
print("[T8] last-4h context:", R_["context_last4h"])

with open(os.path.join(OUT, "t8_results.json"), "w") as f:
    json.dump(R_, f, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print("[done]", os.path.join(OUT, "t8_results.json"))
