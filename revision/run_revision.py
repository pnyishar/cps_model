#!/usr/bin/env python3
"""
Revision analyses T0-T6.

    .venv/bin/python revision/run_revision.py   # -> revision/outputs/revision_results.json
"""
import json
import os
import sys
import time

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

# Settings
T = RA.T_SIM                      # 3600
K0 = RA.K_A0                      # 600
SEEDS = RA.SEEDS                  # 1000..1019
N = len(SEEDS)
BOOT_N, BOOT_BLOCK, BOOT_SEED = 50, 300, 20261018
PERT_LEVELS, PERT_DRAWS, PERT_SEED = [0.05, 0.10], 20, 20261019
MAGS = [round(0.1 * i, 1) for i in range(1, 11)]
DURATIONS = [60, 300, 600, 1200]
DELTA_SCALES = [0.75, 1.00, 1.25]
CASES = [("dos_zero", None, "DoS zero"), ("dos_hold", None, "DoS hold"),
         ("fdi", 0.3, "FDI 0.3"), ("fdi", 0.8, "FDI 0.8"),
         ("bias", 0.5, "Bias 0.5"), ("spoof", 0.5, "Spoofing 0.5")]
LABELS = [c[2] for c in CASES]


# Baseline model
df, info = RA.load_session()
z, n_tr, mu, sd, windows = RA.split_normalize(df)
A, B, c, resid, n_pairs = RA.identify(z, n_tr, True)
one, free, _ = RA.validate(z, n_tr, A, B, c)
A0, B0, c0, _, _ = RA.identify(z, n_tr, False)
one0, free0, _ = RA.validate(z, n_tr, A0, B0, c0)
Sw = np.cov(resid.T)
Lw = np.linalg.cholesky(Sw)
Q, R = RA.Q_W * np.eye(4), RA.R_W * np.eye(4)
K, _ = RA.dlqr(A, B, Q, R)
Acl = A - B @ K
lim = {t: tuple((v - mu[t]) / sd[t] for v in RA.ALARMS_MM[t]) for t in RA.ALARMS_MM}
half = {t: (RA.ALARMS_MM[t][1] - RA.ALARMS_MM[t][0]) / 2 / sd[t] for t in RA.ALARMS_MM}
DELTA = min(half.values())
iL1, iL3 = RA.STATES.index("LIT101"), RA.STATES.index("LIT301")

# per-seed draws, same order as run_all.simulate
X0, Zn = [], []
for s in SEEDS:
    rng = np.random.default_rng(s)
    X0.append(rng.normal(0.0, RA.X0_STD, 4))
    Zn.append(rng.standard_normal((T, 4)))
X0, Zn = np.array(X0), np.array(Zn)          # (N,4), (N,T,4)


# Simulation
def sim_batch(A_, B_, K_, Lw_, case, mag, k0=K0, k1=RA.K_A1):
    """run_all.simulate batched over seeds; attack active for k0 <= k < k1."""
    W = Zn @ Lw_.T
    x = X0.copy()
    Y = np.empty((N, T, 4))
    u_hold = None
    for k in range(T):
        att = case is not None and k0 <= k < k1
        yc = x + mag if (att and case == "spoof") else x
        u = -yc @ K_.T
        if k == k0 - 1:
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


def metrics_batch(Y, Yref, delta, k0=K0, k1=RA.K_A1):
    """Per-seed metrics as in run_all.metrics (recovery NaN = not recovered)."""
    e = np.linalg.norm(Y - Yref, axis=2)
    peak = e.max(1)
    rms = np.sqrt(np.mean(e[:, k0:k1] ** 2, 1))
    rec = np.empty(N)
    for i in range(N):
        post = e[i, k1:]
        above = np.nonzero(post >= 0.05 * peak[i])[0]
        rec[i] = 0.0 if len(above) == 0 else (np.nan if above[-1] == len(post) - 1 else above[-1] + 1.0)
    Rr = 1.0 - np.mean(np.minimum(1.0, e / delta), 1)
    return dict(peak=peak, rms=rms, rec=rec, R=Rr, e=e)


def run_cases(A_, B_, K_, Lw_, k1=RA.K_A1, cases=CASES, keep_e=False):
    Yref = sim_batch(A_, B_, K_, Lw_, None, None)
    out = {}
    for (case, mag, lab) in cases:
        m = metrics_batch(sim_batch(A_, B_, K_, Lw_, case, mag, K0, k1), Yref, DELTA, K0, k1)
        if not keep_e:
            m.pop("e")
        out[lab] = m
    return out


def ranking(res, key):
    """Attack labels ordered from most to least damaging (peak: descending; R: ascending)."""
    vals = {l: np.mean(res[l][key]) for l in res}
    return sorted(vals, key=lambda l: (-vals[l] if key == "peak" else vals[l]))


# Reachability
def dev_radius(Acl_, B_, nsteps, reduce_every=RA.REDUCE_EVERY):
    """Interval-hull radius of the deviation zonotope for dbar = 1 (no noise, zero initial set)."""
    G = np.zeros((4, 0))
    r = [np.zeros(4)]
    for i in range(nsteps):
        G = np.hstack([Acl_ @ G, B_])
        r.append(np.abs(G).sum(1))
        if (i + 1) % reduce_every == 0:
            G = np.diag(np.abs(G).sum(1))
    return np.array(r)


def dbar_min(Acl_, B_, horizon=RA.K_A1 - K0):
    """Smallest dbar at which the deviation set reaches an alarm limit within the horizon."""
    r = dev_radius(Acl_, B_, horizon)
    best = np.inf
    for t, i in [("LIT101", iL1), ("LIT301", iL3)]:
        rr = r[:, i].max()
        if rr > 0:
            best = min(best, min(abs(lim[t][0]), abs(lim[t][1])) / rr)
    return best


def crosses(Acl_, B_, dbar, horizon=RA.K_A1 - K0):
    Z0 = RA.Zonotope(np.zeros(4), np.zeros((4, 0)))
    Zs = RA.reach(Acl_, B_, np.zeros(4), Z0, dbar, k0=K0, k1=K0 + horizon)
    for t, i in [("LIT101", iL1), ("LIT301", iL3)]:
        for Z_ in Zs:
            lo, hi = Z_.interval()
            if lo[i] < lim[t][0] or hi[i] > lim[t][1]:
                return True
    return False


# Graph
def build_graph_scheme(scheme, K_=K, B_=B, Acl_=Acl):
    if scheme == "a_baseline":
        return RA.build_graph(Acl_, B_, K_)[0]
    G, _ = RA.build_graph(Acl_, B_, K_)
    si, ui = {s: i for i, s in enumerate(RA.STATES)}, {u: i for i, u in enumerate(RA.INPUTS)}
    for a, b, d in G.edges(data=True):
        if d["kind"] != "physical":
            continue
        s_, u_ = (b, a) if a in ui else (a, b)          # (sensor, actuator) of the pair
        if scheme == "b_unit":
            d["weight"] = 1.0
        elif scheme == "c_Bonly":
            d["weight"] = abs(B_[si[s_], ui[u_]]) / np.abs(B_).max()
        elif scheme == "d_Konly":
            d["weight"] = abs(K_[ui[u_], si[s_]]) / np.abs(K_).max()
    return G


def criticality(G):
    btw = nx.betweenness_centrality(G, normalized=True)
    aff = {}
    for n in G.nodes:
        s = RA.propagate(G, n)
        aff[n] = sum(1 for m, v in s.items() if m != n and v > RA.AFFECT_THR)
    score = {n: aff[n] + 0.5 * btw[n] for n in G.nodes}     # lexicographic (affected, betweenness)
    rank = sorted(G.nodes, key=lambda n: (-aff[n], -btw[n], n))
    return rank, score, aff, btw


t_start = time.time()
RES = {"settings": dict(
    seeds=SEEDS, horizon_s=T, attack_start_k=K0, attack_end_k_baseline=RA.K_A1, P_a_s=RA.P_A,
    x0_std=RA.X0_STD, Q="I4", R="0.1*I4", Delta=DELTA, alarm_limits_norm=lim,
    bootstrap=dict(n=BOOT_N, block_s=BOOT_BLOCK, rng_seed=BOOT_SEED),
    perturbation=dict(levels=PERT_LEVELS, draws=PERT_DRAWS, rng_seed=PERT_SEED,
                      law="A' = A*(1+p*U), B' = B*(1+p*U), U ~ Uniform(-1,1) element-wise"),
)}

# Baseline reproduction
base = run_cases(A, B, K, Lw, keep_e=True)
prev = json.load(open(os.path.join(ROOT, "results", "run_details.json")))
maxdiff = max(np.nanmax(np.abs(np.array(prev["metrics"][key][m], float) - base[lab][m]))
              for (key, lab) in zip(["dos_zero", "dos_hold", "fdi_03", "fdi_08", "bias", "spoof"], LABELS)
              for m in ["peak", "rms", "R"])
db_base = dbar_min(Acl, B)
db_check = (crosses(Acl, B, db_base + 1e-6), crosses(Acl, B, db_base - 1e-6))
G_base = build_graph_scheme("a_baseline")
rank_base, score_base, aff_base, btw_base = criticality(G_base)
RES["baseline"] = dict(
    n_raw=info["n_raw"], n_grid=info["n_grid"], n_filled=info["n_filled"],
    n_train=windows["train"]["n"], n_val=windows["validation"]["n"], windows=windows, n_pairs=n_pairs,
    fit_one=one.tolist(), fit_free=free.tolist(), fit_one_nooffset=one0.tolist(), fit_free_nooffset=free0.tolist(),
    eigA=[[v.real, v.imag, abs(v)] for v in np.linalg.eigvals(A)],
    eigAcl=[[v.real, v.imag, abs(v)] for v in np.linalg.eigvals(Acl)],
    Delta=DELTA, half_widths=half, lim=lim, c=c.tolist(),
    metrics={l: {m: base[l][m].tolist() for m in ["peak", "rms", "rec", "R"]} for l in LABELS},
    max_abs_diff_vs_results_run_details=float(maxdiff),
    dbar=db_base, dbar_reported_bisection=prev["reach"]["deviation"]["dbar_min"],
    dbar_closed_form_check=dict(crosses_above=db_check[0], crosses_below=db_check[1]),
    dbar_spec_noise=prev["reach"]["spec"]["dbar_min"],
    graph_rank=rank_base, affected=aff_base, betweenness=btw_base,
    rank_peak=ranking(base, "peak"), rank_R=ranking(base, "R"),
)
print(f"[0] max |diff| vs results/run_details.json = {maxdiff:.2e}; δ̄ = {db_base:.6f} check {db_check}")
print("[0] rank peak", RES["baseline"]["rank_peak"], "\n[0] rank R", RES["baseline"]["rank_R"])

# T1: model uncertainty
def summarize_variant(res):
    return dict(peak={l: float(np.mean(res[l]["peak"])) for l in res},
                R={l: float(np.mean(res[l]["R"])) for l in res},
                rank_peak=ranking(res, "peak"), rank_R=ranking(res, "R"))

X, U, ok = z[RA.STATES].values, z[RA.INPUTS].values, z["observed"].values
valid = np.nonzero(ok[:n_tr - 1] & ok[1:n_tr])[0]            # pair start indices k (k+1 < n_tr)
valid_set = np.zeros(n_tr, bool); valid_set[valid] = True
rng = np.random.default_rng(BOOT_SEED)
n_blocks = int(np.ceil((n_tr - 1) / BOOT_BLOCK))
boot = []
for b in range(BOOT_N):
    starts = rng.integers(0, n_tr - 1 - BOOT_BLOCK + 1, n_blocks)
    idx = np.concatenate([s + np.arange(BOOT_BLOCK) for s in starts])
    idx = idx[valid_set[idx]]
    Phi = np.hstack([X[idx], U[idx], np.ones((len(idx), 1))])
    th, *_ = np.linalg.lstsq(Phi, X[idx + 1], rcond=None)
    Ab, Bb = th[:4].T, th[4:8].T
    Lb = np.linalg.cholesky(np.cov((X[idx + 1] - Phi @ th).T))
    Kb, _ = RA.dlqr(Ab, Bb, Q, R)
    Aclb = Ab - Bb @ Kb
    stable = np.abs(np.linalg.eigvals(Aclb)).max() < 1
    v = dict(maxeigA=float(np.abs(np.linalg.eigvals(Ab)).max()),
             maxeigAcl=float(np.abs(np.linalg.eigvals(Aclb)).max()), stable=bool(stable))
    if stable:
        v.update(summarize_variant(run_cases(Ab, Bb, Kb, Lb)), dbar=float(dbar_min(Aclb, Bb)),
                 dbar_h={d: float(dbar_min(Aclb, Bb, d)) for d in DURATIONS})
    boot.append(v)
print(f"[T1] bootstrap done ({sum(v['stable'] for v in boot)}/{BOOT_N} stable)  {time.time()-t_start:.0f}s")

rng = np.random.default_rng(PERT_SEED)
pert = {}
for p in PERT_LEVELS:
    pert[p] = []
    for d in range(PERT_DRAWS):
        Ap = A * (1 + p * rng.uniform(-1, 1, A.shape))
        Bp = B * (1 + p * rng.uniform(-1, 1, B.shape))
        Aclp = Ap - Bp @ K                                     # plant perturbed, baseline controller K
        me = float(np.abs(np.linalg.eigvals(Aclp)).max())
        v = dict(maxeigA=float(np.abs(np.linalg.eigvals(Ap)).max()), maxeigAcl=me, stable=me < 1)
        if me < 1:
            v.update(summarize_variant(run_cases(Ap, Bp, K, Lw)), dbar=float(dbar_min(Aclp, Bp)),
                     dbar_h={d_: float(dbar_min(Aclp, Bp, d_)) for d_ in DURATIONS})
        pert[p].append(v)
    print(f"[T1] ±{p:.0%}: {sum(v['stable'] for v in pert[p])}/{PERT_DRAWS} stable  {time.time()-t_start:.0f}s")
RES["T1"] = dict(bootstrap=boot, perturb={str(p): v for p, v in pert.items()})

# T2: attack magnitude and duration
yref = sim_batch(A, B, K, Lw, None, None)
mag = {}
for case in ["fdi", "bias", "spoof"]:
    for m_ in MAGS:
        r_ = metrics_batch(sim_batch(A, B, K, Lw, case, m_), yref, DELTA)
        mag[f"{case}@{m_}"] = {k: r_[k].tolist() for k in ["peak", "R", "rec", "rms"]}
dur = {}
for d in DURATIONS:
    res = run_cases(A, B, K, Lw, k1=K0 + d)
    dur[d] = {l: {k: res[l][k].tolist() for k in ["peak", "R", "rec", "rms"]} for l in res}
    dur[d]["rank_peak"], dur[d]["rank_R"] = ranking(res, "peak"), ranking(res, "R")
RES["T2"] = dict(magnitude=mag, duration={str(d): v for d, v in dur.items()})
print(f"[T2] done  {time.time()-t_start:.0f}s")

# T3: sensitivity of R to Delta
t3 = {}
for f in DELTA_SCALES:
    t3[str(f)] = {l: float(np.mean(1 - np.mean(np.minimum(1, base[l]["e"] / (f * DELTA)), 1))) for l in LABELS}
    t3[str(f) + "_rank"] = sorted(LABELS, key=lambda l: t3[str(f)][l])
RES["T3"] = t3

# T4: robustness of dbar
RES["T4"] = dict(horizon={d: float(dbar_min(Acl, B, d)) for d in DURATIONS})

# T5: graph weighting schemes
schemes = {}
for sc in ["a_baseline", "b_unit", "c_Bonly", "d_Konly"]:
    G_ = build_graph_scheme(sc)
    rk, scr, aff, bt = criticality(G_)
    schemes[sc] = dict(rank=rk, score=scr, affected=aff,
                       weights={f"{u}->{v}": d["weight"] for u, v, d in G_.edges(data=True) if d["kind"] == "physical"})
Gu = build_graph_scheme("b_unit")
for name, cen in [("degree", dict(Gu.degree())), ("betweenness", nx.betweenness_centrality(Gu, normalized=True)),
                  ("pagerank", nx.pagerank(Gu, weight=None))]:
    schemes[name] = dict(rank=sorted(Gu.nodes, key=lambda n: (-cen[n], n)), score=cen)
nodes = list(G_base.nodes)
tau = {}
for a_ in schemes:
    for b_ in schemes:
        t_, p_ = kendalltau([schemes[a_]["score"][n] for n in nodes], [schemes[b_]["score"][n] for n in nodes])
        tau[f"{a_}|{b_}"] = dict(tau=float(t_), p=float(p_),
                                 top5_overlap=len(set(schemes[a_]["rank"][:5]) & set(schemes[b_]["rank"][:5])))
RES["T5"] = dict(schemes=schemes, tau=tau)
print("[T5]", {k: v["rank"][:5] for k, v in schemes.items()})

# T6: ablation
# (a) simulation only: magnitude at which each simulated attack signal reaches an alarm limit
emp = {}
for case in ["fdi", "bias"]:
    e_ = (sim_batch(A, B, K, Lw, case, 1.0) - yref)[:, K0:RA.K_A1 + 1]
    best = np.inf
    for t, i in [("LIT101", iL1), ("LIT301", iL3)]:
        hi_, lo_ = e_[..., i].max(), e_[..., i].min()
        if hi_ > 0:
            best = min(best, lim[t][1] / hi_)
        if lo_ < 0:
            best = min(best, lim[t][0] / lo_)
    emp[case] = float(best)

# (b') mass-balance model calibrated on the training window
raw = pd.concat([pd.read_csv(os.path.join(ROOT, f), low_memory=False) for f in RA.SESSION_FILES])
raw["t"] = pd.to_datetime(raw["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
raw = raw[raw["t"].dt.year > 2000].drop_duplicates("t").set_index("t")
fit301 = pd.to_numeric(raw["FIT301"], errors="coerce").reindex(df.index).ffill(limit=RA.MAX_FILL_GAP_S).values
P = df[RA.STATES + RA.INPUTS].values.copy()
P = pd.DataFrame(P, columns=RA.STATES + RA.INPUTS, index=df.index)
tr = slice(0, n_tr)
okp = valid
Q1 = P["FIT101"].values[tr][P["MV101"].values[tr] == 1].mean()
Q2 = P["FIT201"].values[tr][P["P101"].values[tr] == 1].mean()
Q3 = fit301[tr][P["P301"].values[tr] == 1].mean()
dL1 = P["LIT101"].values[okp + 1] - P["LIT101"].values[okp]
dL3 = P["LIT301"].values[okp + 1] - P["LIT301"].values[okp]
q1 = P["FIT101"].values[okp] - P["FIT201"].values[okp]
q3 = P["FIT201"].values[okp] - fit301[okp]
k1_ = float(q1 @ dL1 / (q1 @ q1))
k3_ = float(q3 @ dL3 / (q3 @ q3))
Ap_ = np.array([[0, 0, 0, 0], [k1_, 1, -k1_, 0], [0, 0, 0, 0], [0, 0, k3_, 1]], float)
Bp_ = np.array([[Q1, 0, 0, 0], [0, 0, 0, 0], [0, Q2, 0, 0], [0, 0, 0, -k3_ * Q3]], float)
Dx, Du = np.diag(sd[RA.STATES].values), np.diag(sd[RA.INPUTS].values)
mx, mu_u = mu[RA.STATES].values, mu[RA.INPUTS].values
Az = np.linalg.inv(Dx) @ Ap_ @ Dx
Bz = np.linalg.inv(Dx) @ Bp_ @ Du
cz = np.linalg.inv(Dx) @ (Ap_ @ mx + Bp_ @ mu_u - mx)
one_p, free_p, _ = RA.validate(z, n_tr, Az, Bz, cz)
Kp, _ = RA.dlqr(Az, Bz, Q, R)
Aclp_ = Az - Bz @ Kp
db_phys = float(dbar_min(Aclp_, Bz))
RES["T6"] = dict(
    empirical_dbar_sim_only=emp,
    massbalance=dict(Q1_m3h=Q1, Q2_m3h=Q2, Q3_m3h=Q3, kappa1=k1_, kappa3=k3_,
                     area_T101_m2=1000 / 3600 / k1_, area_T301_m2=1000 / 3600 / k3_,
                     fit_one=one_p.tolist(), fit_free=free_p.tolist(),
                     eigA=np.abs(np.linalg.eigvals(Az)).tolist(),
                     maxeigAcl=float(np.abs(np.linalg.eigvals(Aclp_)).max()), dbar=db_phys,
                     dbar_h={d: float(dbar_min(Aclp_, Bz, d)) for d in DURATIONS}),
)
print("[T6]", json.dumps(RES["T6"], default=float)[:600])

with open(os.path.join(OUT, "revision_results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
print(f"[done] {time.time()-t_start:.0f}s -> {OUT}")
