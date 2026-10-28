#!/usr/bin/env python3
"""Build CSV and markdown tables from revision/outputs/revision_results.json."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
J = json.load(open(os.path.join(OUT, "revision_results.json")))
B = J["baseline"]
LABELS = list(B["metrics"].keys())
f3 = lambda x: f"{x:.3f}"
pm = lambda a: f"{np.mean(a):.3f} ± {np.std(a):.3f}"


def rec_str(r):
    r = np.array(r, float)
    n = np.isnan(r).sum()
    if n == len(r):
        return "not recovered"
    s = f"{np.nanmean(r):.3f}" + (f" ± {np.nanstd(r):.3f}" if np.nanstd(r) > 0 else "")
    return s + (f" ({n}/{len(r)} not rec.)" if n else "")


T = {}

# T1
def var_stats(vs):
    st = [v for v in vs if v["stable"]]
    return st


boot = var_stats(J["T1"]["bootstrap"])
p5 = var_stats(J["T1"]["perturb"]["0.05"])
p10 = var_stats(J["T1"]["perturb"]["0.1"])
sets = [("Bootstrap", boot, len(J["T1"]["bootstrap"])), ("±5 %", p5, len(J["T1"]["perturb"]["0.05"])),
        ("±10 %", p10, len(J["T1"]["perturb"]["0.1"]))]
rows = []
for l in LABELS:
    row = {"Attack": l, "Peak baseline": f3(np.mean(B["metrics"][l]["peak"]))}
    for nm, vs, _ in sets:
        row[f"Peak {nm}"] = pm([v["peak"][l] for v in vs])
    row["R baseline"] = f3(np.mean(B["metrics"][l]["R"]))
    for nm, vs, _ in sets:
        row[f"R {nm}"] = pm([v["R"][l] for v in vs])
    rows.append(row)
T["T1_metrics"] = pd.DataFrame(rows)
rows = []
for nm, vs, ntot in sets:
    rows.append({"Variant set": nm, "Stable / drawn": f"{len(vs)}/{ntot}",
                 "max|λ(A)| range": f"{min(v['maxeigA'] for v in vs):.6f}–{max(v['maxeigA'] for v in vs):.6f}",
                 "δ̄ mean ± std": pm([v["dbar"] for v in vs]),
                 "Same peak ranking": f"{sum(v['rank_peak'] == B['rank_peak'] for v in vs)}/{len(vs)}",
                 "Same R ranking": f"{sum(v['rank_R'] == B['rank_R'] for v in vs)}/{len(vs)}",
                 "Spoofing worst (peak & R)": f"{sum(v['rank_peak'][0] == 'Spoofing 0.5' and v['rank_R'][0] == 'Spoofing 0.5' for v in vs)}/{len(vs)}"})
allp = J["T1"]["perturb"]
for p in ["0.05", "0.1"]:
    vs = allp[p]
    rows.append({"Variant set": f"±{float(p):.0%} (all draws)", "Stable / drawn": f"{sum(v['stable'] for v in vs)}/{len(vs)}",
                 "max|λ(A)| range": f"{min(v['maxeigA'] for v in vs):.6f}–{max(v['maxeigA'] for v in vs):.6f}",
                 "δ̄ mean ± std": "—", "Same peak ranking": "—", "Same R ranking": "—",
                 "Spoofing worst (peak & R)": f"unstable draws with open-loop max|λ|>1: "
                 f"{sum((not v['stable']) and v['maxeigA'] > 1 for v in vs)}/{sum(not v['stable'] for v in vs)}"})
T["T1_summary"] = pd.DataFrame(rows)
# which pairs swap
swaps = {}
for nm, vs, _ in sets:
    for key in ["rank_peak", "rank_R"]:
        cnt = {}
        for v in vs:
            if v[key] != B[key]:
                diff = tuple(a for a, b in zip(v[key], B[key]) if a != b)
                cnt[diff] = cnt.get(diff, 0) + 1
        swaps[f"{nm} {key}"] = {" / ".join(k): n for k, n in cnt.items()}
T["T1_swaps"] = pd.DataFrame([{"Variant set / ranking": k.replace("rank_peak", "by peak").replace("rank_R", "by R"),
    "Attacks whose positions changed": a, "Count": n} for k, d in swaps.items() for a, n in d.items()])
n_boot_open_unstable = sum(v["maxeigA"] > 1 for v in J["T1"]["bootstrap"])

# T2 magnitude
mg = J["T2"]["magnitude"]
rows, maxstd = [], 0
for m in [round(0.1 * i, 1) for i in range(1, 11)]:
    row = {"Magnitude": f"{m:.1f}"}
    for case, nm in [("fdi", "FDI"), ("bias", "Bias"), ("spoof", "Spoof")]:
        r = mg[f"{case}@{m}"]
        maxstd = max(maxstd, np.std(r["peak"]), np.std(r["R"]))
        row[f"{nm} peak"] = f3(np.mean(r["peak"]))
        row[f"{nm} R"] = f3(np.mean(r["R"]))
        row[f"{nm} rec (s)"] = rec_str(r["rec"])
    rows.append(row)
T["T2_mag"] = pd.DataFrame(rows)
order_peak = {m: sorted(["fdi", "bias", "spoof"], key=lambda c: -np.mean(mg[f"{c}@{m}"]["peak"])) for m in [round(0.1 * i, 1) for i in range(1, 11)]}
order_R = {m: sorted(["fdi", "bias", "spoof"], key=lambda c: np.mean(mg[f"{c}@{m}"]["R"])) for m in [round(0.1 * i, 1) for i in range(1, 11)]}

# T2 duration
du = J["T2"]["duration"]
D = ["60", "300", "600", "1200"]
for key, nm in [("peak", "peak"), ("R", "R"), ("rec", "rec")]:
    rows = []
    for l in LABELS:
        row = {"Attack": l}
        for d in D:
            a = du[d][l][key]
            row[f"{d} s"] = rec_str(a) if key == "rec" else pm(a)
        rows.append(row)
    T[f"T2_dur_{nm}"] = pd.DataFrame(rows)
T["T2_dur_rank"] = pd.DataFrame([{"Duration": f"{d} s", "Ranking by peak (worst→least)": " > ".join(du[d]["rank_peak"]),
                                  "Ranking by R (worst→least)": " > ".join(du[d]["rank_R"]),
                                  "Same as 600 s peak/R": f"{'Y' if du[d]['rank_peak'] == B['rank_peak'] else 'N'}/"
                                                          f"{'Y' if du[d]['rank_R'] == B['rank_R'] else 'N'}"} for d in D])

# T3
t3 = J["T3"]
rows = [{"Attack": l, **{f"R (Δ×{f} = {float(f) * J['settings']['Delta']:.3f})": f3(t3[f][l]) for f in ["0.75", "1.0", "1.25"]},
         **{f"Rank (Δ×{f})": t3[f + "_rank"].index(l) + 1 for f in ["0.75", "1.0", "1.25"]}} for l in LABELS]
T["T3"] = pd.DataFrame(rows)
peaks_max = max(max(B["metrics"][l]["peak"]) for l in LABELS)

# T4
rows = [{"Setting": "Baseline (600 s horizon)", "n": 1, "δ̄ mean": f3(B["dbar"]), "δ̄ min": f3(B["dbar"]), "δ̄ max": f3(B["dbar"])}]
for nm, vs, _ in sets:
    d = [v["dbar"] for v in vs]
    rows.append({"Setting": f"{nm} models (stable)", "n": len(d), "δ̄ mean": f3(np.mean(d)), "δ̄ min": f3(min(d)), "δ̄ max": f3(max(d))})
for h in D:
    v = J["T4"]["horizon"][h]
    rows.append({"Setting": f"Baseline model, horizon {h} s", "n": 1, "δ̄ mean": f3(v), "δ̄ min": f3(v), "δ̄ max": f3(v)})
for nm, vs, _ in sets:
    for h in D:
        d = [v["dbar_h"][h] for v in vs]
        rows.append({"Setting": f"{nm} models, horizon {h} s", "n": len(d), "δ̄ mean": f3(np.mean(d)), "δ̄ min": f3(min(d)), "δ̄ max": f3(max(d))})
T["T4"] = pd.DataFrame(rows)

# T5
S = J["T5"]["schemes"]
names = {"a_baseline": "(a) baseline", "b_unit": "(b) unit", "c_Bonly": "(c) B_d only", "d_Konly": "(d) K only",
         "degree": "Degree", "betweenness": "Betweenness", "pagerank": "PageRank"}
rows = []
for i in range(12):
    row = {"Rank": i + 1}
    for k, nm in names.items():
        n = S[k]["rank"][i]
        row[nm] = f"{n} ({S[k]['affected'][n]})" if "affected" in S[k] else f"{n} ({S[k]['score'][n]:.3f})"
    rows.append(row)
T["T5_rank"] = pd.DataFrame(rows)
rows = []
for a, an in names.items():
    row = {"Scheme": an}
    for b, bn in names.items():
        t = J["T5"]["tau"][f"{a}|{b}"]
        row[bn] = f"{t['tau']:.3f} ({t['top5_overlap']}/5)"
    rows.append(row)
T["T5_tau"] = pd.DataFrame(rows)
T["T5_weights"] = pd.DataFrame([{"Edge": e, **{names[k]: f3(S[k]["weights"][e]) for k in ["a_baseline", "b_unit", "c_Bonly", "d_Konly"]}}
                                for e in S["a_baseline"]["weights"]])

with open(os.path.join(OUT, "tables.md"), "w") as f:
    for k, v in T.items():
        f.write(f"### {k}\n\n" + (v.to_markdown(index=False, disable_numparse=True) if isinstance(v, pd.DataFrame) else json.dumps(v, indent=1)) + "\n\n")
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(OUT, f"{k}.csv"), index=False)
    f.write(f"n_boot_open_loop_unstable={n_boot_open_unstable}\nmax std over seeds in T2 magnitude (peak,R)={maxstd:.3e}\n"
            f"max peak over all baseline runs={peaks_max:.3f}\norder_peak={order_peak}\norder_R={order_R}\n")
print(open(os.path.join(OUT, "tables.md")).read())
