#!/usr/bin/env python3
"""Write revision_results.md from revision/outputs."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ROOT = os.path.dirname(HERE)
J = json.load(open(os.path.join(OUT, "revision_results.json")))
B = J["baseline"]
S = J["settings"]
M = J["T6"]["massbalance"]
f3 = lambda x: f"{x:.3f}"


def tab(name):
    return pd.read_csv(os.path.join(OUT, f"{name}.csv"), dtype=str, keep_default_na=False).to_markdown(
        index=False, disable_numparse=True)


def md(df):
    return df.to_markdown(index=False, disable_numparse=True)


mean = lambda l, k: float(np.mean(B["metrics"][l][k]))
others_minR = min(mean(l, "R") for l in B["metrics"] if l != "Spoofing 0.5")
db, db_bis = B["dbar"], B["dbar_reported_bisection"]
emp = J["T6"]["empirical_dbar_sim_only"]
ratio_bias, ratio_fdi = emp["bias"] / db, emp["fdi"] / db
eA = sorted(B["eigA"], key=lambda v: -v[2])
fit = lambda arr: [f3(x) for x in arr]
names = ["FIT101", "LIT101", "FIT201", "LIT301"]
boot = [v for v in J["T1"]["bootstrap"] if v["stable"]]
p5 = [v for v in J["T1"]["perturb"]["0.05"] if v["stable"]]
p10 = [v for v in J["T1"]["perturb"]["0.1"] if v["stable"]]
bd = [v["dbar"] for v in boot]
tau = J["T5"]["tau"]
sch = J["T5"]["schemes"]
top5 = lambda k: ", ".join(sch[k]["rank"][:5])
ov = lambda a, b: tau[f"{a}|{b}"]["top5_overlap"]
tv = lambda a, b: f3(tau[f"{a}|{b}"]["tau"])
dur = J["T2"]["duration"]
hz = J["T4"]["horizon"]

# Section 0
rows0 = [
    ("Samples on 1 s grid", "17,221", f"{B['n_grid']:,}", B["n_grid"] == 17221, "run_all.py:80 load_session"),
    ("Dropouts filled (ZOH)", "804", f"{B['n_filled']:,}", B["n_filled"] == 804, "run_all.py:94"),
    ("Training / validation samples", "12,055 / 5,166", f"{B['n_train']:,} / {B['n_val']:,}",
     (B["n_train"], B["n_val"]) == (12055, 5166), "run_all.py:111 split_normalize"),
    ("Spoofing peak deviation", "0.918", f3(mean("Spoofing 0.5", "peak")), f3(mean("Spoofing 0.5", "peak")) == "0.918",
     "run_all.py:171 simulate, :200 metrics"),
    ("Spoofing R", "0.857", f3(mean("Spoofing 0.5", "R")), f3(mean("Spoofing 0.5", "R")) == "0.857", "run_all.py:200 metrics"),
    ("Min R of other attacks", "≥ 0.962", f"{f3(others_minR)} (Bias 0.5)", round(others_minR, 3) >= 0.962, "run_all.py:200 metrics"),
    ("Δ", "1.352", f3(B["Delta"]), f3(B["Delta"]) == "1.352", "run_all.py:388–391"),
    ("δ̄ (actuator bound)", "0.732", f"{f3(db)} (bisection in results/: {f3(db_bis)})", f3(db) == "0.732",
     "run_all.py:441 analyze, :469 (deviation set)"),
    ("LIT101 limits", "[−2.073, 0.639]", f"[{f3(B['lim']['LIT101'][0])}, {f3(B['lim']['LIT101'][1])}]",
     f3(B['lim']['LIT101'][0]) == "-2.073" and f3(B['lim']['LIT101'][1]) == "0.639", "run_all.py:388"),
    ("LIT301 limits", "[−1.711, 0.994]", f"[{f3(B['lim']['LIT301'][0])}, {f3(B['lim']['LIT301'][1])}]",
     f3(B['lim']['LIT301'][0]) == "-1.711" and f3(B['lim']['LIT301'][1]) == "0.994", "run_all.py:388"),
    ("Top-5 components", "LIT301, LIT101, FIT101, PLC1, PLC2", ", ".join(B["graph_rank"][:5]),
     B["graph_rank"][:5] == ["LIT301", "LIT101", "FIT101", "PLC1", "PLC2"], "run_all.py:282, :312, :499"),
    ("Betweenness PLC1 / PLC2 / FIT101", "0.418 / 0.314 / 0.000",
     " / ".join(f3(B["betweenness"][n]) for n in ["PLC1", "PLC2", "FIT101"]),
     [f3(B["betweenness"][n]) for n in ["PLC1", "PLC2", "FIT101"]] == ["0.418", "0.314", "0.000"], "run_all.py:498"),
]
t0 = pd.DataFrame([{"Metric": a, "Paper value": b, "Reproduced value": c, "Match Y/N": "Y" if d else "N"}
                   for a, b, c, d, e in rows0])
where = "\n".join(f"- {a}: `{e}`" for a, b, c, d, e in rows0)

# T6 table
t6 = pd.DataFrame([
    {"Variant": "(a) Simulation only", "What it provides": "Table 2 metrics for the 6 simulated attack signals; no guarantee for other attack signals",
     "Key numbers": f"Spoofing peak {f3(mean('Spoofing 0.5','peak'))}, R {f3(mean('Spoofing 0.5','R'))}; magnitude at which the simulated signal reaches an alarm limit: bias {f3(emp['bias'])}, FDI {f3(emp['fdi'])} ({f3(ratio_bias)}× and {f3(ratio_fdi)}× the reachability δ̄ {f3(db)})"},
    {"Variant": "(b) First-principles tank model, documentation nominal values", "What it provides": "—",
     "Key numbers": "NOT COMPUTED: neither the code nor the project contains SWaT documentation with tank areas or nominal flows"},
    {"Variant": "(b′) Mass-balance tank model, parameters calibrated on the training window",
     "What it provides": "δ̄ from the same zonotope procedure on a physics-structured model (LQR redesigned, Q = I, R = 0.1·I)",
     "Key numbers": f"δ̄ {f3(M['dbar'])} vs {f3(db)}; free-run fit {', '.join(f'{n} {f3(x)}' for n, x in zip(names, M['fit_free']))} %; "
                    f"tank areas T101 {f3(M['area_T101_m2'])} m², T301 {f3(M['area_T301_m2'])} m²; flows Q1 {f3(M['Q1_m3h'])}, Q2 {f3(M['Q2_m3h'])}, Q3 {f3(M['Q3_m3h'])} m³/h"},
    {"Variant": "(c) Topology-only graph (unweighted degree, betweenness)", "What it provides": "Structural centrality; no attack propagation, no plant dynamics",
     "Key numbers": f"Top-5 degree: {top5('degree')}; betweenness: {top5('betweenness')}; overlap with (d): {ov('degree','a_baseline')}/5 and {ov('betweenness','a_baseline')}/5; τ vs (d): {tv('degree','a_baseline')} and {tv('betweenness','a_baseline')}"},
    {"Variant": "(d) Full framework", "What it provides": "Simulation metrics + reachability bound + model-weighted propagation ranking",
     "Key numbers": f"Spoofing peak {f3(mean('Spoofing 0.5','peak'))}, R {f3(mean('Spoofing 0.5','R'))}; δ̄ {f3(db)}; top-5 {', '.join(B['graph_rank'][:5])}"},
])

# Tables 4 and 5
sp = lambda vs, k: f"{f3(np.mean([v[k]['Spoofing 0.5'] for v in vs]))} ± {f3(np.std([v[k]['Spoofing 0.5'] for v in vs]))}"
worst = lambda vs: sum(v["rank_peak"][0] == "Spoofing 0.5" and v["rank_R"][0] == "Spoofing 0.5" for v in vs)
same = lambda vs, k: sum(v[k] == B[k] for v in vs)
mg = J["T2"]["magnitude"]
mpk = lambda c, m: f3(np.mean(mg[f"{c}@{m}"]["peak"]))
mR = lambda c, m: f3(np.mean(mg[f"{c}@{m}"]["R"]))
t4p = pd.DataFrame([
    {"Factor (range)": f"Estimation uncertainty: block bootstrap ({len(boot)} models)", "Spoofing worst by peak and R": f"{worst(boot)}/{len(boot)}",
     "Spoofing peak / R": f"{sp(boot,'peak')} / {sp(boot,'R')}", "δ̄ mean [min, max]": f"{f3(np.mean(bd))} [{f3(min(bd))}, {f3(max(bd))}]",
     "Full ranking unchanged (peak / R)": f"{same(boot,'rank_peak')}/{len(boot)} / {same(boot,'rank_R')}/{len(boot)}"},
    {"Factor (range)": f"Parametric ±5 % on A_d, B_d (stable {len(p5)}/20)", "Spoofing worst by peak and R": f"{worst(p5)}/{len(p5)}",
     "Spoofing peak / R": f"{sp(p5,'peak')} / {sp(p5,'R')}", "δ̄ mean [min, max]": f"{f3(np.mean([v['dbar'] for v in p5]))} [{f3(min(v['dbar'] for v in p5))}, {f3(max(v['dbar'] for v in p5))}]",
     "Full ranking unchanged (peak / R)": f"{same(p5,'rank_peak')}/{len(p5)} / {same(p5,'rank_R')}/{len(p5)}"},
    {"Factor (range)": f"Parametric ±10 % on A_d, B_d (stable {len(p10)}/20)", "Spoofing worst by peak and R": f"{worst(p10)}/{len(p10)}",
     "Spoofing peak / R": f"{sp(p10,'peak')} / {sp(p10,'R')}", "δ̄ mean [min, max]": f"{f3(np.mean([v['dbar'] for v in p10]))} [{f3(min(v['dbar'] for v in p10))}, {f3(max(v['dbar'] for v in p10))}]",
     "Full ranking unchanged (peak / R)": f"{same(p10,'rank_peak')}/{len(p10)} / {same(p10,'rank_R')}/{len(p10)}"},
    {"Factor (range)": "Attack magnitude 0.1–1.0 (FDI, bias, spoofing)", "Spoofing worst by peak and R": "10/10 magnitudes (vs FDI and bias at equal magnitude)",
     "Spoofing peak / R": f"{mpk('spoof',0.1)}–{mpk('spoof',1.0)} / {mR('spoof',0.1)}–{mR('spoof',1.0)}", "δ̄ mean [min, max]": "— (δ̄ does not depend on attack magnitude)",
     "Full ranking unchanged (peak / R)": "spoofing > bias > FDI at every magnitude (10/10)"},
    {"Factor (range)": "Attack duration 60–1200 s", "Spoofing worst by peak and R": "4/4 durations",
     "Spoofing peak / R": f"{f3(np.mean(dur['60']['Spoofing 0.5']['peak']))}–{f3(np.mean(dur['1200']['Spoofing 0.5']['peak']))} / {f3(np.mean(dur['60']['Spoofing 0.5']['R']))}–{f3(np.mean(dur['1200']['Spoofing 0.5']['R']))}",
     "δ̄ mean [min, max]": f"horizon 60 s: {f3(hz['60'])} → 1200 s: {f3(hz['1200'])}",
     "Full ranking unchanged (peak / R)": f"{sum(dur[d]['rank_peak']==B['rank_peak'] for d in dur)}/4 / {sum(dur[d]['rank_R']==B['rank_R'] for d in dur)}/4 (600 s; R also at 300 s)"},
    {"Factor (range)": "Δ × 0.75–1.25", "Spoofing worst by peak and R": "3/3",
     "Spoofing peak / R": f"{f3(mean('Spoofing 0.5','peak'))} / {f3(J['T3']['0.75']['Spoofing 0.5'])}–{f3(J['T3']['1.25']['Spoofing 0.5'])}", "δ̄ mean [min, max]": "— (δ̄ does not depend on Δ)",
     "Full ranking unchanged (peak / R)": f"R ranking: {sum(J['T3'][f+'_rank']==B['rank_R'] for f in ['0.75','1.0','1.25'])}/3"},
])
t5p = pd.DataFrame([
    {"Variant": "(a) Simulation only", "Actuator margin before alarm": f"bias signal {f3(emp['bias'])}; FDI signal {f3(emp['fdi'])} (no worst-case bound)",
     "Critical components (top-5)": "— (no graph)", "Attack severity": f"Spoofing peak {f3(mean('Spoofing 0.5','peak'))}, R {f3(mean('Spoofing 0.5','R'))}"},
    {"Variant": "(b′) Mass-balance model (calibrated) + reachability", "Actuator margin before alarm": f"δ̄ = {f3(M['dbar'])}",
     "Critical components (top-5)": "—", "Attack severity": "— (not simulated)"},
    {"Variant": "(c) Topology only", "Actuator margin before alarm": "—",
     "Critical components (top-5)": f"Betweenness: {top5('betweenness')} (overlap with (d) {ov('betweenness','a_baseline')}/5, τ = {tv('betweenness','a_baseline')})",
     "Attack severity": "—"},
    {"Variant": "(d) Full framework", "Actuator margin before alarm": f"δ̄ = {f3(db)} (worst-case over all |δ_u| ≤ δ̄)",
     "Critical components (top-5)": ", ".join(B["graph_rank"][:5]), "Attack severity": f"Spoofing peak {f3(mean('Spoofing 0.5','peak'))}, R {f3(mean('Spoofing 0.5','R'))}"},
])

# T8 (SWaT.A12, 2026)
T8 = json.load(open(os.path.join(OUT, "t8_results.json")))
I8, D8, S8 = T8["inspect"], T8["ident"], T8["sim"]
EV, FS, CX = T8["evidence_lit101"], T8["evidence_flagged_seconds"], T8["context_last4h"]
m8 = lambda l, k: float(np.mean(S8["metrics"][l][k]))
sd8 = lambda l, k: float(np.std(S8["metrics"][l][k]))
ev_n = [e for e in EV if e["window"] != "attack period"]


def rec8(l):
    r = np.array(S8["metrics"][l]["rec"], float)
    n = np.isnan(r).sum()
    return ("not recovered" if n == len(r) else f"{np.nanmean(r):.3f}" + (f" ± {np.nanstd(r):.3f}" if np.nanstd(r) > 0 else "")
            + (f" ({n}/{len(r)} not rec.)" if n else ""))


t8_inspect = pd.DataFrame([
    {"Item": "File", "Value": f"`{I8['file']}`"},
    {"Item": "Rows / columns", "Value": f"{I8['n_rows']:,} data rows (+1 header line = 28,861 lines) / {I8['n_columns']} columns"},
    {"Item": "Time span", "Value": f"{I8['start']} → {I8['end']} ({I8['n_grid']:,} s grid)"},
    {"Item": "Missing seconds / duplicate timestamps", "Value": f"{I8['missing_seconds']} / {I8['duplicate_ts']}"},
    {"Item": "Baseline tags present", "Value": "all 8; renamed with suffixes: " + ", ".join(f"{k} → {v}" for k, v in I8["tag_names"].items())},
    {"Item": "Tags missing", "Value": "none" if not I8["tags_missing"] else ", ".join(I8["tags_missing"])},
    {"Item": "'Bad Input' values in the 8 tags", "Value": f"{len(I8['bad_input_rows_in_tags'])} consecutive rows, {I8['bad_input_rows_in_tags'][0][11:]}–{I8['bad_input_rows_in_tags'][-1][11:]} (FIT201, MV201)"},
    {"Item": "Attack label column", "Value": "**NO ATTACK LABELS** (no attack/label/annotation column; only process alarm and stage-state columns such as `LSH601.Alarm`, `P1_STATE`)"},
    {"Item": "Last 4 h, LIT101 outside its normal-window range", "Value": f"{CX['n_seconds_outside']:,} s between {CX['first'][11:]} and {CX['last'][11:]} (min {f3(CX['min_mm'])} mm, max {f3(CX['max_mm'])} mm; normal-window range {f3(CX['normal_range_mm'][0])}–{f3(CX['normal_range_mm'][1])} mm) — context only, not analysed"},
])
t8_pre = pd.DataFrame([
    {"Quantity": "Normal window used", "Value": f"{D8['windows']['train']['start']} → {D8['windows']['validation']['end']} (first 4 h)"},
    {"Quantity": "Samples on 1 s grid", "Value": f"{D8['n_grid']:,} ({D8['n_bad_rows_dropped']} 'Bad Input' rows dropped: {D8['n_filled']} filled by ZOH ≤ 3 s, {D8['n_unfilled_gap']} left unfilled and excluded)"},
    {"Quantity": "Training / validation samples", "Value": f"{D8['windows']['train']['n']:,} ({D8['windows']['train']['start'][11:]}–{D8['windows']['train']['end'][11:]}) / {D8['windows']['validation']['n']:,} ({D8['windows']['validation']['start'][11:]}–{D8['windows']['validation']['end'][11:]})"},
    {"Quantity": "Observed one-step pairs used", "Value": f"{D8['n_pairs']:,}"},
])
t8_fit = pd.DataFrame([{"Output": n, "2026 one-step % (with c)": f3(D8["fit_one"][i]), "2026 free-run % (with c)": f3(D8["fit_free"][i]),
                        "2026 free-run % (no c)": f3(D8["fit_free_nooffset"][i]), "2021 free-run % (with c)": f3(B["fit_free"][i])}
                       for i, n in enumerate(names)])
fmt_e = lambda v: (f"{v[0]:.6f} {'+' if v[1] >= 0 else '−'} {abs(v[1]):.6f}j" if abs(v[1]) > 0 else f"{v[0]:.6f}") + f" (|λ| {v[2]:.6f})"
e21A = sorted(B["eigA"], key=lambda v: -v[2]); e21C = sorted(B["eigAcl"], key=lambda v: -v[2])
t8_eig = pd.DataFrame([{"#": i + 1, "A_d 2026": fmt_e(D8["eigA"][i]), "A_d 2021": fmt_e(e21A[i]),
                        "A_d − B_d K 2026": fmt_e(D8["eigAcl"][i]), "A_d − B_d K 2021": fmt_e(e21C[i])} for i in range(4)])
dO, d21 = D8["diff_own_units"], D8["diff_in_2021_units"]
t8_diff = pd.DataFrame([
    {"Matrix": "A_d", "Max abs diff (own z-units)": f3(dO["A"]["max_abs"]), "Rel. Frobenius (own z-units)": f3(dO["A"]["rel_fro"]),
     "Max abs diff (2021 z-units)": f3(d21["A"]["max_abs"]), "Rel. Frobenius (2021 z-units)": f3(d21["A"]["rel_fro"])},
    {"Matrix": "B_d", "Max abs diff (own z-units)": f3(dO["B"]["max_abs"]), "Rel. Frobenius (own z-units)": f3(dO["B"]["rel_fro"]),
     "Max abs diff (2021 z-units)": f3(d21["B"]["max_abs"]), "Rel. Frobenius (2021 z-units)": f3(d21["B"]["rel_fro"])},
    {"Matrix": "c", "Max abs diff (own z-units)": f3(dO["c"]["max_abs"]), "Rel. Frobenius (own z-units)": "—",
     "Max abs diff (2021 z-units)": f3(d21["c"]["max_abs"]), "Rel. Frobenius (2021 z-units)": "—"},
    {"Matrix": "K (LQR)", "Max abs diff (own z-units)": f3(dO["K"]["max_abs"]), "Rel. Frobenius (own z-units)": f3(dO["K"]["rel_fro"]),
     "Max abs diff (2021 z-units)": "—", "Rel. Frobenius (2021 z-units)": "—"},
])
t8_norm = pd.DataFrame([
    {"Quantity": "LIT101 mean / std (mm)", "2021": f"{f3(D8['mean21']['LIT101'])} / {f3(D8['std21']['LIT101'])}", "2026": f"{f3(D8['mean']['LIT101'])} / {f3(D8['std']['LIT101'])}"},
    {"Quantity": "LIT301 mean / std (mm)", "2021": f"{f3(D8['mean21']['LIT301'])} / {f3(D8['std21']['LIT301'])}", "2026": f"{f3(D8['mean']['LIT301'])} / {f3(D8['std']['LIT301'])}"},
    {"Quantity": "LIT101 limits (normalized)", "2021": f"[{f3(B['lim']['LIT101'][0])}, {f3(B['lim']['LIT101'][1])}]", "2026": f"[{f3(S8['lim']['LIT101'][0])}, {f3(S8['lim']['LIT101'][1])}]"},
    {"Quantity": "LIT301 limits (normalized)", "2021": f"[{f3(B['lim']['LIT301'][0])}, {f3(B['lim']['LIT301'][1])}]", "2026": f"[{f3(S8['lim']['LIT301'][0])}, {f3(S8['lim']['LIT301'][1])}]"},
    {"Quantity": "Δ", "2021": f3(B["Delta"]), "2026": f"{f3(S8['Delta'])} (LIT301)"},
])
t8_ev = pd.DataFrame([{"Start": e["start"][11:], "End": e["end"][11:], "Duration (s)": e["duration_s"], "Window": e["window"],
                       "LIT101 change (mm)": f"+{f3(e['lit101_change_mm'])}", "FIT101 max (m³/h)": f3(e["fit101_max"]),
                       "MV101 open fraction": f3(e["mv101_open_fraction"]), "P101 on fraction": f3(e["p101_on_fraction"])} for e in ev_n]
                     + [{"Start": "Total (normal window)", "End": "", "Duration (s)": sum(e["duration_s"] for e in ev_n), "Window": "",
                         "LIT101 change (mm)": f"+{f3(sum(e['lit101_change_mm'] for e in ev_n))}", "FIT101 max (m³/h)": "",
                         "MV101 open fraction": "", "P101 on fraction": ""}])
t8_flag = pd.DataFrame([
    {"Data": "2021 session (25 Jun)", "Flagged seconds": f"{FS['s2021_session']:,} / {FS['s2021_len']:,}", "Share %": f3(100 * FS['s2021_session'] / FS['s2021_len'])},
    {"Data": "2026 normal window (09:00–13:00)", "Flagged seconds": f"{FS['s2026_normal']:,} / {FS['s2026_normal_len']:,}", "Share %": f3(100 * FS['s2026_normal'] / FS['s2026_normal_len'])},
    {"Data": "2026 last 4 h (13:00–17:00)", "Flagged seconds": f"{FS['s2026_last4h']:,} / {FS['s2026_last4h_len']:,}", "Share %": f3(100 * FS['s2026_last4h'] / FS['s2026_last4h_len'])},
])
t8_att = pd.DataFrame([{"Attack": l, "Peak 2026": f"{f3(m8(l,'peak'))} ± {f3(sd8(l,'peak'))}", "RMS 2026": f"{f3(m8(l,'rms'))} ± {f3(sd8(l,'rms'))}",
                        "Recovery 2026 (s)": rec8(l), "R 2026": f"{f3(m8(l,'R'))} ± {f3(sd8(l,'R'))}",
                        "Peak 2021": f3(mean(l, "peak")), "R 2021": f3(mean(l, "R"))} for l in S8["metrics"]])
t8_sum = pd.DataFrame([
    {"Item": "Attack ranking by peak", "2021": " > ".join(B["rank_peak"]), "2026": " > ".join(S8["rank_peak"]), "Same": "Y" if S8["rank_peak_same_as_2021"] else "N"},
    {"Item": "Attack ranking by R", "2021": " > ".join(B["rank_R"]), "2026": " > ".join(S8["rank_R"]), "Same": "Y" if S8["rank_R_same_as_2021"] else "N"},
    {"Item": "δ̄ (deviation set, 600 s)", "2021": f3(S8["dbar_2021"]), "2026": f3(S8["dbar"]), "Same": "—"},
    {"Item": "Graph top-5", "2021": ", ".join(S8["graph_rank_2021"][:5]), "2026": ", ".join(S8["graph_rank"][:5]), "Same": f"{S8['top5_overlap']}/5"},
    {"Item": "Kendall τ-b, graph scores 2026 vs 2021", "2021": "—", "2026": f3(S8["tau_vs_2021"]), "Same": "—"},
])
t8_w = pd.DataFrame([{"Physical edge": e, "Weight 2021": S8["phys_weights_2021"][e].split("= ")[-1],
                      "Weight 2026": S8["phys_weights_2026"][e].split("= ")[-1]} for e in S8["phys_weights_2026"]])
t8_w["Weight 2021"] = t8_w["Weight 2021"].map(lambda s: f3(float(s)))
t8_w["Weight 2026"] = t8_w["Weight 2026"].map(lambda s: f3(float(s)))

T8_MD = f"""## T8. SWaT.A12 second-session and recorded-attack validation

**Settings:**
- Data: `{I8['file']}`. The "normal" period is the first 4 h (09:00:00–12:59:59), as stated in the task; there are no labels to confirm it.
- Preprocessing: the baseline steps unchanged (tag selection, numeric coercion, 1 s grid, ZOH ≤ 3 s, status mapping, chronological 70/30, z-score with this session's training statistics). Exception: an 8 s 'Bad Input' gap is longer than the 3 s fill limit (see Issues 11).
- Identification: `numpy.linalg.lstsq` with offset c. LQR: Q = I, R = 0.1·I.
- Simulation and metrics: the same six attacks, k = 600–1200, N = 20 seeds (1000–1019), Δ from the same alarm levels (mm) in 2026 units.
- δ̄: the same deviation-set zonotope. Graph: the same rule with 2026 B_d and K.
- Script: `revision/run_t8.py` → `revision/outputs/t8_results.json`.

### T8.1 File inspection

{md(t8_inspect)}

### T8.2 Identification on the normal 4 h

{md(t8_pre)}

Validation fit:

{md(t8_fit)}

Eigenvalues (sorted by |λ|):

{md(t8_eig)}

Element-wise difference of the 2026 model from the 2021 model. "Own z-units" compares the matrices as identified, each in its own session's normalization. "2021 z-units" first re-expresses the 2026 model in the 2021 normalization, so that both act on the same physical quantities.

{md(t8_diff)}

Normalization and alarm limits:

{md(t8_norm)}

**Validation failure for LIT101, with evidence.** In the normal window, LIT101 rises repeatedly while MV101 is closed and FIT101 = 0, often while P101 is pumping out. Detector: a rise of more than 5 mm in 60 s with FIT101 < 0.1 m³/h throughout that 60 s. A stage-1 mass balance on the measured tags cannot produce this, and the 2021 session shows almost none of it. The data cannot tell whether this is unmeasured inflow or an unlabeled sensor manipulation.

{md(t8_ev)}

{md(t8_flag)}

### T8.3 Baseline attacks, δ̄ and graph with the 2026 model

{md(t8_att)}

{md(t8_sum)}

Physical-edge weights (the cyber edges are 1 in both sessions):

{md(t8_w)}

### T8.4 Recorded-attack validation

**NO ATTACK LABELS.** Step 4 was skipped as instructed. No attack class, target, interval, recorded-vs-predicted deviation, δ̄ exceedance or alarm crossing is reported for the last 4 h. Each of these is NOT COMPUTED for the same reason.

### T8 summary

On the normal first 4 h of SWaT.A12, the unchanged pipeline gives an A_d close to the 2021 one (relative Frobenius difference {f3(d21['A']['rel_fro'])} in 2021 units) but a different B_d ({f3(d21['B']['rel_fro'])}); the 2026 A_d is open-loop unstable (max |λ| {D8['eigA'][0][2]:.6f}) and fails free-run validation for LIT101 ({f3(D8['fit_free'][1])} %), which coincides with recurring LIT101 rises while MV101 is closed and FIT101 = 0 ({FS['s2026_normal']:,} of {FS['s2026_normal_len']:,} s in the normal window vs {FS['s2021_session']:,} of {FS['s2021_len']:,} s in 2021). With this model the attack rankings by peak and by R and the graph top-5 (τ = {f3(S8['tau_vs_2021'])}) are identical to 2021, sensor spoofing remains the most damaging attack (peak {f3(m8('Spoofing 0.5','peak'))}, R {f3(m8('Spoofing 0.5','R'))}) and δ̄ = {f3(S8['dbar'])} vs {f3(S8['dbar_2021'])}, but these 2026 results are not validated for LIT101. The file contains no attack labels, so the recorded-attack validation (T8.4) could not be performed.

"""
T4_ROW = {"Factor (range)": f"Cross-session: SWaT.A12, 11 Mar 2026 (LIT101 free-run fit {f3(D8['fit_free'][1])} %, not validated)",
          "Spoofing worst by peak and R": "1/1", "Spoofing peak / R": f"{f3(m8('Spoofing 0.5','peak'))} / {f3(m8('Spoofing 0.5','R'))}",
          "δ̄ mean [min, max]": f3(S8["dbar"]),
          "Full ranking unchanged (peak / R)": f"{'Y' if S8['rank_peak_same_as_2021'] else 'N'} / {'Y' if S8['rank_R_same_as_2021'] else 'N'} (graph τ = {f3(S8['tau_vs_2021'])})"}
T8_ISSUES = f"""11. **T8 preprocessing.** 8 consecutive 'Bad Input' rows (09:22:18–09:22:25) lie in the 2026 training window. The baseline ZOH limit of 3 s filled {D8['n_filled']} of them; the remaining {D8['n_unfilled_gap']} were left unfilled and excluded from identification, as the baseline does for any unobserved pair. The baseline code would otherwise stop with "gap longer than MAX_FILL_GAP_S". The idle start-up (flows 0 until about 09:07, LIT301 starting at 655 mm) was kept, because the baseline preprocessing drops no transient.
12. **The 2026 model is not validated for LIT101.** Its free-run fit is {f3(D8['fit_free'][1])} % ({f3(D8['fit_free_nooffset'][1])} % without c), and its A_d is open-loop unstable (max |λ| {D8['eigA'][0][2]:.6f}). The closed loop is stable (max |λ| {D8['eigAcl'][0][2]:.6f}). All 2026 attack, δ̄ and graph numbers inherit this caveat.
13. **The 2026 "normal" window contains LIT101 rises with no measured inflow** (T8.2: +{f3(sum(e['lit101_change_mm'] for e in ev_n))} mm over {sum(e['duration_s'] for e in ev_n):,} s in {len(ev_n)} episodes). The same pattern occurs at a similar rate in the last 4 h ({FS['s2026_last4h']:,} flagged seconds). Whether the first 4 h are attack-free cannot be verified without labels.
14. **T8 step 4: NO ATTACK LABELS.** The recorded-attack mapping, recorded-vs-predicted comparison, δ̄ exceedance and alarm-crossing checks are NOT COMPUTED. LIT101 does leave its normal-window range in the last 4 h (see T8.1), but without labels these excursions cannot be attributed to attacks.
15. **Normalization differs between sessions** (Δ = {f3(S8['Delta'])} in 2026 vs {f3(B['Delta'])} in 2021, and different limits), so 2026 and 2021 peak and R values are each in their own session's units. They are not directly comparable in magnitude; only the rankings are. Also, A_d is close to the identity matrix, which dominates its Frobenius norm, so the small relative difference in A_d (T8.2) does not show that its small off-diagonal coupling terms agree.
16. **Row count.** The task states 28,861 rows. The file has 28,861 lines, of which {I8['n_rows']:,} are data rows plus 1 header line, at 1 s from 09:00:00 to 17:00:59.
"""

# T0 tables
t0_fit = pd.DataFrame([{"Output": n, "One-step fit % (with c)": f3(B["fit_one"][i]), "Free-run fit % (with c)": f3(B["fit_free"][i]),
                        "One-step fit % (no c)": f3(B["fit_one_nooffset"][i]), "Free-run fit % (no c)": f3(B["fit_free_nooffset"][i])}
                       for i, n in enumerate(names)])
t0_eig = pd.DataFrame([{"Eigenvalue of A_d": (f"{v[0]:.6f} {'+' if v[1] >= 0 else '−'} {abs(v[1]):.6f}j" if abs(v[1]) > 0 else f"{v[0]:.3f}"),
                        "|λ|": (f"{v[2]:.6f}" if v[2] > 0.999 else f3(v[2]))} for v in eA])

mx = lambda vs: max(v["maxeigA"] for v in vs)
t4p = pd.concat([t4p, pd.DataFrame([T4_ROW])], ignore_index=True)
rep = f"""# Revision Results

All numbers come from `revision/run_revision.py` (computation), `revision/build_tables.py` (tables) and `revision/build_report.py` (this file). The raw values are in `revision/outputs/revision_results.json`, and every table is also saved as CSV in `revision/outputs/`. The baseline pipeline `run_all.py` and `results/` were not modified.

**Settings common to all runs:**
- Session: SWaT 2021, 25 Jun 2021, 12:13:24–17:00:24.
- Horizon T = {S['horizon_s']} s at Ts = 1 s. Attack starts at k = {S['attack_start_k']}; the baseline attack ends at k = {S['attack_end_k_baseline']} (600 s).
- Monte Carlo: N = 20 seeds, {S['seeds'][0]}–{S['seeds'][-1]} (`numpy.random.default_rng(seed)`). For each seed, the initial state is N(0, {S['x0_std']}²) per state, followed by 3600×4 standard-normal noise draws scaled by the Cholesky factor of the residual covariance.
- LQR: Q = I₄, R = 0.1·I₄. FDI period P_a = {S['P_a_s']:.0f} s.
- Δ = {f3(S['Delta'])}.
- Reachability: zonotope with box reduction every 20 steps.
- Block bootstrap: random-number seed {S['bootstrap']['rng_seed']}. Parametric perturbation: random-number seed {S['perturbation']['rng_seed']}.

## 0. Baseline reproduction

{md(t0)}

Where each value is computed:

{where}

All 12 checks match. The re-run per-seed peak, RMS and R values differ from the stored `results/run_details.json` values by at most {B['max_abs_diff_vs_results_run_details']:.1e}, i.e. floating-point rounding only.

δ̄ here is computed in closed form from the linearity of the zonotope in δ̄, and gives {db:.6f}. The bisection in `run_all.py` (tolerance 1e-4) returned {db_bis:.6f}. These are equal at 3 dp. The exact zonotope check crosses at δ̄ + 1e-6 and does not cross at δ̄ − 1e-6.

## T0. Confirmed facts

1. **Files and release.**
   - Files: `SWaT/_20210625_121324/20210625_121324.csv`, `20210625_131629.csv`, `20210625_141937.csv`, `20210625_152242.csv` and `20210625_162535.csv`. Byte-identical copies are also at the top level of `SWaT/`.
   - Release label: **none appears in the files or the project** (no README, no version field). The folder and file names give only the date and time.
   - The columns include online-detector annotation columns (`A#AEGIS`, `A#AICrit`, `A#DAD`, `A#HybMon`, …) and attack-label columns (`Attack Name`, `Attack State`, …). The attack-label columns are empty in this session.
   - The attribution to iTrust and the version label cannot be confirmed from the data. They must come from the download source.
2. **Sample counts.**
   - Raw rows with valid timestamps: {B['n_raw']:,}.
   - After placing on the 1 s grid: {B['n_grid']:,}, of which {B['n_filled']:,} are dropouts filled by zero-order hold.
   - Training: {B['n_train']:,} samples ({B['windows']['train']['start']} → {B['windows']['train']['end']}).
   - Validation: {B['n_val']:,} samples ({B['windows']['validation']['start']} → {B['windows']['validation']['end']}).
   - Identification used {B['n_pairs']:,} one-step pairs, i.e. pairs in which both samples were actually observed.
3. **Preprocessing, in order** (`run_all.py:80`, `:111`):
   1. Read the 5 CSVs and drop the 1970 placeholder rows.
   2. Select the tags: states/outputs FIT101, LIT101, FIT201, LIT301; inputs MV101, P101, MV201, P301.
   3. Coerce the tags to numbers ("Bad Input" → NaN; 0 rows dropped in this session) and drop duplicate timestamps.
   4. Re-index to a 1 s grid and forward-fill gaps of at most 3 s.
   5. Map the status codes: MV 0/1/2 → 0.5/0/1 and P 1/2 → 0/1.
   6. Split chronologically 70/30.
   7. Z-score all 8 signals with the **training-window** mean and population std (ddof = 0). The values are stored in `results/matrices.json`.
   - **No filtering, smoothing or detrending** is applied.
   - **Offset c:** estimated jointly with A_d and B_d by `numpy.linalg.lstsq`, as a constant regressor column (`run_all.py:136`); c = [{', '.join(f'{x:.6f}' for x in B['c'])}]. c is used in the validation fits only. It is **not** used in the LQR design, the attack simulations or the reachability analysis, which all work in deviation coordinates around the normalized origin (the training-mean operating point).
4. **Δ** (`run_all.py:388–391`): Δ = min over the two tanks of (H − L)/2/σ.
   - LIT101: (800 − 500)/2/{110.615:.3f} = {f3(B['half_widths']['LIT101'])}.
   - LIT301: (1000 − 800)/2/{73.948:.3f} = {f3(B['half_widths']['LIT301'])}.
   - Δ = {f3(B['Delta'])} (LIT301). **Yes**, this is the half-width of the normalized LIT301 band: ({f3(B['lim']['LIT301'][1])} − ({f3(B['lim']['LIT301'][0])}))/2 = {f3((B['lim']['LIT301'][1]-B['lim']['LIT301'][0])/2)}.
5. **Attack timing** (`run_all.py:180`): `att = K_A0 <= k < K_A1`. The attack modifies the applied input (or, for spoofing, the controller's measurement) for k = 600…1199, i.e. 600 steps.
   - DoS hold uses the command computed at k = 599.
   - The state is **never reset**. Only the input path switches at onset and at the end, and the plant state carries over in both directions.
   - Recovery is measured from k = 1200.
   - The no-attack reference uses the same seed, so the initial state and noise are identical.
6. **Validation fit and eigenvalues** (`run_all.py:151`):

{md(t0_fit)}

{md(t0_eig)}

   - LIT101 free-run fit: {f3(B['fit_free'][1])} % with c, {f3(B['fit_free_nooffset'][1])} % without c.
   - |λ| is shown to 6 dp for the near-unit pair because 3 dp would round it to 1.000.
   - max |λ(A_d − B_d K)| = {f3(max(v[2] for v in B['eigAcl']))}.
7. **Reachability bound δ̄** (`run_all.py:261`, `:441`, `:469`):
   - Method: zonotope (center + generator matrix), G ← [A_cl G, B_d·δ̄·I]. Every 20 steps the zonotope is reduced to its interval hull (box).
   - Horizon: 600 steps (k = 600 → 1200).
   - Crossing test: interval hull of LIT101 or LIT301 outside its normalized alarm limits at any k in the window.
   - Bisection: bracket [0, 1], doubled until a crossing occurs, then halved until hi − lo ≤ 1e-4. The upper end is reported.
   - **The 0.732 value does not include process noise.** It is the reachable set of the attack-induced deviation x − x_ref, starting from e = 0 at the operating point; the shared noise cancels.
   - The as-specified variant includes noise as a ±3σ box per step, starting from the noise-only set at k = 600. It already crosses with no attack (δ̄ = {f3(B['dbar_spec_noise'])}).
8. **Graph** (`run_all.py:282`, `:312`, `:498`):
   - Edges: sensor → PLC, PLC → actuator, SCADA ↔ PLC and PLC1 ↔ PLC2 ↔ PLC3, all with weight 1.
   - Physical path MV101 → LIT101 → P101 → FIT201 → MV201 → LIT301 → P301:
     - actuator → sensor edges get |B_d[sensor, actuator]|/max|B_d|;
     - sensor → actuator edges get |K[actuator, sensor]|/max|K|.
   - Propagation: s[k+1] = min(1, s[k] + 0.5·Wᵀ s[k]) for 50 iterations, starting from s = 1 at the seed node.
   - "Affected" means s > 0.5, excluding the seed node.
   - Ranking: by affected count, then unweighted betweenness centrality (normalized), then name.

## T1. Model uncertainty

**Settings:**
- **Bootstrap:** {S['bootstrap']['n']} moving-block resamples of the training window. Block length {S['bootstrap']['block_s']} s; {int(np.ceil((B['n_train']-1)/300))} blocks per resample with uniform random starts. Only observed one-step pairs are used.
  - Each resample re-identifies A_d, B_d and c and **redesigns K** (Q = I, R = 0.1·I), using its own residual covariance as the noise.
- **Perturbation:** {S['perturbation']['law']}, 20 draws per level. The perturbed matrices are the plant; the **baseline K** is the controller. Draws with max|λ(A′ − B′K)| ≥ 1 are discarded.
- Δ = {f3(S['Delta'])} in all variants.

{tab('T1_metrics')}

{tab('T1_summary')}

{tab('T1_swaps')}

**Summary:**
- Under block-bootstrap re-identification, every bootstrap mean (peak and R) lies within one bootstrap std of the baseline value. Sensor spoofing is the worst attack by both peak and R in {worst(boot)}/{len(boot)} models, and the R ranking is unchanged in {same(boot,'rank_R')}/{len(boot)}.
- The peak ranking changes in {len(boot)-same(boot,'rank_peak')}/{len(boot)} models. 14 of these are a swap of FDI 0.3 and DoS zero, whose baseline peaks are close ({f3(mean('DoS zero','peak'))} and {f3(mean('FDI 0.3','peak'))}).
- **The closed loop is fragile to element-wise parameter perturbation.** With the baseline K, only {len(p5)}/20 (±5 %) and {len(p10)}/20 (±10 %) perturbed plants remain stable. Every unstable draw also had an open-loop max |λ| > 1 (15/15 and 17/17). This is consistent with the near-unit tank self-coupling being pushed above 1 while the controller has little direct authority over the tank levels; the mechanism was not isolated further.
- Among the few stable perturbed plants, spoofing remains the worst attack in every case, but the full rankings and δ̄ vary widely.

## T2. Attack magnitude and duration

**Magnitude settings:** the baseline 600 s window and 20 seeds. FDI amplitude A, bias b and spoofing offset are each swept over 0.1–1.0; all other settings are as in the baseline. For these three attack classes the deviation is identical across seeds (max std {1.1e-15:.1e}), so only means are shown. Recovery times are in s.

{tab('T2_mag')}

**Duration settings:** the attack starts at k = 600 and lasts d s. Baseline magnitudes. Recovery is measured from the end of the attack. Values are mean ± std over 20 seeds.

Peak deviation:

{tab('T2_dur_peak')}

R:

{tab('T2_dur_R')}

Recovery time (s):

{tab('T2_dur_rec')}

{tab('T2_dur_rank')}

**Summary:**
- **Magnitude.** At every magnitude, spoofing > bias > FDI by both peak and R; the ranking never changes. Peak scales linearly with magnitude, e.g. spoofing goes from {mpk('spoof',0.1)} at 0.1 to {mpk('spoof',1.0)} at 1.0. R is linear up to spoofing 0.7. From 0.8 upward the spoofing deviation exceeds Δ (peak {mpk('spoof',0.8)} > {f3(S['Delta'])}), so the min(1, ·) saturates and R flattens ({mR('spoof',0.8)}, {mR('spoof',0.9)}, {mR('spoof',1.0)}).
- **Duration.** Spoofing is the worst attack at every duration.
  - The lower part of the ranking depends on duration. At 60 s, FDI 0.3 ranks above both DoS cases. At 1200 s, DoS hold rises to second place, with peak {f3(np.mean(dur['1200']['DoS hold']['peak']))} and R {f3(np.mean(dur['1200']['DoS hold']['R']))}.
  - The full peak ranking matches the 600 s ranking only at 600 s. The R ranking also matches at 300 s.
  - The FDI peak does not depend on duration ({f3(np.mean(dur['60']['FDI 0.8']['peak']))} at every duration), while R falls as the duration grows.

## T3. Resilience Δ sensitivity

Settings: the baseline runs (600 s window, 20 seeds), with R recomputed for Δ × 0.75, 1.00 and 1.25 ({f3(0.75*S['Delta'])}, {f3(S['Delta'])} and {f3(1.25*S['Delta'])}). Values are means over the 20 seeds.

{tab('T3')}

**Summary:** the ranking by R is identical for all three Δ values. Spoofing's R ranges from {f3(J['T3']['0.75']['Spoofing 0.5'])} at 0.75Δ to {f3(J['T3']['1.25']['Spoofing 0.5'])} at 1.25Δ. No baseline deviation exceeds 0.75Δ (largest peak {f3(max(max(B['metrics'][l]['peak']) for l in B['metrics']))} < {f3(0.75*S['Delta'])}), so R = 1 − mean(e)/Δ for every attack, and scaling Δ cannot reorder the attacks.

## T4. δ̄ robustness

Settings: deviation-set zonotope (no noise), box reduction every 20 steps, horizon starting at k = 600, closed-form crossing (equal to bisection, see §0).

{tab('T4')}

**Summary:**
- Under bootstrap re-identification, δ̄ = {f3(np.mean(bd))} (range {f3(min(bd))}–{f3(max(bd))}), close to the baseline {f3(db)}.
- δ̄ falls monotonically with the attack horizon: {f3(hz['60'])} at 60 s, {f3(hz['300'])} at 300 s, {f3(hz['600'])} at 600 s and {f3(hz['1200'])} at 1200 s. Longer horizons let the slow closed-loop modes accumulate more actuator deviation.
- In the few stable ±5 % and ±10 % plants, δ̄ ranges from {f3(min(v['dbar'] for v in p5+p10))} to {f3(max(v['dbar'] for v in p5+p10))}. These are not a representative sample (see T1).

## T5. Graph weighting

Scheme definitions (topology identical in all schemes):
- (a) Baseline weights.
- (b) Weight 1 on every edge.
- (c) Every physical edge weighted |B_d[sensor, actuator]|/max|B_d| for its sensor–actuator pair, in either direction.
- (d) Every physical edge weighted |K[actuator, sensor]|/max|K|, in either direction.
- Cyber edges are 1 in all schemes.

The criticality ranking (affected count in parentheses) uses the baseline rule. The centrality baselines use the unweighted graph: total degree, normalized betweenness and PageRank (α = 0.85), with the score in parentheses. Ties are broken alphabetically.

{tab('T5_rank')}

Physical-edge weights per scheme:

{tab('T5_weights')}

Kendall τ-b on the node scores (affected count + 0.5·betweenness for (a)–(d), i.e. lexicographic), with the top-5 overlap in parentheses:

{tab('T5_tau')}

**Summary:**
- **(a) and (c) produce identical rankings (τ = {tv('a_baseline','c_Bonly')}), and so do (b) and (d) (τ = {tv('b_unit','d_Konly')}).** The criticality ranking depends only on whether the two actuator → tank edges (MV101 → LIT101, MV201 → LIT301) are weak, as with B_d: weights {f3(sch['a_baseline']['weights']['MV101->LIT101'])} and {f3(sch['a_baseline']['weights']['MV201->LIT301'])}.
- When those edges are strong (unit or K weights), the PLCs and SCADA also reach the tank sensors. The top-5 then becomes {top5('b_unit')} (τ vs (a) = {tv('a_baseline','b_unit')}, overlap {ov('a_baseline','b_unit')}/5).
- PLC1 and PLC2 are in the top-5 under every scheme and every centrality measure. The tank sensors LIT301 and LIT101 rank top only under B_d-based weights.
- Plain centralities agree with each other (τ {tv('betweenness','pagerank')}–{tv('degree','betweenness')} between degree, betweenness and PageRank) but only weakly with the baseline scheme (τ {tv('a_baseline','pagerank')}–{tv('a_baseline','degree')}, overlap 2/5).

## T6. Ablation and baselines

{md(t6)}

Settings:
- (a): the deviation per unit magnitude of the bias and FDI signals over k = 600…1200 (seed-independent), scaled to the first alarm limit it reaches.
- (b′):
  - LIT[k+1] = LIT[k] + κ·(Q_in − Q_out), with κ fitted by least squares on observed training pairs (κ1 = {M['kappa1']:.3f}, κ3 = {M['kappa3']:.3f} mm per (m³/h·s)).
  - Q_out of T301 is measured with FIT301, which is not one of the model states.
  - Flows: FIT101 = Q1·MV101, FIT201 = Q2·P101, T301 outflow = Q3·P301, each Q being the mean flow when the actuator is on/open in training. MV201 has no effect in this model.
  - The model is converted to the same z-units; it has integrator eigenvalues {', '.join(f3(x) for x in M['eigA'])}. One-step fit: {', '.join(f'{n} {f3(x)}' for n, x in zip(names, M['fit_one']))} %.
  - With the LQR redesigned, max |λ(A_cl)| = {f3(M['maxeigAcl'])}.
  - Horizon dependence of δ̄ (b′): 60 s {f3(M['dbar_h']['60'])}, 300 s {f3(M['dbar_h']['300'])}, 600 s {f3(M['dbar_h']['600'])}, 1200 s {f3(M['dbar_h']['1200'])}.

**Summary:**
- **(a) Simulation alone** ranks the attacks, but for the two simulated actuator signals it implies alarm-reaching magnitudes {f3(ratio_bias)}× (bias) and {f3(ratio_fdi)}× (FDI) larger than the worst-case reachability bound. It therefore overstates the safe actuator margin.
- **(b′) The calibrated mass-balance model** gives a smaller δ̄ ({f3(M['dbar'])} vs {f3(db)}). This is consistent with its pure-integrator tank dynamics and direct flow-to-level coupling; the cause was not tested further. Its flow-channel fits are lower (FIT201 free-run {f3(M['fit_free'][2])} %).
- **(c) Topology-only centrality** shares {ov('betweenness','a_baseline')}/5 top components with the full framework and misses the tank sensors.

{T8_MD}## 7. Proposed Table 4 (Sensitivity of main findings)

{md(t4p)}

## 8. Proposed Table 5 (Ablation and baseline comparison)

{md(t5p)}

## 9. One-sentence conclusion for the paper

Across block-bootstrap re-identification ({len(boot)} models), attack magnitudes from 0.1 to 1.0, durations from 60 to 1200 s and Δ scaled by 0.75–1.25, sensor spoofing remained the most damaging attack by both peak deviation and R in every stable case, the reachability bound stayed at δ̄ = {f3(np.mean(bd))} ({f3(min(bd))}–{f3(max(bd))}) under re-identification but fell from {f3(hz['60'])} to {f3(hz['1200'])} as the attack horizon grew from 60 to 1200 s, and the ablation showed that simulation alone overstates the alarm-reaching actuator magnitude by {f3(ratio_bias)}–{f3(ratio_fdi)}×, a calibrated mass-balance model gives a tighter δ̄ of {f3(M['dbar'])}, and topology-only centrality recovers only {ov('betweenness','a_baseline')} of the 5 critical components, although ±5 % and ±10 % element-wise perturbations of A_d and B_d destabilized {20-len(p5)}/20 and {20-len(p10)}/20 of the baseline closed loops.

## 10. Issues / discrepancies

1. **δ̄ = 0.732 is not the noise-inclusive bound.** It comes from the supplementary attack-induced-deviation zonotope, which has no process noise. The as-specified noise-bounded set gives δ̄ = {f3(B['dbar_spec_noise'])}, because the 3σ noise accumulation alone crosses the alarm limits. The paper should state which set 0.732 refers to.
2. **Rounding inconsistency in the earlier `results/results_summary.md`.** DoS-hold R is {mean('DoS hold','R'):.5f}, i.e. **{f3(mean('DoS hold','R'))}** at 3 dp. The earlier summary (§8, §12) shows 0.973 because it rounded the 4 dp value 0.9725 a second time. If the paper uses 0.973, it should be corrected to {f3(mean('DoS hold','R'))}. All other baseline values agree at 3 dp.
3. **Dataset release label and the iTrust attribution: NOT CONFIRMED.** There is no version label in the files or the project (T0.1).
4. **T1 perturbation results rest on very few stable draws** ({len(p5)}/20 and {len(p10)}/20). The stable subset is selected by stability, so its statistics are biased and should not be read as a distribution.
   - Assumption: "±p %" is interpreted as an element-wise uniform factor in [1 − p, 1 + p].
   - Bootstrap and perturbation are treated differently: the bootstrap redesigns K, while the perturbation keeps the baseline K, which is the robustness question.
   - {sum(v['maxeigA'] > 1 for v in J['T1']['bootstrap'])}/50 bootstrap models have open-loop max |λ(A_d)| slightly above 1 ({mx([v for v in J['T1']['bootstrap'] if v['maxeigA'] > 1]):.6f}). Their redesigned closed loops are stable.
5. **Zero spread across seeds is structural.** For FDI, bias and spoofing, std over seeds = 0 (the model is linear and the reference shares the noise realization). Only the DoS cases vary across seeds.
6. **T2:** "bias injection at A" is implemented as bias b ∈ {{0.1, …, 1.0}} applied to all four inputs. FDI recovery times (10–63 s) depend on the sinusoid phase at the attack end (sin = 0 at k = 660, 900, 1200 and 1800).
7. **T5:**
   - Schemes (c) and (d) needed a rule for edges whose direction has no native B_d or K entry. The pair's coupling magnitude was used.
   - In the degree ranking, ranks 5–9 are tied at degree 3 (FIT201, LIT101, LIT301, MV201, P101), so degree-top-5 membership is arbitrary at rank 5. With another tie-break, the overlap with (a) could be 3/5 instead of 2/5. The betweenness top-5 overlap (2/5) is not affected by ties.
8. **T6(b) with documentation-nominal tank areas: NOT COMPUTED** (no SWaT documentation in the project). The substitute (b′) calibrates the tank areas from the same training data, so it is not independent of the data.
9. **Description of the controller.** The paper calls the closed loop "LQR output feedback". With C = I it is full-state feedback u = −K x.
10. **Eigenvalues.** The near-unit eigenvalues of A_d are reported to 6 dp (0.999921), because 3 dp would display 1.000 and wrongly suggest a pure integrator.
{T8_ISSUES}"""
with open(os.path.join(ROOT, "revision_results.md"), "w") as f:
    f.write(rep)
print("written", os.path.join(ROOT, "revision_results.md"), len(rep), "chars")
