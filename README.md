# Hybrid Mathematical Modeling of Cyber-Physical System Vulnerabilities: A Water Treatment Case Study

This repository contains the code and the computed outputs behind the paper. Every number, table and figure in the paper can be regenerated with the commands below.

## Repository layout

| Path | Content |
|---|---|
| `run_all.py` | Main pipeline. It covers data preprocessing, system identification (A_d, B_d, c), the LQR controller, six attack simulations, metrics (peak, RMS, recovery, R), zonotope reachability (δ̄) and the dependency graph. Writes to `results/`. |
| `results/` | Outputs of `run_all.py`: Figs. 3–6 (PNG at 300 dpi and PDF), Tables 1–3 (CSV), `matrices.json` and `run_details.json`. |
| `revision/` | Revision analyses (T0–T8): model uncertainty, attack magnitude and duration, Δ sensitivity, δ̄ robustness, graph weighting, ablation, and the second-session check on SWaT.A12. |
| `revision/outputs/` | Outputs of the revision scripts (JSON, CSV tables, `workflow.png`). |
| `requirements.txt` | Exact package versions used. |

## Data (not included)

The SWaT datasets are distributed on request by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design (<https://itrust.sutd.edu.sg/>), under iTrust's terms of use. They are therefore not included in this repository. Request access through iTrust's datasets page, then place the files as shown below. The relative paths are hard-coded in `run_all.py` (`SESSION_FILES`) and `revision/run_t8.py`.

**Why the 25 June 2021 session.** The project's SWaT copy holds the 2021 collection: 10 unlabeled sessions, with no SWaT A1 "Normal" file. The 25 June session shows the most regular cycling and the fewest online-detector alarms. On 12, 14 and 21 July, P101 and MV201 are stuck at 0 while flow is moving, and other days show unexplained level plateaus.

```
SWaT/_20210625_121324/20210625_121324.csv     # SWaT 2021 collection, session of 25 June 2021
SWaT/_20210625_121324/20210625_131629.csv
SWaT/_20210625_121324/20210625_141937.csv
SWaT/_20210625_121324/20210625_152242.csv
SWaT/_20210625_121324/20210625_162535.csv
SWaT.A12_OTDataset_Mar_26/11-Mar-2026_0900_1700.csv   # SWaT.A12, 11 March 2026 (revision T8 only)
```

To check that your files are identical to the ones used in the paper, run `shasum -a 256 <file>` (macOS) or `sha256sum <file>` (Linux) and compare with this list:

```
fc887e4048067bb1bc712574e35f0fb2c2485b16933052ca4213563919af5987  SWaT/_20210625_121324/20210625_121324.csv
4d598b64df70188a94c6b864a9918f1b204b39ed84069c668a9abb403d1a2922  SWaT/_20210625_121324/20210625_131629.csv
eb158bee5b9e17fe45b1550be9ffdc3c5594dcd6a9b57f035323e42ffdbf1d60  SWaT/_20210625_121324/20210625_141937.csv
828adcd960ac34291ec0ffa3a8f6493b0b6049a61f21d538b3020125cefbae99  SWaT/_20210625_121324/20210625_152242.csv
f55d9fb783e2bdf7088b9025b4cc8114215b67878406f13062e0a25e56ba8b57  SWaT/_20210625_121324/20210625_162535.csv
d1dd05034b393a6b201f39e86ac3b933000551d26680c5d91c5ff8db0559cf72  SWaT.A12_OTDataset_Mar_26/11-Mar-2026_0900_1700.csv
```

## Reproducing the results

Python 3.12.10 was used. Set up the environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then run the steps in order:

1. `.venv/bin/python run_all.py` — main results (Figs. 3–6, Tables 1–3) → `results/`. Takes about 1–2 min.
2. `.venv/bin/python revision/run_revision.py` — sensitivity and ablation analyses (T1–T6).
3. `.venv/bin/python revision/run_t8.py` — second session, SWaT.A12 (T8). Needs the A12 file.
4. `.venv/bin/python revision/build_tables.py` — builds the revision tables.
5. `.venv/bin/python revision/build_report.py` — writes the full revision report, `revision_results.md`. It is generated locally and not committed; it includes the list of known issues and assumptions (§10).
6. `.venv/bin/python revision/workflow_figure.py` — writes `revision/outputs/workflow.png`.

All randomness is seeded, so a rerun reproduces every number exactly:
- Monte Carlo seeds 1000–1019 (`numpy.random.default_rng`);
- block-bootstrap seed 20261018;
- parametric-perturbation seed 20261019.

With the same data and package versions, the committed tables and JSON files in `results/` and `revision/outputs/` regenerate identically; the PDF figures differ only in their embedded creation date. The committed outputs therefore serve as the reference to compare against.

Notes:
- `run_all.py --upto 3` stops after identification and the LQR checks.
- `run_all.py --no-offset` uses the identification equation (Eq. 8) without the constant offset term c.
- The revision scripts import functions from `run_all.py`; they never call its `main()` and never write to `results/`.

## Where to find each result

| Paper item | File |
|---|---|
| Fig. 3–6 | `results/fig3_…` to `results/fig6_…` (`.png` and `.pdf`) |
| Tables 1–3 | `results/table1_identification.csv`, `results/table2_attack_impact.csv`, `results/table3_graph.csv` |
| Identified matrices, LQR gain, normalization, Δ | `results/matrices.json` |
| Per-seed metrics, reachability data, graph weights | `results/run_details.json` |
| Sensitivity (Table 4), ablation (Table 5), second session | `revision/outputs/*.csv` and `*.json`; full report via step 5 (`revision_results.md`) |
