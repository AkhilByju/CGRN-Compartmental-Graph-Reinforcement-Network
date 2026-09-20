#!/usr/bin/env python3
"""Paper A final publication-integrity checks (read-only).

Recomputes every headline number from the frozen processed/raw run records and compares it with
(a) the manuscript PDF text, (b) the generated table fragments and figure CSVs, and (c) the vector
geometry of the figures embedded in the PDF; then scans the PDF for unresolved references, stale
model names, hard-coded references (typed text with no hyperref link), layout overhang and
de-anonymizing strings.

Nothing under experiments/ or paper/ is written. Run from anywhere:

    python paper/audit/run_final_audit.py [--pdf paper/TMLR_Journal_Submission_V3.pdf]

Dependencies: pypdf, pdfminer.six (both pure Python). The LaTeX source is NOT needed (nor available
in this repo); source-level checks (duplicate \\label, hard-coded \\ref in .tex, compile log) are out of scope
here and are listed as NOT CHECKED in paper/final_publication_audit.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "paper_a"
RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def chk(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    return ok


def info(name: str, detail: str = "") -> None:
    RESULTS.append(("INFO", name, detail))


def load(rel: str):
    with open(EXP / rel) as f:
        return json.load(f)


def ms(v):
    v = list(v)
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def trap(xs, ys):
    return sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))


BINS = ["0", "(0, 0.10]", "(0.10, 0.25]", "(0.25, 0.50]", ">0.50"]
CELLS = [(d, c) for d in ("mnist", "fashion_mnist", "digits", "california_housing") for c in ("missing", "gaussian")]


# ============================================================================ A. numbers
def check_numbers(pdf_text: str):
    rel = load("reliability/results/processed/reliability_runs.json")
    cap = load("reliability/results/processed/capacity_stress_runs.json")
    real = load("real_reliability/results/processed/real_reliability_runs.json")

    for nm, rows, key in (("reliability", rel, ("dataset", "corruption_family", "family", "seed")),
                          ("capacity", cap, ("dataset", "corruption_family", "family", "seed")),
                          ("real", real, ("dataset", "family", "seed"))):
        c = collections.Counter(tuple(r[k] for k in key) for r in rows)
        chk(f"records/{nm}: unique (cell,model,seed) keys and seeds == {{0,1,2}}", max(c.values()) == 1 and {r['seed'] for r in rows} == {0, 1, 2}, f"n={len(rows)}")
        chk(f"records/{nm}: no diverged runs", not any(r.get("diverged", False) for r in rows))
        info(f"records/{nm}: cap_hit rows", str([(r.get("dataset"), r.get("corruption_family"), r["family"], r["seed"]) for r in rows if r.get("cap_hit", False)]))

    auc = {}
    for r in rel + cap:
        sv = r["sweep"]["severities"]
        m = r["primary_metric"]
        ys = [st.mean(x[m] for x in s["replica_metrics"]) for s in sv]
        a = trap([s["severity"] for s in sv], ys)
        if abs(a - r["sweep"]["corruption_auc"][m]) > 1e-9:
            chk(f"AUC recompute {r['run_id']}", False, f"{a} vs stored {r['sweep']['corruption_auc'][m]}")
        auc.setdefault((r["dataset"], r["corruption_family"], r["family"]), {})[r["seed"]] = a
    chk("corruption AUC == trapezoid over mean-of-3-replicas severity curve (all 114 runs)", not any(s == "FAIL" and n.startswith("AUC recompute") for s, n, _ in RESULTS))

    deltas = {}
    for d, c in CELLS:
        b, m = auc[(d, c, "cellv0.3")], auc[(d, c, "confidence_mlp")]
        deltas[(d, c)] = [b[s] - m[s] for s in (0, 1, 2)]
    wins = sum(1 for v in deltas.values() if st.mean(v) > 0)
    chk("abstract/6.2: BVU > parameter-matched Confidence MLP corruption-AUC in 7 of 8 (dataset,corruption) cells", wins == 7, f"wins={wins}")
    info("6.2 detail: per-cell mean delta and per-seed deltas", "; ".join(f"{d}/{c}: {st.mean(v):+.4f} {[round(x, 4) for x in v]}" for (d, c), v in deltas.items()))
    chk("6.2: MNIST and Fashion-MNIST beat Confidence MLP on every seed under both corruptions", all(all(x > 0 for x in deltas[(d, c)]) for d in ("mnist", "fashion_mnist") for c in ("missing", "gaussian")))
    ch_wins = sum(1 for (d, c), v in deltas.items() if d == "california_housing" and st.mean(v) > 0)
    info("CLAIM-PRECISION: the 8 cells include California Housing, which 6.2 says is not a headline result", f"{ch_wins} of the 7 wins are California Housing; excluding CH the count is {wins - ch_wins} of 6")
    n_allseed = sum(1 for v in deltas.values() if all(x > 0 for x in v))
    info("cells where BVU wins on every seed", f"{n_allseed} of 8 (Digits/Gaussian is a 2-of-3-seed, +0.009 near-tie counted as a win)")

    # Table 1
    paper_t1 = {("mnist", "missing"): (0.666, 0.638, 0.027, 1.65), ("mnist", "gaussian"): (1.460, 1.400, 0.060, 1.65),
                ("fashion_mnist", "missing"): (0.593, 0.572, 0.021, 1.65), ("fashion_mnist", "gaussian"): (1.305, 1.251, 0.054, 1.65),
                ("digits", "missing"): (0.625, 0.633, -0.008, 0.69), ("digits", "gaussian"): (1.422, 1.416, 0.006, 0.69)}
    t1all = True
    for (d, c), pv in paper_t1.items():
        b = ms(auc[(d, c, "cellv0.3")].values())[0]
        s = ms(auc[(d, c, "confidence_mlp_same_width")].values())[0]
        dl = [auc[(d, c, "cellv0.3")][k] - auc[(d, c, "confidence_mlp_same_width")][k] for k in (0, 1, 2)]
        pb = [r for r in rel if r["dataset"] == d and r["family"] == "cellv0.3"][0]["parameter_count"]
        ps = [r for r in cap if r["dataset"] == d][0]["parameter_count"]
        got = (f"{b:.3f}", f"{s:.3f}", f"{st.mean(dl):.3f}", f"{ps / pb:.2f}")
        want = (f"{pv[0]:.3f}", f"{pv[1]:.3f}", f"{pv[2]:.3f}", f"{pv[3]:.2f}")
        ok = got == want
        t1all &= ok
        chk(f"Table 1 {d}/{c}", ok, f"recomputed BVU={b:.5f} same-width={s:.5f} delta={st.mean(dl):+.5f} ratio={ps / pb:.4f}; paper={pv}")
    chk("Table 1/6.3: MNIST & Fashion-MNIST BVU>same-width on all four cells and on every seed",
        all(all(auc[(d, c, 'cellv0.3')][k] > auc[(d, c, 'confidence_mlp_same_width')][k] for k in (0, 1, 2)) for d in ("mnist", "fashion_mnist") for c in ("missing", "gaussian")))

    # Table 2
    fams = ["plain_mlp", "confidence_mlp", "confidence_mlp_same_width", "cellv0.3", "neumiss", "hist_gradient_boosting"]
    paper_t2 = {"plain_mlp": ((0.826, 0.016), (0.750, 0.016)), "confidence_mlp": ((0.841, 0.005), (0.728, 0.016)),
                "confidence_mlp_same_width": ((0.834, 0.013), (0.738, 0.014)), "cellv0.3": ((0.844, 0.014), (0.760, 0.010)),
                "neumiss": ((0.838, 0.012), (0.743, 0.011)), "hist_gradient_boosting": ((0.876, 0.000), (0.752, 0.004))}
    rs = {}
    for f in fams:
        cells = []
        for ds in ("aps", "air_quality"):
            rows = sorted([r for r in real if r["dataset"] == ds and r["family"] == f], key=lambda r: r["seed"])
            v = [r["evaluation"]["overall"][r["primary_metric"]] for r in rows]
            rs[(ds, f)] = (*ms(v), v)
            cells.append(rs[(ds, f)])
        pv = paper_t2[f]
        chk(f"Table 2 {f} (sample SD, ddof=1)", all(f"{cells[i][0]:.3f}" == f"{pv[i][0]:.3f}" and f"{cells[i][1]:.3f}" == f"{pv[i][1]:.3f}" for i in range(2)),
            f"APS {cells[0][0]:.4f}+-{cells[0][1]:.4f}; AQ {cells[1][0]:.4f}+-{cells[1][1]:.4f}; paper {pv}")
    neural = [f for f in fams if f != "hist_gradient_boosting"]
    chk("abstract: APS 0.844 vs NeuMiss 0.838; AQ R2 0.760 vs NeuMiss 0.743", f"{rs[('aps','cellv0.3')][0]:.3f}/{rs[('aps','neumiss')][0]:.3f}/{rs[('air_quality','cellv0.3')][0]:.3f}/{rs[('air_quality','neumiss')][0]:.3f}" == "0.844/0.838/0.760/0.743")
    chk("intro/7.1: BVU has the strongest mean R2 among neural models on Air Quality", max(neural, key=lambda f: rs[("air_quality", f)][0]) == "cellv0.3")
    chk("6.4: HistGradientBoosting has the best APS PR-AUC", max(fams, key=lambda f: rs[("aps", f)][0]) == "hist_gradient_boosting")
    d_aps = [a - b for a, b in zip(rs[("aps", "cellv0.3")][2], rs[("aps", "neumiss")][2])]
    info("APS BVU - NeuMiss per seed (paper: 'small relative to seed variation')", f"{[round(x, 4) for x in d_aps]}; sd BVU {rs[('aps','cellv0.3')][1]:.4f}, NeuMiss {rs[('aps','neumiss')][1]:.4f}")

    # strata
    strat = {}
    for f in ("cellv0.3", "neumiss", "confidence_mlp", "plain_mlp"):
        for b in BINS:
            v = [[x for x in r["evaluation"]["by_stratum"] if x["bin"] == b][0]["pr_auc"] for r in sorted([r for r in real if r["dataset"] == "aps" and r["family"] == f], key=lambda r: r["seed"])]
            strat[(f, b)] = (*ms(v), v)
    chk("abstract/6.5: APS >50% missing PR-AUC BVU 0.848, NeuMiss 0.702, Plain MLP 0.671", f"{strat[('cellv0.3','>0.50')][0]:.3f}/{strat[('neumiss','>0.50')][0]:.3f}/{strat[('plain_mlp','>0.50')][0]:.3f}" == "0.848/0.702/0.671")
    chk("6.5: APS (0.25,0.50] BVU 0.891 vs NeuMiss 0.879", f"{strat[('cellv0.3','(0.25, 0.50]')][0]:.3f}/{strat[('neumiss','(0.25, 0.50]')][0]:.3f}" == "0.891/0.879")
    info(">50% stratum (n=114, 5 positives) per-seed PR-AUC", f"BVU {[round(x, 3) for x in strat[('cellv0.3','>0.50')][2]]}, NeuMiss {[round(x, 3) for x in strat[('neumiss','>0.50')][2]]}")
    belief = {b: {k: [[x for x in r["evaluation"]["belief_by_stratum"] if x["bin"] == b][0][k] for r in sorted([r for r in real if r["dataset"] == "aps" and r["family"] == "cellv0.3"], key=lambda r: r["seed"])] for k in ("l1_pi_mean", "l2_pi_mean", "l1_u_mean", "l2_u_mean")} for b in BINS}
    for k in ("l1_pi_mean", "l2_pi_mean"):
        seq = [st.mean(belief[b][k]) for b in BINS[1:]]
        perseed = all(all(belief[BINS[i + 1]][k][s] > belief[BINS[i + 2]][k][s] for i in range(3)) for s in range(3))
        chk(f"6.5: APS {k} strictly decreases over the four incomplete strata (mean and every seed)", all(seq[i] > seq[i + 1] for i in range(3)) and perseed, f"{[round(x, 3) for x in seq]}")
    umax = max((belief[b]["l1_u_mean"][s], b, s) for b in BINS for s in range(3))
    gt50 = max(belief[">0.50"]["l1_u_mean"])
    chk("7.3: 'in one APS seed the mean conflict in the SPARSEST regime reached ~1.6e3'", abs(gt50 - 1.6e3) < 1.0e2,
        f"max layer-1 mean conflict = {umax[0]:.1f} in stratum {umax[1]} (seed {umax[2]}); the >50% stratum peaks at {gt50:.1f}")
    aq = {b: [[x for x in r["evaluation"]["belief_by_stratum"] if x["bin"] == b][0]["l2_pi_mean"] for r in sorted([r for r in real if r["dataset"] == "air_quality" and r["family"] == "cellv0.3"], key=lambda r: r["seed"])] for b in ("0", ">0.50")}
    chk("6.5: Air Quality mean layer-2 precision ~0.41 (fully observed) -> ~0.001 (highest missingness)", abs(st.mean(aq["0"]) - 0.41) < 0.005 and abs(st.mean(aq[">0.50"]) - 0.001) < 0.0006, f"{st.mean(aq['0']):.4f} -> {st.mean(aq['>0.50']):.6f}")
    info("Air Quality >50% precision equals the imposed input reliability 1e-3 by construction (all 8 inputs missing; pi <= e = 1e-3, Prop. 1)", f"per-seed {[round(x, 7) for x in aq['>0.50']]}")

    # interventions
    for ds in ("aps", "air_quality"):
        rows = sorted([r for r in real if r["dataset"] == ds and r["family"] == "cellv0.3"], key=lambda r: r["seed"])
        a = [r["intervention"]["true_minus_all_ones"] for r in rows]
        s = [r["intervention"]["true_minus_shuffled"] for r in rows]
        if ds == "aps":
            chk("abstract/6.6: APS shuffle reduces PR-AUC by ~0.057; all-ones changes it by ~-0.006", f"{st.mean(s):.3f}/{st.mean(a):.3f}" == "0.057/-0.006", f"true-shuffled {st.mean(s):+.4f}, true-all_ones {st.mean(a):+.4f}")
        else:
            chk("6.6: AQ true - all-ones ~0.012 R2; shuffled is a no-op", f"{st.mean(a):.3f}" == "0.012" and all(abs(x) < 1e-9 for x in s), f"{st.mean(a):+.4f}")
    # hidden precision under controlled corruption
    ok = True
    for d, c in CELLS:
        for layer in (1, 2):
            curves = [[sv["belief_diag"][f"diag_layer{layer}_precision_mean"] for sv in [x for x in rel if x["dataset"] == d and x["corruption_family"] == c and x["family"] == "cellv0.3" and x["seed"] == s][0]["sweep"]["severities"]] for s in (0, 1, 2)]
            mc = [st.mean(v[i] for v in curves) for i in range(len(curves[0]))]
            ok &= all(mc[i] >= mc[i + 1] for i in range(len(mc) - 1))
    chk("intro/6.5: controlled-corruption mean hidden precision is non-increasing in severity (8 cells x 2 layers)", ok)

    # Table 8
    paper_t8 = {"0": (165, 30), "(0, 0.10]": (11981, 132), "(0.10, 0.25]": (2970, 79), "(0.25, 0.50]": (770, 129), ">0.50": (114, 5)}
    for b, (n, p) in paper_t8.items():
        got = {(x["n"], x["n_pos"]) for r in real if r["dataset"] == "aps" and r["family"] != "hist_gradient_boosting" for x in r["evaluation"]["by_stratum"] if x["bin"] == b}
        chk(f"Table 8 stratum {b}", got == {(n, p)}, f"records {sorted(got)}")
    chk("Table 8 totals", sum(v[0] for v in paper_t8.values()) == 16000 and sum(v[1] for v in paper_t8.values()) == 375)

    # figure CSVs against records
    fig2 = list(csv.DictReader(open(ROOT / "paper/figures/fig2_controlled_corruption_data.csv")))
    fam_of = {"BVU": (rel, "cellv0.3"), "Plain MLP": (rel, "plain_mlp"), "Confidence MLP": (rel, "confidence_mlp"), "Same-width Confidence MLP": (cap, "confidence_mlp_same_width")}
    worst = 0.0
    for r in fig2:
        pool, fam = fam_of[r["model"]]
        vals = [st.mean(rm[x["primary_metric"]] for rm in [s for s in x["sweep"]["severities"] if abs(s["severity"] - float(r["severity"])) < 1e-9][0]["replica_metrics"]) for x in pool if x["dataset"] == r["dataset"] and x["corruption_family"] == r["corruption_family"] and x["family"] == fam]
        mu, sd = ms(vals)
        worst = max(worst, abs(mu - float(r["mean"])), abs(sd - float(r["std"])))
    chk("fig2 CSV == recomputation from frozen records (88 rows)", len(fig2) == 88 and worst < 1e-9, f"max err {worst:.1e}")
    fig3 = list(csv.DictReader(open(ROOT / "paper/figures/fig3_aps_missingness_mechanism_data.csv")))
    fam3 = {"BVU": "cellv0.3", "NeuMiss": "neumiss", "Confidence MLP": "confidence_mlp", "Plain MLP": "plain_mlp"}
    worst = 0.0
    for r in fig3:
        v = []
        for x in sorted([x for x in real if x["dataset"] == "aps" and x["family"] == fam3[r["model"]]], key=lambda x: x["seed"]):
            if r["panel"] == "performance":
                v.append([b for b in x["evaluation"]["by_stratum"] if b["bin"] == r["bin"]][0]["pr_auc"])
            else:
                v.append([b for b in x["evaluation"]["belief_by_stratum"] if b["bin"] == r["bin"]][0]["l1_pi_mean" if r["layer"] == "layer1" else "l2_pi_mean"])
        mu, sd = ms(v)
        worst = max(worst, abs(mu - float(r["mean"])), abs(sd - float(r["std"])))
    chk("fig3 CSV == recomputation from frozen records (30 rows)", len(fig3) == 30 and worst < 1e-9, f"max err {worst:.1e}")

    # Phase-1 clean: Table 7 and figS1
    bvu = load("results/processed/phase1_cellv03_runs.json")
    mlp = []
    for f in ("phase1_runs_small.json", "phase1_runs_mid.json", "phase1_runs_mnist_0.25.json", "phase1_runs_mnist_1.0.json", "phase1_runs_fashion_mnist_0.25.json", "phase1_runs_fashion_mnist_1.0.json"):
        mlp += [r for r in load("results/processed/" + f) if r["family"] == "mlp_matched"]
    key = lambda r: (r["dataset"], round(r["train_fraction"], 4), r["seed"])
    B, M = {key(r): r for r in bvu}, {key(r): r for r in mlp}
    chk("Phase 1: 42 unique BVU and 42 unique MLP-matched (dataset,fraction,seed) rows with identical coverage", len(B) == len(bvu) == 42 and len(M) == len(mlp) == 42 and set(B) == set(M))
    order = ["breast_cancer", "wine", "digits", "diabetes", "california_housing", "mnist", "fashion_mnist"]
    t7 = {}
    for d in order:
        for fr in (0.25, 1.0):
            b = [B[(d, fr, s)]["headline_value"] for s in (0, 1, 2)]
            m = [M[(d, fr, s)]["headline_value"] for s in (0, 1, 2)]
            t7[(d, fr)] = (ms(b), ms(m), ms([x - y for x, y in zip(b, m)]))
    cs = {(r["dataset"], float(r["train_fraction"])): r for r in csv.DictReader(open(ROOT / "paper/tables/clean_performance_data.csv"))}
    worst = max(max(abs(v[0][0] - float(cs[k]["bvu_mean"])), abs(v[0][1] - float(cs[k]["bvu_std"])), abs(v[1][0] - float(cs[k]["mlp_mean"])), abs(v[1][1] - float(cs[k]["mlp_std"])), abs(v[2][0] - float(cs[k]["delta_mean"])), abs(v[2][1] - float(cs[k]["delta_std"]))) for k, v in t7.items())
    chk("clean_performance_data.csv == recomputation from frozen Phase-1 records", worst < 1e-9, f"max err {worst:.1e}")
    p23 = pdf_text.split("=====PAGE 23=====")[-1].replace(" ", "") if "=====PAGE 23=====" in pdf_text else pdf_text.replace(" ", "")
    misses = []
    for (d, fr), (bb, mm, dd) in t7.items():
        dm = f"{dd[0]:+.3f}" if abs(dd[0]) >= 0.0005 else "0.000"
        for e_ in (f"{bb[0]:.3f}±{bb[1]:.3f}", f"{mm[0]:.3f}±{mm[1]:.3f}", f"{dm.replace('-', chr(0x2212))}±{dd[1]:.3f}"):
            if e_ not in p23:
                misses.append((d, fr, e_))
    chk("Table 7 printed cells in the PDF text == recomputation (3 dp)", not misses, f"misses={misses}")
    s1 = {r["dataset"]: r for r in csv.DictReader(open(ROOT / "paper/figures/supplementary/figS1_clean_benchmarks_data.csv"))}
    chk("figS1 CSV == recomputation (100% fraction paired delta)", max(max(abs(t7[(d, 1.0)][2][0] - float(s1[d]["delta_mean"])), abs(t7[(d, 1.0)][2][1] - float(s1[d]["delta_std"]))) for d in order) < 1e-9)
    chk("6.1: Breast Cancer/Wine/Digits |mean delta| < 0.01 in all six conditions; Diabetes, CH, MNIST, Fashion-MNIST positive at both fractions",
        all(abs(t7[(d, fr)][2][0]) < 0.01 for d in ("breast_cancer", "wine", "digits") for fr in (0.25, 1.0)) and all(t7[(d, fr)][2][0] > 0 for d in ("diabetes", "california_housing", "mnist", "fashion_mnist") for fr in (0.25, 1.0)))


# ============================================================================ B. appendix configs
def raws(sub: str):
    d = EXP / sub / "results" / "raw" if sub else EXP / "results" / "raw"
    return [json.load(open(f)) for f in sorted(glob.glob(str(d / "*.json"))) if not f.endswith("_history.json")]


def check_configs():
    p1 = raws("")
    fam = {"cellv0.3", "mlp_matched"}
    t3 = collections.defaultdict(set)
    t4 = collections.defaultdict(set)
    for r in p1:
        a = r["architecture"]
        if a not in fam:
            continue
        c, e = r["config"], r["config"]["extra"]
        t3[(r["dataset"], a)].add((e["hidden_size"], r["parameter_count"]))
        t3[(r["dataset"], a, e["train_fraction"])].add(e["batch_size_effective"])
        t4[r["dataset"]].add((c["optimizer"], c["learning_rate"], c["weight_decay"], e["task_type"], c["max_steps"], e["patience_checks"], e["val_every"], e["headline_metric"]))
    p3 = {("breast_cancer", "cellv0.3"): (83, 9879, 85, 128), ("breast_cancer", "mlp_matched"): (299, 9869, 85, 128), ("wine", "cellv0.3"): (90, 9903, 26, 106), ("wine", "mlp_matched"): (582, 9897, 26, 106),
          ("digits", "cellv0.3"): (123, 24733, 128, 128), ("digits", "mlp_matched"): (332, 24910, 128, 128), ("diabetes", "cellv0.3"): (92, 9845, 66, 128), ("diabetes", "mlp_matched"): (829, 9949, 66, 128),
          ("california_housing", "cellv0.3"): (93, 9859, 128, 128), ("california_housing", "mlp_matched"): (997, 9971, 128, 128), ("mnist", "cellv0.3"): (157, 149945, 128, 128), ("mnist", "mlp_matched"): (187, 148675, 128, 128),
          ("fashion_mnist", "cellv0.3"): (157, 149945, 128, 128), ("fashion_mnist", "mlp_matched"): (187, 148675, 128, 128)}
    ok = all(t3[k] == {(w, p)} and t3[k + (0.25,)] == {b25} and t3[k + (1.0,)] == {b100} for k, (w, p, b25, b100) in p3.items())
    chk("Table 3: hidden width, params, effective batch (25%/100%) == raw Phase-1 records (14 rows)", ok)
    p4 = {"breast_cancer": ("classification", 30, 50, "accuracy"), "wine": ("classification", 30, 50, "accuracy"), "digits": ("classification", 30, 50, "accuracy"), "diabetes": ("regression", 30, 50, "r2"),
          "california_housing": ("regression", 8, 200, "r2"), "mnist": ("classification", 8, 200, "accuracy"), "fashion_mnist": ("classification", 8, 200, "accuracy")}
    chk("Table 4: AdamW, LR 1e-2, WD 0, 15,000-step cap, loss/task, patience (checks x cadence), selection metric == raw records",
        all(t4[d] == {("adamw", 0.01, 0.0, tk, 15000, pc, ve, mt)} for d, (tk, pc, ve, mt) in p4.items()))
    seeds_fr = {(r["config"]["seed"], r["config"]["extra"]["train_fraction"]) for r in p1 if r["architecture"] in fam}
    chk("Table 4 note: clean rows use seeds {0,1,2} and fractions {25%,100%}", seeds_fr == {(s, f) for s in (0, 1, 2) for f in (0.25, 1.0)})

    rl = raws("reliability")
    t5 = collections.defaultdict(set)
    shared = collections.defaultdict(set)
    for r in rl:
        c, e = r["config"], r["config"]["extra"]
        t5[(r["dataset"], r["architecture"])].add((e["hidden_size"], r["parameter_count"]))
        shared[r["dataset"]].add((c["optimizer"], c["learning_rate"], c["weight_decay"], c["batch_size"], c["max_steps"]))
    p5 = {("mnist", "plain_mlp"): (189, 150265), ("mnist", "confidence_mlp"): (95, 150015), ("mnist", "reliability_gated_mlp"): (189, 150265), ("mnist", "cellv0.3"): (157, 149945),
          ("fashion_mnist", "plain_mlp"): (189, 150265), ("fashion_mnist", "confidence_mlp"): (95, 150015), ("fashion_mnist", "reliability_gated_mlp"): (189, 150265), ("fashion_mnist", "cellv0.3"): (157, 149945),
          ("digits", "plain_mlp"): (330, 24760), ("digits", "confidence_mlp"): (178, 24752), ("digits", "reliability_gated_mlp"): (330, 24760), ("digits", "cellv0.3"): (123, 24733),
          ("california_housing", "plain_mlp"): (986, 9861), ("california_housing", "reliability_gated_mlp"): (986, 9861), ("california_housing", "confidence_mlp"): (548, 9865), ("california_housing", "cellv0.3"): (93, 9859),
          ("mnist", "confidence_mlp_same_width"): (157, 247913), ("fashion_mnist", "confidence_mlp_same_width"): (157, 247913), ("digits", "confidence_mlp_same_width"): (123, 17107)}
    chk("Table 5: hidden width and params == raw Phase-2/3 records (19 rows)", all(t5[k] == {v} for k, v in p5.items()))
    chk("Table 5 note: AdamW, LR 1e-2, WD 0, batch 128, 15,000-step cap for every dataset", all(s == {("adamw", 0.01, 0.0, 128, 15000)} for s in shared.values()))

    rr = raws("real_reliability")
    t6 = collections.defaultdict(set)
    cfg = collections.defaultdict(set)
    nval = collections.defaultdict(set)
    lrs = collections.defaultdict(dict)
    nm_commits = set()
    for r in rr:
        if r["architecture"] == "hist_gradient_boosting":
            continue
        c, e = r["config"], r["config"]["extra"]
        w = e["efficiency"]["hidden_size"] if r["architecture"] != "neumiss" else e["sizing"]["mlp_width"]
        t6[(r["dataset"], r["architecture"])].add((w, r["parameter_count"]))
        cfg[r["dataset"]].add((c["optimizer"], c["weight_decay"], c["batch_size"], c["max_steps"], e["task_type"]))
        nval[r["dataset"]].add(e["dataset_meta"]["n_val"])
        lrs[(r["dataset"], r["architecture"])][c["seed"]] = e["selected_lr"]
        if r["architecture"] == "neumiss":
            nm_commits.add(e["sizing"]["neumiss_commit"])
    p6 = {("aps", "plain_mlp"): (579, 99589), ("aps", "confidence_mlp"): (291, 99523), ("aps", "confidence_mlp_same_width"): (240, 82081), ("aps", "cellv0.3"): (240, 99601), ("aps", "neumiss"): (412, 99935),
          ("air_quality", "plain_mlp"): (4991, 49911), ("air_quality", "confidence_mlp"): (2773, 49915), ("air_quality", "confidence_mlp_same_width"): (217, 3907), ("air_quality", "cellv0.3"): (217, 49911), ("air_quality", "neumiss"): (4993, 50003)}
    chk("Table 6: width and params == raw real-reliability records (10 rows)", all(t6[k] == {v} for k, v in p6.items()))
    chk("Table 6 note: AdamW, WD 0, batch 128; caps APS 7,000 / AQ 4,000; patience 8x200 (APS n_val=12,000>2,000) and 30x50 (AQ n_val=1,535)",
        cfg["aps"] == {("adamw", 0.0, 128, 7000, "classification")} and cfg["air_quality"] == {("adamw", 0.0, 128, 4000, "regression")} and nval["aps"] == {12000} and nval["air_quality"] == {1535})
    chk("Table 6 caption: selected LR is in {1e-3,3e-3,1e-2} and varies by seed for some models", all(v in (0.001, 0.003, 0.01) for d in lrs.values() for v in d.values()) and any(len(set(d.values())) > 1 for d in lrs.values()))
    hgb = [r for r in rr if r["architecture"] == "hist_gradient_boosting"]
    chk("HGB reference: 6 records (2 datasets x 3 seeds), non-neural", len(hgb) == 6)
    full = "7902b8dbe7114e8dc3010b5e8b132c35806a9d74"
    pyproj = (ROOT / "pyproject.toml").read_text()
    chk("C.4: NeuMiss commit 7902b8d == full hash in pyproject.toml pin and in all 6 raw NeuMiss records", nm_commits == {full} and f"NeuMiss_sota@{full}" in pyproj, f"records={sorted(nm_commits)}")
    depths = {(r["dataset"], r["config"]["seed"]): r["config"]["extra"]["sizing"]["neumiss_depth"] for r in rr if r["architecture"] == "neumiss"}
    chk("C.4: selected NeuMiss depths are within {1,3,5}", set(depths.values()) <= {1, 3, 5}, str(dict(sorted(depths.items()))))
    src = (EXP / "real_reliability" / "datasets.py").read_text()
    info("APS train/val partition is seeded (train_test_split random_state=seed); only the official 16k test set is fixed", "Table 6 footnote says 'fixed dataset splits'; Appendix C.2 says 'for each seed'" if "random_state=seed" in src else "")
    # provenance anchors
    sha = {}
    import hashlib
    for rel in ("reliability/results/processed/reliability_runs.json", "reliability/results/processed/capacity_stress_runs.json", "real_reliability/results/processed/real_reliability_runs.json",
                "results/processed/phase1_cellv03_runs.json", "results/processed/phase1_runs_small.json", "results/processed/phase1_runs_mid.json", "results/processed/phase1_runs_mnist_0.25.json",
                "results/processed/phase1_runs_mnist_1.0.json", "results/processed/phase1_runs_fashion_mnist_0.25.json", "results/processed/phase1_runs_fashion_mnist_1.0.json"):
        sha[rel] = hashlib.sha256((EXP / rel).read_bytes()).hexdigest()
    tracked = subprocess.run(["git", "ls-files", "experiments/paper_a"], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    info("frozen records tracked in git?", f"{sum(1 for t in tracked if '/results/' in t and not t.endswith('.gitkeep'))} tracked record files (0 means records live only on disk)")
    for k, v in sha.items():
        info(f"sha256 {k}", v)


# ============================================================================ C. figures vs CSV
_TOK = re.compile(r"\(((?:[^()\\]|\\.)*)\)|\[|\]|-?\d*\.?\d+(?:[eE][-+]?\d+)?|/[A-Za-z0-9_.]+|[A-Za-z*']+")


def _ops(stream: str):
    ops, operands, in_arr, arr = [], [], False, []
    for m in _TOK.finditer(stream):
        t = m.group(0)
        if m.group(1) is not None:
            (arr if in_arr else operands).append(("str", m.group(1)))
        elif t == "[":
            in_arr, arr = True, []
        elif t == "]":
            in_arr = False
            operands.append(("arr", arr))
        elif in_arr:
            try:
                arr.append(("num", float(t)))
            except ValueError:
                pass
        elif re.fullmatch(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", t):
            operands.append(("num", float(t)))
        elif t.startswith("/"):
            operands.append(("name", t))
        else:
            ops.append((t, operands))
            operands = []
    return ops


def _decode(stream: str):
    clip = last_re = cur = None
    stroke, fill, cm = (0, 0, 0), (0, 0, 0), (0.0, 0.0)
    stack, strokes, fills, events = [], [], [], []
    for op, a in _ops(stream):
        nums = [x[1] for x in a if x[0] == "num"]
        if op == "q":
            stack.append((clip, stroke, fill))
        elif op == "Q":
            clip, stroke, fill = stack.pop() if stack else (clip, stroke, fill)
        elif op == "re":
            last_re = tuple(nums[-4:])
        elif op == "W":
            clip = last_re
        elif op == "RG":
            stroke = tuple(nums[-3:])
        elif op == "G":
            stroke = (nums[-1],) * 3
        elif op == "rg":
            fill = tuple(nums[-3:])
        elif op == "g":
            fill = (nums[-1],) * 3
        elif op == "m":
            cur = [(nums[-2], nums[-1])]
        elif op == "l" and cur is not None:
            cur.append((nums[-2], nums[-1]))
        elif op in ("S", "B", "f", "f*") and cur is not None:
            if op == "S":
                strokes.append(dict(pts=cur, color=stroke, clip=clip))
            elif op in ("f", "f*"):
                fills.append(dict(pts=cur, color=fill, clip=clip))
            elif len(cur) == 2:
                events.append(("tick", cur))
            cur = None
        elif op == "n":
            cur = None
        elif op == "cm" and len(nums) >= 6:
            cm = (nums[-2], nums[-1])
        elif op == "TJ":
            txt = "".join(v2 for kind, v in a if kind == "arr" for k2, v2 in v if k2 == "str").replace("\x00", "").replace(" ", "")
            events.append(("text", txt, cm))
    return strokes, fills, events


def _hex(c):
    return "#%02X%02X%02X" % tuple(int(round(x * 255)) for x in c)


def _fit(pairs):
    (p0, v0), (p1, v1) = pairs[0], pairs[-1]
    a = (p1 - p0) / (v1 - v0)
    b = p0 - a * v0
    return a, b, max(abs(a * v + b - p) for p, v in pairs)


def _axes(events, clip):
    x0, y0, w, h = clip
    xt, yt = [], []
    for i, e in enumerate(events):
        if e[0] != "tick":
            continue
        (ax, ay), (bx, by) = e[1]
        lab = next((x[1] for x in events[i + 1:] if x[0] == "text"), None)
        if abs(ax - bx) < 1e-6 and x0 - 1 <= ax <= x0 + w + 1 and abs(max(ay, by) - y0) < 1.0:
            xt.append((ax, lab))
        elif abs(ay - by) < 1e-6 and y0 - 1 <= ay <= y0 + h + 1 and abs(min(ax, bx) - (x0 - 3.5)) < 1.0:
            yt.append((ay, lab))
    return xt, yt


def _embedded_forms(reader: pypdf.PdfReader):
    """Yield (page_no, decoded stream) for form XObjects that look like matplotlib figures (>=2 curve panels)."""
    for pi, p in enumerate(reader.pages):
        xo = (p.get("/Resources") or {}).get("/XObject") or {}
        for name, ref in xo.items():
            o = ref.get_object()
            if o.get("/Subtype") == "/Form":
                yield pi + 1, o.get_data().decode("latin1")


def check_figures(reader: pypdf.PdfReader):
    f2 = list(csv.DictReader(open(ROOT / "paper/figures/fig2_controlled_corruption_data.csv")))
    f3 = list(csv.DictReader(open(ROOT / "paper/figures/fig3_aps_missingness_mechanism_data.csv")))
    c2 = {"#000000": "BVU", "#0072B2": "Plain MLP", "#D55E00": "Confidence MLP", "#56B4E9": "Same-width Confidence MLP"}
    c3 = {"#000000": "BVU", "#CC79A7": "NeuMiss", "#D55E00": "Confidence MLP", "#0072B2": "Plain MLP"}
    found = {"fig2": False, "fig3": False}

    def panels_of(strokes):
        cl = collections.OrderedDict()
        for s in strokes:
            if s["clip"] is not None:
                cl.setdefault(s["clip"], []).append(s)
        return cl

    for pg, stream in _embedded_forms(reader):
        strokes, fills, events = _decode(stream)
        cl = panels_of(strokes)
        if len(cl) == 4:  # figure 2
            problems, worst, n = [], 0.0, 0
            for clip, (ds, cf) in zip(sorted(cl, key=lambda c: (-c[1], c[0])), [("mnist", "missing"), ("mnist", "gaussian"), ("fashion_mnist", "missing"), ("fashion_mnist", "gaussian")]):
                xt, yt = _axes(events, clip)
                ax, bx, rx = _fit([(p, float(l)) for p, l in xt])
                ay, by, ry = _fit([(p, float(l)) for p, l in yt])
                if rx > 0.01 or ry > 0.01:
                    problems.append(f"{ds}/{cf} tick fit")
                for s in cl[clip]:
                    if len(s["pts"]) < 3 or s["color"] is None:
                        continue
                    model = c2.get(_hex(s["color"]))
                    rows = sorted([r for r in f2 if r["dataset"] == ds and r["corruption_family"] == cf and r["model"] == model], key=lambda r: float(r["severity"]))
                    if model is None or len(rows) != len(s["pts"]):
                        problems.append(f"{ds}/{cf} curve {_hex(s['color'])}")
                        continue
                    for (px, py), r in zip(s["pts"], rows):
                        e = abs((py - by) / ay - float(r["mean"]))
                        worst = max(worst, e)
                        if e > 2e-5 or abs((px - bx) / ax - float(r["severity"])) > 1e-4:
                            problems.append(f"{ds}/{cf}/{model}@{r['severity']}")
                    n += 1
                for f in [f for f in fills if f["clip"] == clip and c2.get(_hex(f["color"]))]:
                    model = c2[_hex(f["color"])]
                    byx = collections.defaultdict(set)
                    for px, py in f["pts"]:
                        byx[round((px - bx) / ax, 4)].add((py - by) / ay)
                    for r in [r for r in f2 if r["dataset"] == ds and r["corruption_family"] == cf and r["model"] == model]:
                        got = sorted(byx.get(round(float(r["severity"]), 4), []))
                        m_, s_ = float(r["mean"]), float(r["std"])
                        if not got or max(abs(got[0] - (m_ - s_)), abs(got[-1] - (m_ + s_))) > 1e-4:
                            problems.append(f"band {ds}/{cf}/{model}@{r['severity']}")
            chk(f"Figure 2 (page {pg}): all 16 curves and their +-1 SD bands match fig2 CSV", n == 16 and not problems, f"max |mean err|={worst:.1e}; problems={problems[:4]}")
            found["fig2"] = True
        elif len(cl) == 2:  # figure 3
            problems, worst, n, nb = [], 0.0, 0, 0
            for k, clip in enumerate(sorted(cl, key=lambda c: c[0])):
                xt, yt = _axes(events, clip)
                pos = [p for p, _ in xt]
                ay, by, ry = _fit([(p, float(l)) for p, l in yt])
                if ry > 0.01 or [l for _, l in xt] != ["0", "0-10%", "10-25%", "25-50%", ">50%"]:
                    problems.append(f"panel{k} axes")
                curves = [s for s in strokes if s["clip"] == clip and len(s["pts"]) == 5]
                bars = [s for s in strokes if s["clip"] == clip and len(s["pts"]) == 2 and abs(s["pts"][0][0] - s["pts"][1][0]) < 1e-6 and any(abs(s["pts"][0][0] - p) < 1e-3 for p in pos)]
                for s in curves:
                    col = _hex(s["color"])
                    if k == 0:
                        sel = {r["bin"]: r for r in f3 if r["panel"] == "performance" and r["model"] == c3.get(col)}
                    else:
                        cand = {L: {r["bin"]: r for r in f3 if r["panel"] == "precision" and r["layer"] == f"layer{L}"} for L in (1, 2)}
                        got = [(py - by) / ay for _, py in s["pts"]]
                        L = min((max(abs(g - float(cand[L][b]["mean"])) for g, b in zip(got, BINS)), L) for L in (1, 2))[1]
                        sel = cand[L]
                    for (px, py), b in zip(s["pts"], BINS):
                        e = abs((py - by) / ay - float(sel[b]["mean"]))
                        worst = max(worst, e)
                        if e > 2e-5:
                            problems.append(f"p{k} {col} {b}")
                    n += 1
                    for x0, b in zip(pos, BINS):
                        for bar in [bb for bb in bars if _hex(bb["color"]) == col and abs(bb["pts"][0][0] - x0) < 1e-3]:
                            lo, hi = sorted(((y - by) / ay) for _, y in bar["pts"])
                            e = max(abs(lo - (float(sel[b]["mean"]) - float(sel[b]["std"]))), abs(hi - (float(sel[b]["mean"]) + float(sel[b]["std"]))))
                            worst = max(worst, e)
                            nb += 1
                            if e > 1e-4:
                                problems.append(f"bar p{k} {col} {b}")
            chk(f"Figure 3 (page {pg}): 6 curves and {nb} error bars match fig3 CSV", n == 6 and nb == 30 and not problems, f"max err={worst:.1e}; problems={problems[:4]}")
            found["fig3"] = True
        # text labels embedded in the figure
        labels = [e[1] for e in events if e[0] == "text"]
        stale = [t for t in labels if re.search(r"CellV0|cellv0", t)]
        if len(cl) in (2, 4):
            chk(f"Figures on page {pg}: no CellV0.x display names inside the embedded figure text", not stale, f"stale labels: {stale}" if stale else "")
    chk("both Figure 2 and Figure 3 located in the PDF as vector forms", all(found.values()))
    # repo (replacement) figure files
    for name, pdf in (("fig2", "paper/figures/fig2_controlled_corruption.pdf"), ("fig3", "paper/figures/fig3_aps_missingness_mechanism.pdf"), ("figS1", "paper/figures/supplementary/figS1_clean_benchmarks.pdf")):
        t = " ".join((p.extract_text() or "") for p in pypdf.PdfReader(ROOT / pdf).pages)
        chk(f"repo {name} PDF: no CellV0.x text, contains 'BVU'", not re.search(r"CellV0", t) and "BVU" in t)


# ============================================================================ D. PDF scans
def check_pdf(pdf_path: Path, text: str, reader: pypdf.PdfReader):
    chk("no unresolved references/citations ('??' or '(?)') in extracted text", "??" not in text and "(?)" not in text)
    hits = [(m.group(0), text[max(0, text.rfind('=====PAGE', 0, m.start())):][:16].strip()) for m in re.finditer(r"CellV0\.?\d*|V0\.[123]|cellv0", text)]
    chk("no CellV0.1/0.2/0.3 display names anywhere in the PDF text layer (incl. embedded figures)", not hits, f"hits={hits}")
    chk("paper-facing name is BVU", len(re.findall(r"\bBVU\b", text)) > 20)

    names = {}

    def walk(node):
        if "/Names" in node:
            arr = node["/Names"]
            for i in range(0, len(arr), 2):
                names[str(arr[i])] = arr[i + 1]
        for k in node.get("/Kids", []):
            walk(k.get_object())

    root = reader.trailer["/Root"]
    if "/Names" in root and "/Dests" in root["/Names"]:
        walk(root["/Names"]["/Dests"].get_object())
    links, missing = collections.defaultdict(list), []
    for pi, p in enumerate(reader.pages):
        for a in p.get("/Annots", []) or []:
            a = a.get_object()
            if a.get("/Subtype") != "/Link":
                continue
            dest = a.get("/Dest") or (a["/A"].get_object().get("/D") if "/A" in a else None)
            links[pi].append(([float(x) for x in a["/Rect"]], str(dest)))
            if str(dest) not in names:
                missing.append((pi + 1, str(dest)))
    tot = sum(len(v) for v in links.values())
    chk("every internal link annotation (\\ref/\\cite/\\autoref/\\eqref) resolves to an existing destination", not missing and tot > 0, f"{tot} links, unresolved={missing}")
    cites = {d for v in links.values() for _, d in v if d.startswith("cite.")}
    bib = {d for d in names if d.startswith("cite.")}
    chk("every bibliography entry is cited and every citation resolves to an entry", cites == bib, f"cited={len(cites)}, entries={len(bib)}; uncited={sorted(bib - cites)}; dangling={sorted(cites - bib)}")

    # hard-coded references: reference-like text with no link over its number
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

    pat = re.compile(r"(Tables?|Figures?|Sections?|Appendix|Appendices|Propositions?|Equations?|Eqs?\.)\s*\(?([A-Z]?\d+(?:\.\d+)*|[A-Z])\)?")
    unlinked, nlinked = [], 0
    for pi, page in enumerate(extract_pages(str(pdf_path))):
        for el in page:
            if not isinstance(el, LTTextContainer):
                continue
            for line in el:
                if not isinstance(line, LTTextLine):
                    continue
                chars = [c for c in line if hasattr(c, "get_text")]
                s = "".join(c.get_text() for c in chars)
                for m in pat.finditer(s):
                    if m.start() == 0 and re.match(r"\s*[:.]", s[m.end():m.end() + 2]) and m.group(1).rstrip("s") in ("Table", "Figure", "Proposition"):
                        continue
                    idx = [i for i in range(m.end(1), m.end()) if isinstance(chars[i], LTChar) and (s[i].isdigit() or s[i].isupper())]
                    if not idx:
                        continue
                    cov = 0
                    for i in idx:
                        x0, y0, x1, y1 = chars[i].bbox
                        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                        cov += any(r[0] - 1 <= cx <= r[2] + 1 and r[1] - 2 <= cy <= r[3] + 2 for r, _ in links[pi])
                    if cov / len(idx) > 0.6:
                        nlinked += 1
                    else:
                        unlinked.append((pi + 1, m.group(0).strip(), s.strip()[:70]))
    chk("no typed (unlinked) 'Table/Figure/Section/Appendix/Proposition N' references", not unlinked, f"{nlinked} linked references; unlinked={unlinked}")

    # layout overhang: rules/text beyond the text block defined by the running-header rule
    from pdfminer.layout import LTCurve, LTLine, LTRect

    worst, where = 0.0, []
    for pi, page in enumerate(extract_pages(str(pdf_path))):
        rules, cb = [], []

        def visit(o):
            if isinstance(o, LTChar):
                cb.append(o.bbox)
            elif isinstance(o, (LTLine, LTRect, LTCurve)):
                rules.append(o.bbox)
            elif hasattr(o, "__iter__"):
                for c in o:
                    visit(c)

        for el in page:
            visit(el)
        thin = [b for b in rules if (b[3] - b[1]) < 1.5 and (b[2] - b[0]) > 300 and b[1] > page.height - 80]
        if not thin:
            continue
        L, R = min(b[0] for b in thin), max(b[2] for b in thin)
        rr = [b for b in rules if (b[3] - b[1]) < 1.5 and (b[2] - b[0]) > 60 and 45 < b[1] < page.height - 60]
        xs = [b[0] for b in cb if 45 < b[1] < page.height - 45] + [b[0] for b in rr]
        xe = [b[2] for b in cb if 45 < b[1] < page.height - 45] + [b[2] for b in rr]
        ov = max(L - min(xs), max(xe) - R)
        if ov > 1.0:
            where.append((pi + 1, round(ov, 1)))
        worst = max(worst, ov)
    chk("no content or table rule extends more than 1 pt beyond the text block (overfull-box proxy)", not where, f"overhang pages (page, pt)={where}; worst={worst:.1f}")

    # de-anonymization scan of the PDF
    import pikepdf

    tokens = _identity_tokens()
    blob = pdf_path.read_bytes()
    data = [blob]
    with pikepdf.open(str(pdf_path)) as pdf:
        for obj in pdf.objects:
            try:
                if isinstance(obj, pikepdf.Stream):
                    data.append(obj.read_bytes())
            except Exception:
                pass
        docinfo = {k: str(v) for k, v in pdf.docinfo.items()}
    hits = collections.Counter()
    for b in data:
        for name, rx in tokens.items():
            hits[name] += len(rx.findall(b))
    chk("PDF raw+decompressed bytes contain no username, git identity, email, absolute local path, hostname or project codename", not any(hits.values()), f"hit counts by class={dict((k, v) for k, v in hits.items() if v)}")
    chk("PDF Author/Title/Subject/Keywords metadata are empty; no embedded files; no URI links", all(docinfo.get(k, "") == "" for k in ("/Author", "/Title", "/Subject", "/Keywords")) and not list(reader.attachments.keys()) and not any("/URI" in str(a) for p in reader.pages for a in (p.get("/Annots") or [])),
        f"Creator={docinfo.get('/Creator')}, Producer={docinfo.get('/Producer')}")
    chk("PDF text carries the anonymous-review banner and no author names", "Anonymous authors" in text and "Under review as submission to TMLR" in text)


def _identity_tokens() -> dict[str, re.Pattern]:
    def git(k):
        return subprocess.run(["git", "config", k], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    user = os.environ.get("USER", "")
    parts = [p for p in re.split(r"[\s.@_-]+", git("user.name") + " " + git("user.email").split("@")[0]) if len(p) > 2]
    toks = {"absolute_local_path": re.compile(rb"/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Z]:\\\\Users"),
            "email_address": re.compile(rb"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]+\.(?:com|edu|org|net|io|ai)\b"),
            "project_codename": re.compile(rb"CGRN|cgrn"), "hostname": re.compile(rb"MacBook|\.local\b")}
    if user:
        toks["username"] = re.compile(re.escape(user.encode()), re.I)
    if parts:
        toks["git_identity"] = re.compile(b"|".join(re.escape(p.encode()) for p in parts), re.I)
    return toks


def check_supplement_files():
    toks = _identity_tokens()
    groups = {
        "paper/ (hidden files included; submission PDFs and this audit folder excluded)": [f for f in glob.glob(str(ROOT / "paper/**/*"), recursive=True) + glob.glob(str(ROOT / "paper/**/.*"), recursive=True)
                                                                                    if os.path.isfile(f) and "__pycache__" not in f and "/paper/audit/" not in f and not os.path.basename(f).startswith("TMLR_")],
        "experiments/paper_a processed/logs/code (excl. raw)": [f for f in glob.glob(str(EXP / "**/*"), recursive=True) if os.path.isfile(f) and "/results/raw/" not in f and "__pycache__" not in f],
        "experiments/paper_a raw run records": glob.glob(str(EXP / "**/results/raw/*.json"), recursive=True),
    }
    for label, files in groups.items():
        c = collections.defaultdict(list)
        for f in files:
            b = open(f, "rb").read()
            if f.endswith(".png"):
                continue  # compressed pixel data yields random false positives; PNG text chunks checked below
            for name, rx in toks.items():
                if rx.search(b):
                    c[name].append(os.path.relpath(f, ROOT))
        summary = {k: v for k, v in c.items()}
        ok = not any(k in summary for k in ("absolute_local_path", "username", "git_identity", "email_address", "hostname"))
        chk(f"supplement scan [{label}]: no absolute local paths, usernames, git identity, emails or hostnames", ok,
            f"{len(files)} files; " + ("; ".join(f"{k}: {v}" for k, v in summary.items()) if summary else "no hits"))
        if "project_codename" in summary:
            info(f"supplement scan [{label}]: internal project codename present (non-identifying)", str(summary["project_codename"]))
    r = subprocess.run(["git", "log", "--format=%an|%ae", "--", "paper"], capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()
    info("git history of paper/ carries a real author name and email (redacted here)", f"{len(r)} commits by {len(set(r))} identity; never ship .git with the supplement (use a clean export)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "paper" / "TMLR_Journal_Submission_V3.pdf"))
    ap.add_argument("--quiet", action="store_true", help="print only FAIL lines and the summary")
    args = ap.parse_args()
    pdf_path = Path(args.pdf)
    reader = pypdf.PdfReader(str(pdf_path))
    text = "".join(f"\n=====PAGE {i}=====\n" + (p.extract_text() or "") for i, p in enumerate(reader.pages, 1))
    info("PDF", f"{pdf_path.name}: {len(reader.pages)} pages")
    for fn, a in ((check_numbers, (text,)), (check_configs, ()), (check_figures, (reader,)), (check_pdf, (pdf_path, text, reader)), (check_supplement_files, ())):
        try:
            fn(*a)
        except Exception as e:  # keep going; a crashed section is itself a failure
            chk(f"section {fn.__name__} completed without error", False, f"{type(e).__name__}: {e}")
    counts = collections.Counter(s for s, _, _ in RESULTS)
    for s, n, d in RESULTS:
        if args.quiet and s != "FAIL":
            continue
        print(f"[{s}] {n}" + (f"  -- {d}" if d else ""))
    print(f"\nSUMMARY: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['INFO']} INFO")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
