# Paper A: final publication-integrity audit

Audited artifact: `paper/TMLR_Journal_Submission_V3.pdf` (23 pages, SHA-256 `4b71b43c77ff27c3…`), plus the generated tables, figure CSVs/PDFs and frozen run records it depends on.
Audit date: 2026-09-20, repo HEAD `81d1dd1` + working tree.
Scope rule followed: no experiment was run, no model retrained, nothing tuned, **no result record was modified**. Only mechanical LaTeX/layout edits to table fragments were made (section 5).

> Internal document. It is not part of the anonymous submission; do not include it in the supplementary bundle.

## 1. Bottom line

**The submission is not yet publication-clean, and one requested step could not be performed.**

* **BLOCKED: the clean LaTeX build.** The manuscript source (`main.tex`, `.bib`, style file, `.log`) is not in this repo, not in any git branch or history, and not anywhere on this machine. The only copy is in Overleaf, which I cannot reach. I therefore could **not** build the final PDF, could not confirm "zero undefined references/citations" from a compile log, and could not check duplicate `\label`s or hard-coded references at source level. I did not fabricate a source. PDF-level proxies for every one of these checks are reported below, and the whole audit is re-runnable on the next build (section 7).
* **Scientific numbers are sound except two.** Every number I could tie to a frozen record matches, with two exceptions that need an author decision: Table 1 (one cell off by 0.001 from double rounding) and one sentence in Section 7.3 that puts a conflict value in the wrong missingness stratum.
* **The BVU renaming is incomplete in the submitted PDF.** `CellV0.3` still appears in the Figure 2 legend and in the Figure 3 legend and panel title. The regenerated figure files in `paper/figures/` are already correct; the PDF simply embeds an older build of the figures.
* **De-anonymization: the PDF itself is clean.** The supplementary material is not: four run logs carry an absolute `/Users/<user>/…` path, the account name and the git author name.

Automated tally from `paper/audit/run_final_audit.py` on V3: **71 PASS, 8 FAIL, 25 INFO**. All 8 FAILs map to items D1–D3, D5, D7 and D9 below.

## 2. Requirement-by-requirement status

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Numerical claims in abstract, intro, results, discussion, figures, Tables 1–7 match frozen records | **Partial: 2 discrepancies** | Every abstract/intro/results/discussion number recomputed from processed records. Mismatches: D2 (Table 1), D3 (Sec. 7.3). Everything else matches, including Table 8. |
| 2 | Every mean/std uses the aggregation the experiment documents | **Pass** | Sample SD (ddof=1) over seeds 0/1/2 everywhere; corruption replicas averaged within model/seed/severity first; AUC is the trapezoid over the severity grid; paired Δ computed per seed then averaged. Recomputed stored `corruption_auc` for all 114 Phase 2/3 runs exactly. Wording gaps in D12. |
| 3 | Every figure value matches its companion CSV | **Pass** | Vector geometry of Figures 2 and 3 *as embedded in V3* was inverted through the drawn tick marks and compared with the CSVs: 16/16 curves and all ±1 SD bands (Fig. 2), 6 curves and 30 error bars (Fig. 3), max error ≤ 3e-9. CSVs themselves match a from-scratch recomputation from the frozen records (88 + 30 rows, max error 1e-16). Same result for the repo figure PDFs. |
| 4 | No CellV0.1/0.2/0.3 display names; paper name is BVU | **FAIL** | D1: `CellV0.3` inside the embedded Fig. 2 legend and Fig. 3 legend/title (pages 11, 13). Body text uses BVU throughout. Repo figure PDFs and table fragments use BVU only. |
| 5 | All `\ref`, `\cite`, `\label`, figure/table/section/appendix references resolve | **Pass (PDF level only)** | 58/58 internal link destinations resolve; 16/16 bibliography entries cited and linked, none dangling; no `??` or `(?)` in text. Source-level check not possible. |
| 6 | No duplicate LaTeX labels | **NOT CHECKED at source** | Not detectable from a PDF (hyperref anchors are counter-based). The three fragment files define three distinct labels. |
| 7 | No hard-coded references where `\ref` belongs | **FAIL (5 found)** | D5: four `Proof of Proposition N` headings and `Figure~3` in the Table 8 caption are typed text with no link. All other 13 reference-like mentions are real links. |
| 8 | No overfull boxes, clipped tables, undefined refs/citations in the final build | **Partial** | No build log available. Layout proxy: everything sits inside the 468 pt text block except Table 7, 2.4 pt too wide (D7, fixed in the fragment, needs a recompile). No undefined refs/citations visible in the PDF. |
| 9 | Submission PDF and supplement free of de-anonymizing information | **PDF pass, supplement FAIL** | PDF: no author/title metadata, no XMP, no embedded files, no URIs, no local paths, usernames, emails or git identity in raw or decompressed bytes; "Anonymous authors" banner present. Supplement: D9. |
| 10 | NeuMiss commit, splits, seeds, training settings, parameter counts in the appendix match the frozen configuration | **Pass (one wording issue)** | NeuMiss `7902b8d` = `7902b8dbe7114e8dc3010b5e8b132c35806a9d74` in `pyproject.toml` and in all 6 raw NeuMiss records. Tables 3–6 (every width, parameter count, effective batch size, optimizer, LR, weight decay, step cap, patience and split entry) match the raw run records. Wording issue D8. |
| 11 | Build the final PDF with a clean compile | **NOT DONE (blocked)** | See section 1 and section 6. |

## 3. Discrepancies

Severity: **High** = must be fixed before submission; **Medium** = a reviewer can reasonably object; **Low** = polish. "Not fixed" means it needs the Overleaf source or an author decision; nothing scientific was changed.

### High

**D1. Stale `CellV0.3` display name in Figures 2 and 3 of the submitted PDF.** Not fixed (figure files are embedded in the Overleaf project).
* Where: Fig. 2 legend (p. 11); Fig. 3 left-panel legend and right-panel title `CellV0.3 -- internal precision by missingness` (p. 13).
* Cause: V3 embeds figure builds from before the BVU relabel. The plotted data is identical: geometry matches the CSVs to 3e-9 in both the embedded and the repo files. The repo files carry the `BVU` label and contain no `CellV0` text.
* Action: in Overleaf, replace the Fig. 2 and Fig. 3 files with `paper/figures/fig2_controlled_corruption.pdf` and `paper/figures/fig3_aps_missingness_mechanism.pdf`. The bounding boxes differ by under 1 pt because the legend label is shorter.

**D2. Table 1, MNIST / missing-feature: BVU corruption AUC printed as 0.666, exact value 0.66547, which rounds to 0.665.** Not fixed (scientific value, author decision; the table is not generated by any script in the repo).
* Evidence: per-seed AUCs 0.666565, 0.664867, 0.664973, mean 0.665468. `experiments/paper_a/publication_validation_results.md` prints 0.6655 (4 dp); rounding that again to 3 dp gives 0.666. The Δ column (+0.027) is correct, and with 0.665 the row is arithmetically consistent (0.665 − 0.638 = 0.027; with 0.666 it reads 0.028).
* The other 23 Table 1 cells match at 3 dp.
* Action: change `0.666` to `0.665`, and preferably generate Table 1 from the records like Tables 7 and 8.

**D3. Section 7.3 places the large-conflict value in the wrong stratum.** Not fixed (claim wording).
* Text: "In one APS seed, the mean conflict in the sparsest regime reached approximately 1.6×10³."
* Record: the value (layer-1 mean conflict, seed 1) is **1578.4 in the (0.25, 0.50] stratum**. In the >50% stratum the layer-1 mean conflict is 51.8 / 54.0 / 48.5 across the three seeds. The Phase 3 report lists the same thing (`aps/seed1 bin '(0.25, 0.50]' … ~1578`).
* Suggested wording: "…in the (0.25, 0.50] missingness stratum (layer 1, one seed), while the >50% stratum peaked near 54."

### Medium

**D4. The "seven of eight stable comparisons" count (abstract, Sec. 1, Sec. 6.2) is arithmetically right but its denominator is under-explained.** Not fixed (claim wording).
* The eight cells are 4 datasets × 2 corruptions **including California Housing**; 2 of the 7 wins are California Housing, yet Sec. 6.2 says California Housing is not used as a headline result. "Stable" refers to the Confidence-MLP comparator being stable there (the Plain and Reliability-gated MLPs diverge on some seeds); the paper never says so.
* Digits/Gaussian (+0.0094, positive on 2 of 3 seeds) is counted as a win while the text calls it "approximately tied".
* BVU wins on every seed in 6 of the 8 cells. Excluding California Housing the count is 5 of 6.
* Suggested: state the denominator (4 datasets × 2 corruptions), or report "5 of 6 excluding California Housing; every seed in 6 of 8".

**D5. Hard-coded references (typed, no link).** Not fixed (label names live in the Overleaf source).
* Appendix headings A.1–A.4 ("Proof of Proposition 1/2/3/4", pp. 17–18).
* `Figure~3` in the Table 8 caption, `paper/tables/aps_strata_counts.tex` line 7 (p. 23).
* Action: replace with `\ref{…}` to the proposition and figure labels. I did not guess the label names: a wrong guess would turn into `??`.

**D6. Repo fragment and compiled PDF disagree about table labels.**
* In V3, "Tables 3, 4, 5, and 6" (p. 20) are four working links. `paper/tables/reproducibility_settings.tex` defines only one label (`tab:reproducibility_settings`, Table 3). Tables 4–6 have no label in the repo copy.
* So the Overleaf copy of this fragment differs from the repo copy. Rebuilding from the repo would leave Tables 4–6 unreferenceable.
* Action: copy the Overleaf version of the fragment (or its label names) back into the repo, then commit.

**D7. Table 7 is 2.4 pt wider than the text block (overfull `\hbox`).** **Fixed in the fragment** (needs a recompile to confirm; V3 still has it).
* The generated tabular measured 470.4 pt in a 468.0 pt line. Column widths in `make_clean_performance_table.py` changed from 0.17 to 0.16 of `\linewidth` for the three numeric columns; the regenerated table is 456.5 pt wide. No cell content changed (CSV byte-identical, the `.tex` diff is the one column-spec line). My width model (sum of column widths + 8 pt per inner gap) reproduces the measured widths of Tables 3–6 to 0.2 pt.

**D8. Table 6 footnote says "fixed dataset splits"; Appendix C.2 says the APS train/validation split is made "for each seed".** Not fixed (methodology wording).
* Code: the APS 80/20 split uses `train_test_split(..., random_state=seed)`. Only the official 16,000-example test set is fixed; Air Quality is chronological and seed-independent.
* Suggested footnote: "official test set fixed; APS train/validation split seeded per seed; Air Quality chronological".

**D9. Supplement files carry identifying information.** Not fixed (they are frozen result-adjacent files).
* Four processed logs contain a line `Wrote … to /Users/<user>/projects/CGRN/…`, i.e. an absolute path with the account name, and the git author name: `experiments/paper_a/results/processed/cellv03_run.log`, `…/reliability/results/processed/run.log`, `…/reliability/results/processed/capacity_stress.log`, `…/real_reliability/results/processed/run.log`.
* The git history of `paper/` carries a real author name and email (7 commits), so a repo or `.git` must never be part of the supplement.
* The internal project codename `CGRN` appears in the logs, `datasets.py`, `paper/tables/aps_strata_counts_sources.md`. Non-identifying, but consider removing it.
* Raw run records (734 files), processed JSONs, figures, tables: no hits.
* Action: ship a clean export that excludes `*.log`, `.git`, `.DS_Store`, `.Rhistory`, `__pycache__`, or sanitize a *copy* of the logs.

**D10. "Same-width Confidence MLP" is not a stronger-capacity control on the real datasets.** Not fixed (claim wording).
* Section 5.1 says it is used "even when this gives the conventional model more parameters" and the introduction lists "stronger capacity controls". On MNIST/Fashion-MNIST it has 1.65× the parameters. On Digits (0.69×), APS (82,081 vs 99,601 = 0.82×) and Air Quality (3,907 vs 49,911 = 0.08×) it is **smaller** than BVU. Only the Digits case is discussed in the text.
* Suggested: state the parameter ratio next to the Same-width row of Table 2, and restrict "stronger capacity control" to the image datasets.

**D11. Early-stopping non-termination is not disclosed.** Not fixed (disclosure).
* Runs that hit the step cap instead of early-stopping: BVU APS seeds 0 and 2; BVU California Housing/Gaussian seed 1; Air Quality Confidence MLP seed 2, Same-width seed 1, NeuMiss seeds 1 and 2 (recorded in the Phase 2/3 reports, absent from the paper). Table 6's "7,000-step cap" invites the reading that all runs converged.
* Suggested: one sentence in Appendix C.5 listing the affected runs.

### Low

* **D12. Statistic wording.** Table 2 and the figure captions say "standard deviation"; Table 7 says "sample SD". The computed value is the sample SD (ddof=1) throughout; the population SD would change the third decimal in 11 of 12 Table 2 cells, so say so once. Table 2 bolds two values in the APS column (BVU 0.844 and HGB 0.876) without defining bold. The HGB APS SD prints as 0.000 (true 0.0005). The Table 1 caption does not say that AUC is the unnormalised trapezoid over the severity grid (maximum 0.7 for missingness, 1.5 for Gaussian).
* **D13. Table 6 LR cells** wrapped with a dangling "×" (`10⁻³, 3 ×` / `10⁻³, 10⁻²`). **Fixed in the fragment**: `3\!\times\!10^{-3}` is now grouped in braces so it cannot break. Cosmetic; recompile to confirm.
* **D14. Typography/layout.** Table 1 prints `-0.008` with a hyphen while Table 7 uses a true minus (use `$-0.008$`). Table 2 is placed before Table 1 on p. 12 although Table 1 is cited first (p. 11); pp. 11 and 21–22 have large float gaps.
* **D15. Duplicate sentence.** "We first evaluate seven standard classification and regression datasets…" appears in Section 5 and again in 5.2, with the same citations.
* **D16. Precision claim for Air Quality (interpretation).** The ">50% missing" Air Quality group has all 8 inputs missing, so every input has reliability 10⁻³ and Proposition 1 forces π ≤ e = 10⁻³. The reported drop "0.41 → 0.001" (0.4123 → 0.00100 in the records) is therefore imposed by the input, not evidence of a learned response. The APS and controlled-corruption precision results do not have this property.
* **D17. Small-stratum headline.** The abstract's "0.848 versus 0.702" is the >50% APS stratum: 114 test examples, 5 positives (Table 8). Per seed BVU beats NeuMiss every time (0.927/0.742/0.877 vs 0.710/0.710/0.686) but BVU's SD is 0.096. Consider adding "(114 examples, 5 positive)" where the number is first quoted.
* **D18. Frozen records are not under version control.** `.gitignore` excludes `results/raw/*` and `results/processed/*`; 0 record files are tracked. Everything above was verified against the copies on this machine. SHA-256 anchors are in section 7. Back them up and preferably archive them (Zenodo/OSF) before release.

## 4. What was verified (no action needed)

* Abstract, all numbers: 0.844 vs 0.838 (APS PR-AUC), 0.848 vs 0.702 (>50% stratum), R² 0.760 vs 0.743, shuffle −0.057, 1.65× parameters, "seven of eight" (count correct, see D4).
* Introduction and Sections 6.1–6.6: all values, per-seed claims ("every random seed" on MNIST/Fashion-MNIST, both corruptions, vs Confidence MLP and vs Same-width), the |Δ| < 0.01 claim for Breast Cancer/Wine/Digits, APS all-ones −0.0056 / shuffled +0.0571, Air Quality all-ones +0.0123 and shuffled = no-op, precision falling monotonically over the four incomplete APS strata on every seed and both layers, and mean hidden precision falling with corruption severity in 8/8 cells × 2 layers.
* Table 1 (except D2), Table 2 (all 12 cells, sample SD), Table 7 (all 42 printed cells, PDF text vs recomputation, plus `clean_performance_data.csv` and `figS1` CSV), Table 8 (5 strata + totals, identical across seeds), Tables 3, 4, 5, 6.
* Figures 2 and 3 (see requirement 3). Figure S1 is not in the PDF; its CSV matches the records.
* Appendix formula: the layer parameter count `mn + 2m` is consistent with the recorded totals (MNIST BVU: 784·157+2·157 + 157·157+2·157 + a 157→10 linear readout 157·10+10 = 149,945, exactly the recorded count).

## 5. Mechanical fixes applied in this pass

| File | Change | Why it cannot alter science |
|---|---|---|
| `paper/tables/make_clean_performance_table.py`, `paper/tables/clean_performance.tex` | numeric column widths 0.17 → 0.16 of `\linewidth` (D7) | layout only; regenerated `.tex` differs by that one line; CSV byte-identical |
| `paper/tables/reproducibility_settings.tex` | `3\!\times\!10^{-3}` wrapped in `{…}` in two LR cells (D13) | typesetting only; identical rendered symbols |
| `paper/tables/reproducibility_settings.tex` (already pending in the working tree) | `CellV0.1/0.2/0.3` → `BVU`, state-count rows removed | display-name change made earlier; HEAD still had the old names, so committing it closes that gap |

Not compile-verified (no TeX toolchain, no manuscript). I verified brace/environment balance and computed the resulting widths only.

## 6. Not checked / blocked

* **Compile-level checks** (need the Overleaf project): final PDF build, `.log` overfull/underfull/undefined warnings, duplicate `\label` detection, source-level hard-coded references, `\cite` keys vs `.bib`.
* **Reference metadata** (authors, venues, DOIs, years of the 16 entries) was not verified externally.
* Per-run `_history.json` training curves were not audited beyond file counts (228 + 66 raw record files consistent with the processed exports).
* Figure PNG rasters were not compared pixel by pixel; the vector PDFs (which the manuscript embeds) were.

## 7. Reproduce and re-run

```bash
python paper/audit/run_final_audit.py                                # audits V3
python paper/audit/run_final_audit.py --pdf paper/<next_build>.pdf   # after fixing Overleaf
```
Needs `pypdf` and `pdfminer.six`; read-only; repo-relative paths only. Section coverage: numbers, appendix configs, figure geometry, PDF link/citation/hard-coded-reference/layout/anonymity scans, supplement scan.

Frozen processed records used (SHA-256, taken 2026-09-20; the records are untracked):

| File (under `experiments/paper_a/`) | SHA-256 |
|---|---|
| `reliability/results/processed/reliability_runs.json` | `dc6b6f5efdad6778a063f8555a6c4888def90382ed764467ef63a82dd370ae20` |
| `reliability/results/processed/capacity_stress_runs.json` | `0c4320c1acc72aa563e3fefb5bdef49fe0a36b2fa32e338d9c813f474df39fc2` |
| `real_reliability/results/processed/real_reliability_runs.json` | `90ef508a32d239fda905a8733b6890ba311948d1b3d97ea46032320003ed067c` |
| `results/processed/phase1_cellv03_runs.json` | `c48009bc90f2bc576c4d0cbc93df383c8adb480b4d148b7532cf1c7c2eefadd0` |
| `results/processed/phase1_runs_small.json` | `bba95fcb2a5b7ce48a6ad854ba5fe0100af889252dfec0c71eefa8162f0708ed` |
| `results/processed/phase1_runs_mid.json` | `83c6e1ffc26f1aeb5f1436008d0b3cd1c09fda87a7e7a1f789e3baf2b99ffac5` |
| `results/processed/phase1_runs_mnist_0.25.json` | `f6dd9f6cdee31f2388b2cacb3ed5362e2da851ee291fa16137c5ceb72d8830aa` |
| `results/processed/phase1_runs_mnist_1.0.json` | `c2c9a61d72769ee556f491274a55feedf2096ad1dc3aaf70f94a20a141ec279d` |
| `results/processed/phase1_runs_fashion_mnist_0.25.json` | `c9e112eb98a4b0d4d5cc5b3044e3305629d78b921f20dfa2dd7d0aa6b176617a` |
| `results/processed/phase1_runs_fashion_mnist_1.0.json` | `de6ca16d92ed3e8b635d85d39b348b7a3fe46bbb3cc3c2156065d166e4a55898` |

## 8. Checklist to reach a clean submission

1. Overleaf: swap in `paper/figures/fig2_controlled_corruption.pdf` and `fig3_aps_missingness_mechanism.pdf` (D1).
2. Overleaf: Table 1 `0.666` → `0.665` (D2); fix the Section 7.3 sentence (D3); decide on D4, D8, D10, D11 wording.
3. Overleaf: replace typed references with `\ref` (D5) and paste in the updated `clean_performance.tex` and `reproducibility_settings.tex` (D7, D13); copy the Overleaf label names for Tables 4–6 back into the repo (D6).
4. Rebuild in Overleaf, check the log for zero undefined references/citations and zero overfull boxes, download as V4.
5. `python paper/audit/run_final_audit.py --pdf paper/<V4>.pdf` should then show only the D9/D18 items (supplement hygiene) as open.
6. Build the supplement from a clean export (no `.git`, `*.log`, hidden files, or the codename).
