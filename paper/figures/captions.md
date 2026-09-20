# Paper A -- proposed figure captions

These start from the task-provided draft captions but are edited wherever
the regenerated data does not exactly support the original wording (see
inline notes). Every number below is read from
`fig2_controlled_corruption_data.csv` / `fig3_aps_missingness_mechanism_data.csv`,
not retyped from memory.

## Figure 2 -- controlled corruption robustness

> **Figure 2.** Robustness under controlled heterogeneous input
> reliability. Test accuracy (mean over 3 seeds; shaded regions denote
> ±1 sample standard deviation) is shown on MNIST and Fashion-MNIST as
> missing-feature probability (left column) or maximum heterogeneous
> Gaussian noise scale $s$ (right column) increases. BVU is compared
> with a plain MLP, a parameter-matched Confidence MLP that receives the
> same reliability metadata, and a Same-width Confidence MLP (Phase 3
> capacity-stress control) that is allowed additional parameters at
> BVU's exact hidden width. Dashed vertical lines mark the largest
> corruption severity seen during training ($p{=}0.3$, $s{=}0.75$); the
> region to the right ("OOD") is severity-extrapolation, not seen in
> training. BVU is ahead of all three baselines at every plotted
> severity, on both datasets, under both corruption families, including
> outside the training range.

_No change in substance from the task's draft caption; reworded slightly
for precision (the original said "shows mean performance" -- averaging is
over seeds only, corruption replicas are already averaged within each
seed before this)._

## Figure 3 -- real missingness and mechanism (APS Failure at Scania Trucks)

> **Figure 3.** Performance and internal computational precision as
> naturally occurring missingness increases on APS Failure at Scania
> Trucks. Left: PR-AUC (mean over 3 seeds ± 1 sample standard deviation)
> stratified by the fraction of missing input features, for BVU,
> NeuMiss, a parameter-matched Confidence MLP, and a plain MLP. Right:
> BVU's own mean hidden precision $\pi = e / (1 + e \cdot u)$ over the
> same missingness strata, shown separately for layer 1 and layer 2. The
> four models track each other closely through the three lowest-
> missingness strata; BVU separates from the others in the two
> highest-missingness strata and most clearly in the $>$50%-missing
> stratum (PR-AUC 0.848 vs. 0.702 for the next-best model, NeuMiss).
> Internal precision is **not** monotonic in missingness: it rises from
> the fully-observed stratum to the 0-10%-missing stratum, then falls
> monotonically as missingness increases further, reaching its minimum in
> the $>$50% stratum. No trend line is fit.

_Changed from the task's draft caption:_ the draft states internal
precision "decreases with available information," implying a monotonic
fall across the whole missingness axis. The regenerated data
(`fig3_aps_missingness_mechanism_data.csv`, `panel=precision`) shows a
rise from the `0` bin (L1 $\pi\approx0.40$, L2 $\pi\approx0.36$) to the
`(0, 0.10]` bin (L1 $\pi\approx0.84$, L2 $\pi\approx0.76$), and only then a
monotonic fall through the remaining three bins to the `>0.50` minimum (L1
$\pi\approx0.21$, L2 $\pi\approx0.19$). The caption above reports that
shape rather than the simpler monotonic claim, per the task instruction
not to assert what the data does not show.

## Figure S1 (supplementary, not for the main paper) -- clean 7-dataset benchmark

> **Figure S1.** BVU vs. a parameter-matched MLP on the frozen
> Phase-1 clean (uncorrupted) 7-dataset screen, full training fraction.
> Points show the mean paired per-seed difference in each dataset's
> headline metric (accuracy or $R^2$, annotated per row); error bars are
> ±1 sample standard deviation over 3 seeds. BVU is at or slightly
> behind the MLP on the three small tabular tasks (Breast Cancer, Wine,
> Digits; all within noise) and ahead on Diabetes, California Housing,
> MNIST, and Fashion-MNIST.
