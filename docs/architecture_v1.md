# Architecture V1 — Dynamic Belief Graph (specified and implemented)

**Status: SPECIFIED AND IMPLEMENTED.** The user fully specified CellV1's
math across two conversation turns (2026-09-01 sketch, 2026-09-02 full
formulas) and explicitly asked for it to be implemented. It is, in
`src/models/architecture_v1/` — see §7 for the module map. Two pieces of
glue code any implementation needs regardless of the belief mechanism
(input encoding, output decoding) were never pinned down by the user and
are documented in §6 as implementation-choice defaults, not specified
formulas. Nothing here supersedes or blocks CellV0/`docs/architecture_v0.md`
as an experimental line — Experiments 002–004 stand on their own.

**CellV1.1 (§11, same day):** dense CellV1's `O(n_cells^2)` routing is a
scaling dead end, so the same cell/update math now also has an
LSH-based sparse routing variant (`SparseDynamicBeliefGraph`), validated
numerically identical to dense at small scale and empirically faster with
a widening margin at larger `n_cells` (2.15x at 128, 12.00x at 512).
Dense stays in the repository, unmodified, as "CellV1 Dense Reference."

**CellV1.2 (§12, same day):** `v1_001_dynamic_groups`'s first real run
showed CellV1 (dense *and* sparse) flatlining near `R^2=0` on both train
and test while `mlp`/`cellv0.1` learned fine at matched parameter counts
— an isolation diagnostic (dense reference vs. sparse) narrowed this to
the recurrent state-write itself, not routing. Every cell was being fully
overwritten by the fused proposal every step, with no mechanism to
resist it — diagnosed as a state-collapse/over-mixing failure (the
recurrent/message-passing analogue of GNN oversmoothing). Fixed with a
learned per-channel write gate: `next = (1 - beta) * old + beta *
proposal`.

**CellV1.3 (§13, next day):** CellV1.1's discrete LSH candidate search
replaced with a continuous, learned density field (differentiable
mean-shift, self-anchored orthogonal random Fourier features) — local
structure emerges as density modes, no `num_groups`/`K_local`
hyperparameter anywhere. Local-only, one step (`T=1`); global field
communication designed but deferred.

**CellV1.3.1 (§14, same day):** The deferred global communication added
on top of CellV1.3's local field, reusing dense CellV1's send/need/
query-key design (§5 Part III) via exact linear attention (Katharopoulos
et al. 2020), plus a new task (`dynamic_groups_global`) built specifically
to need it. A real, scale-dependent instability was found and partly
chased down — one seed of the *local-only* field still collapses at the
largest tested scale even after the fix that resolved an earlier,
different collapse in the global pathway. Flagged open, not resolved.

**CellV1.4 (§15, same day):** The ORFF-approximated local kernel replaced
with an exact, learned one (`phi_i = softplus(F_assoc(...))`,
`K_ij = phi_i . phi_j`) — no routing coordinate, no bandwidth/mass
functions, no random-feature convergence question. Matches CellV0.1's
accuracy across a 4-level complexity sweep with substantially less seed
variance than either field variant, and no collapse on any of 12
(level, seed) combinations tested.

**CellV1.5 (§16, PROPOSED, NOT FINALIZED, same day):** A persistent,
sparse, slowly-learned structural graph `w_ij` gating CellV1.4's dynamic
kernel — `c_ij(t) = w_ij * a_ij(t)`. Motivated by cell-assembly theory
rather than "rebuild the graph every step." Captured as a proposal with
several open architecture questions listed (§16), not implemented.

## 0. Relationship to Architecture V0

`docs/architecture_v0.md` froze the "what is one cell" question in favor
of a single belief state `(mu, e, u)` (§1) with five candidate aggregation
operators, run through a **fixed** stack of `BeliefLayer`s (currently two).
Its §3 ("Graph / cluster organization") and §4 ("Fast dynamic association")
sketched — only abstractly, as `[OPEN]` — that circuits/clusters and a
temporary `A_ij(t)`-style association might matter later.

This document replaces the fixed layer/cluster structure entirely with a
single persistent, input-dependent, self-organizing cell population — it
is the concrete mechanism V0 §3/§4 only sketched.

**Literature context (from the user's 2026-09-02 search):** no single prior
architecture is "this idea already done," but several pieces are close to
parts of it — DGCNN (dynamic graph recomputed from feature similarity,
arXiv:1801.07829), Routing Transformer (content-based clustering for
sparse routing, arXiv:2003.05997), Slot Attention (iterative competitive
grouping, NeurIPS 2020), capsule routing-by-agreement (NeurIPS 2017),
Growing Neural Gas (topology itself learned/adapted, NeurIPS 1994), RIMs
(sparsely communicating recurrent modules, arXiv:1909.10893), and Global
Workspace models (bandwidth-limited competition to broadcast, arXiv:2103.01197).
The combination this document specifies — persistent belief-bearing cells
+ input-dependent sparse local topology + emergent variable communities +
dynamically selected long-range communication + iterative graph
reorganization, all over cells that carry evidence/uncertainty as first-class
state rather than an undifferentiated content vector — was not found
already assembled in one place in that search, though the search was not
exhaustive enough to claim novelty. See §8 for the explicit contrast with a
Transformer.

## 1. Core hypothesis

> Computation can emerge from an input-dependent graph of persistent
> belief-bearing cells, where local functional assemblies self-organize
> according to state similarity, and salient cells dynamically mediate
> long-range communication.

Contrast with everything tested in Experiments 002–004: those use a fixed
architecture — the same two `BeliefLayer`s, the same connectivity, for
every input. Here the *wiring itself* is a function of the current input
and the cells' current states, recomputed every refinement step. There is
no permanently designated compartment, cluster, or "global" cell, and no
fixed count of either — "compartments" are just dense regions of a graph
that reforms as beliefs change.

## 2. What changes relative to CellV0

- **No layers.** A single persistent pool of `N` `BeliefCell`s (default
  `n_cells=128`), not a stack.
- **No fixed compartments/clusters, and no permanent per-pair connection
  parameters.** CellV0's `BeliefLayer` learns a fixed `w_ij`/`g_ij` for
  every pair, forever. CellV1 has no equivalent — every edge is
  *recomputed* from the two cells' current states at every step; the same
  pair can be strongly connected on one input/step and disconnected on the
  next.
- **Repeated in-place refinement.** The same cells persist and update over
  `T` refinement steps (default `num_steps=6`) via one *shared* step
  module (`DynamicBeliefGraphStep`), not `T` independently-parameterized
  layers.
- **Association becomes part of cell state.** Argued in §4: `(mu, e, u)`
  alone can't support meaningful dynamic grouping. A fourth per-cell field,
  `z`, carries it.

## 3. CellV1 state

```text
Cell_i^t = (mu_i^t, e_i^t, u_i^t, z_i^t)

mu_i  -- content: what the cell currently believes             (unchanged from CellV0)
e_i   -- evidence: how much supporting information produced it (unchanged)
u_i   -- uncertainty: how (un)confident the cell is in mu_i     (unchanged)
z_i   -- association / key vector, R^d, d=8 by default: what
         this belief concerns / who this cell should talk to    (NEW)
```

`z` is not the content itself and is kept L2-normalized (unit
hypersphere) after every update. `src/models/architecture_v1/cell.py`'s
`BeliefCellV1`.

## 4. Why `(mu, e, u)` isn't enough for dynamic grouping

Two cells can have near-identical `mu` (e.g. `0.70` vs. `0.72`) while
representing completely unrelated things. "Similar `mu` implies same
concept" is too simplistic to drive association scoring — `z` exists to
carry "aboutness" separately from belief value/confidence.

## 5. The mechanism, per refinement step

`src/models/architecture_v1/dynamics.py`'s `DynamicBeliefGraphStep`, one
shared instance applied `T` times.

### Part I — Local association (`routing.py::LocalAssociation`)

Semantic distance between every pair `(i, j)`:

```text
D_ij = ||L(z_i - z_j)||^2 + lambda * (mu_i - mu_j)^2 / (u_i^2 + u_j^2 + eps)
```

`L` is one learned linear map, shared by every pair. Association score
`s_ij = -D_ij / tau`; self-loops masked out. `sparsemax` (§6, an
implementation choice for the user's "entmax") over `[s_i1, ..., s_iN,
s_i∅]` — the extra `∅` option, a learned scalar logit, lets a cell route
to nothing — produces a directed, exactly-sparse `a_ij` (many entries
exact zero, not just small). Made mutual:

```text
A^L_ij = sqrt(a_ij * a_ji)
```

so a cell isn't "locally associated" with another just because it alone
wants the connection. Dense regions of `A^L` are the emergent local
assemblies — there is no `cluster_id`.

### Part II — Local belief fusion (`fusion.py::precision_fusion`)

Message content `m_ij = F_msg(mu_i, e_i, u_i, mu_j, e_j, u_j)` — one small
shared MLP (`shared_functions.py::MessageFunction`), not a per-edge
transformation. Fused via the **same formula as Method E /
"CellV0.1"** (`src.models.architecture_v0.integration._scale_stable_precision_fusion`,
Experiment 004I) with `A^L` playing the role of the learned relevance gate
`g` and the effective-source-count normalizer `n_eff = (sum A)^2 /
(sum A^2 + eps)` handling a dynamically-sized, dynamically-chosen
neighbor set exactly the way it was built to handle a variable
`hidden_cells`. Produces `Local_i = (mu_i^L, e_i^L, u_i^L)`.

### Part III — Need / offer / global routing

```text
delta_i = |mu_i - mu_i^L| / sqrt(u_i^2 + (u_i^L)^2 + eps)
need_i  = sigmoid(F_need(e_i, u_i, delta_i, z_i))
offer_j = sigmoid(F_offer(e_j, u_j, z_j, local_centrality_j))
```

`local_centrality_j = sum_i A^L_ij` (how much local demand there
currently is for `j`). `F_need`/`F_offer` are small shared MLPs
(`shared_functions.py`). Global routing (`routing.py::GlobalRouting`) is
directional query/key compatibility, not symmetric distance — two
different concepts may need to interact despite being dissimilar:

```text
S^G_ij = (q_i . k_j) / sqrt(d) + log(need_i + eps) + log(offer_j + eps) - gamma * A^L_ij
```

`q = W_Q z`, `k = W_K z` (separate learned linear maps). The `-gamma *
A^L_ij` term steers global edges away from pairs already locally
connected. `sparsemax` with its own learned null option again makes this
sparse; `A^G` is **not** symmetrized (routing is directed). There are no
permanently designated "global" cells — which cell (if any) plays that
role is input- and step-dependent.

### Part III (continued) — Global belief fusion

Identical to Part II, run over `A^G` instead of `A^L`, using the same
`F_msg` (§"Then global information uses the SAME belief mathematics").
Produces `Global_i = (mu_i^G, e_i^G, u_i^G)`.

### Part IV — Fuse self + local + global

Three belief sources — `Self_i = (mu_i, e_i, u_i)`, `Local_i`, `Global_i`
— fused with the *same* `precision_fusion` rule again, over a learned
per-source gate (`sigmoid` of a 3-element parameter, shared across all
cells — the closest reading of "reuse the same rule," since Method E's
gate was never hardcoded to 1). Produces `(mu_i^{t+1}, e_i^{t+1},
u_i^{t+1})`.

### Part V — Semantic-address update

```text
z_bar_i^L = sum_j alpha_ij^L z_j      (alpha from Part II's fusion)
z_bar_i^G = sum_j alpha_ij^G z_j      (alpha from Part III's fusion)
z_i^{t+1} = normalize(z_i^t + eta * F_z(z_i^t, z_bar_i^L, z_bar_i^G, mu_i^{t+1}, e_i^{t+1}, u_i^{t+1}))
```

`F_z` is a small shared MLP; `normalize` is L2 (§6). Because beliefs and
addresses both change every step, association scores change too — local
groupings can merge, split, or reform between steps, not just once at the
start.

## 6. Implementation-choice defaults (not specified by the user)

Everything above came from the user. These two pieces are glue code any
implementation needs regardless of the belief-mechanism math, and were not
pinned down in either conversation — flagged here and in the relevant
module's docstring so they're easy to find and revisit:

- **Sparsification function.** The user proposed "alpha-entmax." Implemented
  as `sparsemax` (`sparsemax.py`) — the alpha=2 member of the same family,
  chosen for a closed form needing no extra dependency (`entmax` is not in
  `pyproject.toml`) and no bisection. entmax with alpha in `(1, 2)` is a
  possible refinement, not required by anything specified so far.
- **Encoding** (`encoder.py::PopulationEncoder`). How raw input features
  become the initial `N`-cell population: learned linear projections to
  `mu` and `z`, `e = u = 1` for every cell (mirroring CellV0's
  `from_observed_features` convention that every cell starts equally
  uncertain).
- **Decoding** (`decoder.py::PopulationDecoder`). How a variable,
  input-dependent population is read into a fixed-size prediction:
  precision-weighted pooling (`evidence / (uncertainty^2 + eps)`) of both
  `mu` and `z` across all `N` cells, then one linear head. No designated
  "output cells."
- **`z`'s normalization.** L2 (unit hypersphere), applied after every
  update including at encoding. Not specified by the user.
- **Hyperparameter defaults** (`n_cells=128`, `association_dim=8`,
  `num_steps=6`, `lambda_=1.0`, `tau=1.0`, `gamma=1.0`, `eta=0.1`) are the
  user's suggested orders of magnitude where given, otherwise a reasonable
  starting point — none have been tuned or validated by any experiment yet.

## 7. Implementation map

`src/models/architecture_v1/`:

| Module | Contents |
|---|---|
| `cell.py` | `BeliefCellV1` — the `(mu, e, u, z)` state container (§3) |
| `sparsemax.py` | `sparsemax`, `sparse_association_with_null` — §1's sparsification (§6 implementation choice) |
| `routing.py` | `LocalAssociation` (§5 Part I), `GlobalRouting` (§5 Part III) |
| `shared_functions.py` | `MessageFunction`, `NeedFunction`, `OfferFunction`, `SemanticUpdateFunction` — the small shared MLPs (`F_msg`, `F_need`, `F_offer`, `F_z`) |
| `fusion.py` | `precision_fusion` — Method E's formula, reproduced (not imported) to also return `alpha` (§5 Part V needs it) |
| `dynamics.py` | `DynamicBeliefGraphStep` (one refinement step), `DynamicBeliefGraphCore` (applies it `T` times) |
| `encoder.py` | `PopulationEncoder` (§6 implementation choice) |
| `decoder.py` | `PopulationDecoder` (§6 implementation choice) |
| `model.py` | `DynamicBeliefGraph` — encoder → core → decoder, standard `nn.Module` interface |

Tests: `tests/test_belief_cell_v1.py`, `tests/test_sparsemax.py`,
`tests/test_routing_v1.py`, `tests/test_fusion_v1.py`,
`tests/test_dynamics_v1.py`, `tests/test_model_v1.py` — shape/invariant/
gradient-finiteness checks (this is new, unvalidated architecture math;
"correctness" here means "matches the specified formulas and produces
valid finite output," not "produces good results" — that's an experiment
question, not yet run for CellV1).

**A bug caught during implementation, worth recording:** `sqrt(a_ij *
a_ji)` (§5 Part I's mutual local association) has an infinite gradient at
exactly `a_ij * a_ji == 0` — and `sparsemax` produces exact zeros *by
design*, routinely, so this poisoned every parameter's gradient with
`nan`/`inf` on the very first backward pass. Fixed with a shifted
gradient-safe sqrt (`routing.py::_safe_sqrt`, `sqrt(x + eps) - sqrt(eps)`)
that preserves exact-zero *values* (true sparsity, unlike `sqrt(x + eps)`
alone, which leaves a small nonzero floor everywhere) while keeping the
gradient finite everywhere. `tests/test_model_v1.py::test_full_forward_backward_has_finite_gradients`
is a regression test for this.

## 8. Why this isn't just a Transformer

| Transformer-ish architecture | CellV1 |
|---|---|
| Tokens move through layers | Same persistent cells repeatedly update |
| Dense/structured attention | Sparse graph emerges from cell state (`sparsemax`, exact zeros) |
| Usually one content vector per token | Explicit `mu` / `evidence` / `uncertainty` / `z` separation |
| Attention mostly decides content mixture | Routing (§5 Parts I/III) and belief-confidence fusion (Parts II/IV) are separate mechanisms |
| No emergent local "community" requirement | Mutual sparse links (`A^L`) form dynamic local assemblies |
| Global attention generally the same mechanism as local | Local (distance-based) and global (query/key-based) routing use different formulas |
| Layer-specific parameters | One step module, shared across all `T` iterations |
| Static depth | Recurrent self-reorganization; the graph is recomputed every step |

## 9. Staging relative to "isolation of variables"

Flagged, not resolved: this is a larger jump than any single CellV0
change so far (004F/004I each changed one normalization term) — it adds a
new state field *and* replaces fixed topology with a fully dynamic one,
*and* introduces several new learned functions (`F_msg`, `F_need`,
`F_offer`, `F_z`, `L`, `W_Q`, `W_K`) at once. No experiment has been run
yet. Before drawing any conclusion from a future CellV1 experiment, worth
deciding whether to isolate the "self-organizing local regions" hypothesis
from the "dynamic global-salience routing" hypothesis (e.g. a local-only
ablation, §5 Parts I-II only, no Part III) so a negative result stays
interpretable — this is an experiment-design decision for whenever CellV1
experiments are scoped, not resolved by implementing the mechanism itself.

## 10. Naming

**Dynamic Belief Graph** (alternatively "Self-Organizing Belief Network"),
per the user. The extended per-cell state is **CellV1**. Avoid
"compartmental network" naming for this direction specifically — unlike
the CellV0-era compartment framing that was explicitly rejected
(`docs/architecture_v0.md` §1), this has no fixed compartments at all; the
term would be actively misleading here.

## 11. CellV1.1 — sparse routing (`O(n_cells^2)` doesn't scale)

**Status: implemented, validated for correctness, not yet run as an
experiment.** Everything in §1-§10 above (`Cell = (mu, e, u, z)`, the
scale-stable-precision fusion, `T` recurrent shared-parameter updates,
local-vs-global communication, the shared update functions) is **unchanged**
— CellV1.1 only replaces *how a cell finds candidates worth scoring*.
CellV0.1's math stays byte-for-byte the same wherever it's reused.

**The problem:** §5's local/global association scores every one of the
other `n_cells - 1` cells before sparsifying — `O(n_cells^2)` per
refinement step. Fine at `n_cells=128`; a dead end for the population
sizes (thousands+) this architecture's hypothesis is ultimately about.

**The fix — LSH-based candidate pre-filtering, not masking.** A masked
`(n_cells, n_cells)` tensor would still cost `O(n_cells^2)` to build, even
if the *expensive* scoring were skipped for masked entries — no actual
saving. Instead (`src/models/architecture_v1/lsh.py`):

1. **Hash.** `H` fixed (not learned — gradients can't flow through a
   discrete bucket decision anyway) random-hyperplane projections turn
   each cell's routing vector into an integer bucket id
   (sign-random-projection LSH, `bucket_ids`) — cells with similar vectors
   land in the same bucket with high probability.
2. **Sort + `searchsorted`, not chunk-position.** Sort all cells by bucket
   id (`O(n_cells log n_cells)`). For each *query* cell, `torch.searchsorted`
   finds where its own bucket id would insert into that sorted array —
   `O(log n_cells)` per query, vectorized. This works even when the query
   and key vectors differ (global routing's `q` vs. `k`), unlike Reformer's
   original trick, which assumes query=key. A small window of the sorted
   order around that insertion point is the candidate pool
   (`lsh_candidates`) — no `(n_cells, n_cells)` tensor, ever.
3. **Exact scoring, restricted to the pool.** The *same* semantic-distance
   (local) / query-key (global) formulas from §5 Parts I/III run only on
   the gathered `O(pool_size)` candidates per cell
   (`sparse_routing.py::SparseLocalAssociation`/`SparseGlobalRouting`),
   producing a per-cell, per-candidate weight — `sparsemax` (with the same
   learned null option) sparsifies within *that* pool.
4. **Mutuality without a dense lookup.** Local association is still made
   mutual, `A^L_ij = sqrt(a_ij * a_ji)`, but `a_ji` is found via
   `lsh.gather_rows`: fetch candidate `j`'s *own* (small) candidate row and
   check whether `i` is in it, rather than indexing a dense matrix. If `j`
   never considered `i` a candidate at all — a real possibility with
   approximate search, unlike dense's guaranteed-symmetric `a_ji` — the
   reciprocal weight is `0`, not an error.
5. **Global's local-edge penalty, looked up the same way.** The `-gamma *
   A^L_ij` term (§5 Part III) is looked up against the *local* routing's
   candidate pool (`lsh.lookup_value`) instead of a dense `A^L`.

Every gather above is `O(n_cells * pool_size)`, `pool_size` a small fixed
constant (`num_hashes * (2*window+1) * chunk_size`) — genuinely linear in
`n_cells`, not quadratic. Complexity, end to end:
`O(T * (H * n_cells * log(n_cells) + n_cells * pool_size))`.

**Differentiability.** Bucket assignment and the sort/`searchsorted`
candidate selection are discrete and not differentiated through (same as
Reformer's LSH attention) — the hyperplanes are fixed buffers, not
parameters. Every learned parameter (the metric `L`, `W_Q`/`W_K`, the
shared MLPs) still receives correct gradients through the *values* at the
selected candidates, exactly like standard sparse/MoE-style routing.

**Correctness validation — sparse must equal dense, not just "look
similar."** `tests/test_sparse_dense_consistency.py`: with one hash round
and `chunk_size == n_cells` (the pool covers every cell — nothing is
actually filtered), `SparseDynamicBeliefGraphStep` and
`DynamicBeliefGraphStep` given identical weights produce **numerically
identical output up to float32 rounding** (`~6e-8` max absolute
difference on `mu`, exactly `0` on `evidence`/`uncertainty`/`z`), for both
the full and local-only ablation, and over multiple chained steps. This is
exactly what keeps `model.py`'s dense implementation in the repository —
"CellV1 Dense Reference" (§7's original modules, untouched) — good for
correctness checks at small `n_cells`, not for the real experiment.

**Empirical scaling, measured (not just asserted):**

| `n_cells` | dense ms/step | sparse ms/step | speedup |
|---|---|---|---|
| 128 | 351.0 | 163.5 | 2.15x |
| 256 | 1307.7 | 287.9 | 4.54x |
| 512 | 6903.3 | 575.2 | 12.00x |

(CPU, `batch=16`, `T=6`, `association_dim=8`, `hidden_dim=32`,
`chunk_size_local=10`, `chunk_size_global=4`.) Dense's cost roughly
quadruples per doubling of `n_cells`, matching `O(n_cells^2)`; sparse's
grows much more slowly, matching the fixed-pool-size design — the
speedup widens with scale exactly as intended, not a fixed constant
factor.

**Implementation map** (`src/models/architecture_v1/`), parallel to §7's
dense modules, none of which were modified:

| Module | Contents |
|---|---|
| `lsh.py` | `random_hyperplanes`, `bucket_ids`, `lsh_candidates`, `gather_scalar`/`gather_vector`/`gather_rows`, `lookup_value` — the indexing primitives, no `(n, n)` tensors |
| `sparse_routing.py` | `SparseLocalAssociation`, `SparseGlobalRouting` |
| `sparse_dynamics.py` | `SparseDynamicBeliefGraphStep`, `SparseDynamicBeliefGraphCore` |
| `sparse_model.py` | `SparseDynamicBeliefGraph` — same `nn.Module`/encoder-pluggable interface as dense `model.py::DynamicBeliefGraph` |

`shared_functions.py::MessageFunction` gained `forward_sparse` (same
learned weights, applied to gathered `(batch, n, pool)` pairs instead of
`(batch, n, n)`); nothing else in the dense-only modules changed.

Tests: `tests/test_lsh.py`, `tests/test_sparse_routing.py`,
`tests/test_sparse_dynamics.py`, `tests/test_sparse_model.py`,
`tests/test_sparse_dense_consistency.py` — 31 tests, all passing (full
suite 223/223).

**Update:** `hash`/`chunk_size`/`window` (`num_hashes_local=2`,
`bits_local=6`, `chunk_size_local=10`; `num_hashes_global=2`,
`bits_global=6`, `chunk_size_global=4`; all `window=0`) are reasonable
defaults matching the user's suggested orders of magnitude, not tuned.
`experiments/v1_001_dynamic_groups/`'s harness was later wired to
`SparseDynamicBeliefGraph` (§12 below covers what that first run found).

## 12. CellV1.2 — gated write update (state-collapse fix)

**Status: implemented, one confirming diagnostic run.** `v1_001_dynamic_groups`'s
first real run (`dynamic_groups`, V1-S0 = 128 cells, 2 seeds, 1500 steps)
found `cellv1_local`/`cellv1_full` (sparse) essentially flat —
`R^2 ~ 0.00-0.03` — while `mlp` (`R^2 ~ 0.61-0.68`) and `cellv0.1`
(`R^2 ~ 0.74-0.77`) learned fine at a matched parameter count. Since
`cellv1_full`'s parameter count was *equal to* the baselines', this
wasn't "V1 just has fewer parameters."

**Isolation diagnostic**
(`experiments/v1_001_dynamic_groups/diagnose_v1_learning.py`), per the
user's explicit fork (LSH routing destroying the signal? generalization
issue? optimization issue?): trained `dense` (CellV1 Dense Reference, §7 —
zero LSH involved) on the same task/scale/seed and recorded *train* R² as
well as test R². Result: `dense` train R² = `0.0164`, test R² = `0.0079`
— matching sparse's failure almost exactly, and failing on the training
set itself. This ruled out both hypotheses the sparse routing could have
explained (LSH destroying the signal; generalization/overfitting) and
pointed at the core recurrent update: `O(n_cells^2)` vs. `O(n_cells *
pool_size)` routing isn't the variable that matters here, and a model
that can't fit its own training data isn't overfitting.

**Diagnosis:** every cell was being *fully overwritten* by the self+local+
global fused proposal every step (§5 Part IV: `mu_next, e_next, u_next,
_ = precision_fusion(...)`, then no mechanism to resist that value), with
`z` the only channel that already had any residual/inertia (`z + eta *
z_delta`, a small fixed step). Six full-overwrite rounds of message-
passing + consensus-style fusion, over a *shared* update rule, is exactly
the recurrent/message-passing analogue of GNN oversmoothing — cells wash
toward similar states because nothing lets a cell partially refuse an
update.

**Fix — a learned write gate, applied uniformly to every channel**
(`shared_functions.py::WriteGateFunction`): the existing self+local+global
fusion now produces a *proposal* (`mu_hat`/`e_hat`/`u_hat`), and
`beta = sigmoid(F_gate(mu, e, u, mu_local, e_local, u_local[, mu_global,
e_global, u_global]))` — a per-channel, per-cell, per-example gate from
one shared trunk (`beta_mu`, `beta_e`, `beta_u`, `beta_z`) — decides how
much of it is accepted:

    next = (1 - beta) * old + beta * proposal

`z`'s fixed `eta` scalar is retired entirely; `beta_z` takes over the
identical residual-update role (`z + beta_z * z_delta`, same shape,
now learned and state-dependent instead of a constant). Gate bias
initialized to `-2.0` (`sigmoid(-2.0) ~= 0.12`, the user's suggested
0.1-0.2 range) so cells mostly preserve themselves early in training and
learn when communication is worth accepting.

Applied identically in `dynamics.py` (dense) and `sparse_dynamics.py`
(sparse) — this is a change to the shared recurrent-update math, not a
routing change, so both variants needed it. `tests/test_sparse_dense_consistency.py`
still passes (`write_gate`'s weights are now also copied between the
dense/sparse pair the test builds), confirming the two stay exact
restrictions of the same math after this change too.

**Confirming run:** `normal_sparse` (the harness's real LSH config, not
the full-pool diagnostic config), same task/scale/seed, 500 steps — train
R² moved from `0.0164` (ungated `dense`) to **`0.7193`**, test R² from
`0.0079` to `0.7299` — within the user's "0.5-0.8" bar for "the bottleneck
is found," and test tracking train closely (no new overfitting gap opened
by the fix), roughly matching `cellv0.1`'s level (`0.74-0.77`) from the
original run. The gated write is confirmed as the fix, not routing/pool
size, not initialization elsewhere. Next: resume the `V1-S1`/`V1-S2` scale
sweep with the gate in place.

## 13. CellV1.3 — Self-Organizing Refinement Field

**Status: implemented (`src/models/architecture_v1/field_*.py`,
`random_features.py`), unit-tested, one training smoke test. No
experiment run.** Replaces CellV1.1's discrete LSH-routed graph with a
continuous, learned density field: cells no longer search for a
candidate pool of neighbors (§11) — every cell exerts a soft, kernel-
weighted "attraction" on every other, and local structure emerges as
density modes, with no `num_groups`/`K_local` hyperparameter anywhere.
`(mu, e, u, z)` and the scale-stable-precision fusion formula
(`fusion.py`) are **unchanged**; only how a cell's neighborhood is found
and weighted changes.

### The role split

- **`mu`/`e`/`u` (belief content)** — still a precision-weighted kernel
  average, exactly `fusion.py`'s Method E formula, just with the
  per-pair weight `a_ij = mass_j * K(r_i, r_j)` instead of a learned
  dense/sparsemax weight.
- **`r` (routing position)** — new: `FieldCellState.r`, persistent across
  steps like `z`. Where a cell currently sits in the self-organizing
  field. Moves via a **gated mean-shift step**: `field_fusion.py`
  computes the density-weighted local mean `r_bar` (the classic
  mean-shift target — for a Gaussian kernel, moving toward it is
  gradient ascent on the local log-density); `field_dynamics.py` applies
  `r <- r + beta_r * (r_bar - r)`, `beta_r` a learned per-cell gate
  (`RoutingGateFunction`), not a fixed step size.
- **`z` (semantic identity)** — **not** reconstructed as a kernel-weighted
  average (that was the original design's actual bug, not a numerical
  one — see below). Updated by the same learned, gated mechanism the
  rest of CellV1 uses (`shared_functions.py::WriteGateFunction`'s
  `beta_z` output, `field_functions.py::FieldSemanticUpdateFunction`),
  fed `(r, r_bar)` as its "what's around me" signal instead of a
  z-specific density average.
- **`h`/`m` (bandwidth, mass)** — learned per-cell, per-step
  (`BandwidthFunction`, `MassFunction`): how broad a cell's neighborhood
  currently is, and how much it weighs in the field, both state-
  dependent rather than fixed hyperparameters.

### The debugging path (full detail in `docs/research_log.md`'s CellV1.3
entries) — kept here because each step changed the actual math, not just
tuned a constant:

1. **Wrong kernel.** First attempt used Performer/FAVOR+-style positive
   random features (`exp(x@W - 0.5||x||^2)`, approximating `exp(x^Ty)`,
   the *softmax* kernel) for what is a *Gaussian*-kernel local-density
   field. These are different kernels; matching values at cherry-picked
   inputs doesn't make the feature maps interchangeable. Fixed by
   switching to standard Random Fourier Features (Rahimi & Recht, 2007;
   `random_features.py::RandomFourierFeatures`), the textbook map for
   `K(x,y) = exp(-||x-y||^2/(2 sigma^2))`.
2. **Per-row stabilizer.** An earlier numerical-overflow guard computed
   per-cell, not per-batch-item — an uncancelled distortion, not just
   noise (silently corrupted every downstream sum; caught because the
   error *grew*, not shrank, with more features). Moot once features
   moved to `cos()` (bounded, no stabilizer needed at all).
3. **`n_eff` was secretly `O(n * R^2)`**, not `O(n * R)` — a `(R, R)`
   second-moment matrix for the effective-source-count denominator.
   Fixed: `K(x,y)^2` is *itself* a Gaussian kernel (bandwidth
   `sigma/sqrt(2)`), so `sum_j(weight_j^2 * K_ij^2)` is a second
   *linear* reduction via a second `RandomFourierFeatures(sigma=.../sqrt(2))`
   instance, not a quadratic form.
4. **Signed-kernel division blowups.** Unlike sparsemax-derived weights
   (always >= 0), `RandomFourierFeatures`' `cos`-based estimate of
   `K_ij` can be noisy-negative for a given pair. Sums built from these
   (`a_sum`, `p_sum`) could land at/near zero, and dividing by them blew
   up by orders of magnitude — worse with orthogonal features
   (variance-reduced, still not blowup-proof) than plain i.i.d. ones.
5. **Self-anchored density (the actual fix).** `K(i,i) = 1` *exactly*,
   always — never something to estimate. `field_fusion.py::local_field_fusion`
   now subtracts the RF's own noisy self-estimate from every raw
   reduction and replaces it with the exact value, clamps only the
   remaining *off-cell* contribution to non-negative (mass-like terms
   only — signed content like `s_mu`/`s_mr` gets self-anchored but not
   clamped), and sets `a_sum = weight_i + max(0, off_cell)`. This
   guarantees `a_sum`/`p_sum`/`a_sq_sum` are each `>= their exact self
   term > 0` — structurally impossible to blow up, not a bigger epsilon.
   Confirmed: max errors across outputs dropped from the *millions* to
   single digits at the same `R`.
6. **Mean-shift as displacement, not `r_bar` then subtract.** `r_bar`'s
   exact self-terms cancel algebraically in `r_bar - r` (a cell
   contributes nothing to its own displacement) — computing the
   displacement numerator directly (`s_mr - r * a_sum`, then one
   division) avoids ever forming two large, close-in-value numbers to
   subtract.

### Where this landed (R=64/128/256 sweep, orthogonal RFF, self-anchored,
vs. the exact `O(n^2)` reference)

No catastrophic spikes at any `R` in that range (max error under 10 across
all four outputs, vs. millions before the self-anchor). `mu` and `r_bar`
converge cleanly with `R` (correlation to the exact reference: `mu`
0.71->0.72->0.86, `r_bar` 0.75->0.82->0.87 at R=64/128/256). `evidence`/
`uncertainty` are bounded but noisier and don't show as clean a trend —
flagged as unresolved, explicitly **not** further tuned per the user's
instruction to move to end-to-end evaluation instead.

### Implementation (`R=256`, `T=1` — the user's stated experimental
defaults; `num_steps>1` works, `T=1` is the current default)

| Module | Contents |
|---|---|
| `random_features.py` | `RandomFourierFeatures` (Gaussian kernel, orthogonal-or-i.i.d.), `exact_gaussian_kernel`; `PositiveRandomFeatures` kept unused, documented as wrong-kernel for this use case |
| `field_functions.py` | `RoutingFunction`, `BandwidthFunction`, `MassFunction`, `RoutingGateFunction`, `FieldSemanticUpdateFunction` (`GlobalQueryKeyFunction`/`StochasticGateFunction` written for a future global-field phase, unused — this pass is local-only, one step) |
| `field_fusion.py` | `local_field_fusion` (self-anchored ORFF, scalable), `exact_local_field_fusion` (dense `O(n^2)` reference — diagnostic only, never in the trained model) |
| `field_dynamics.py` | `FieldCellState` (`BeliefCellV1` + persistent `r`), `FieldRefinementStep`, `FieldRefinementCore` |
| `field_model.py` | `FieldEncoder` (wraps the existing `PopulationEncoder`/`ObjectSeededEncoder`, adds initial `r_0 = F_route(...)`), `SelfOrganizingRefinementField` |

`WriteGateFunction` (`shared_functions.py`) is reused unmodified for
`(mu, e, u, z)`'s gated write — its `(mu, e, u, mu_local, e_local,
u_local) -> 4 gates` interface (the `use_global=False` case) is exactly
what `(mu, e, u, mu_field, e_field, u_field) -> (beta_mu, beta_e, beta_u,
beta_z)` needs.

Tests: `tests/test_random_features.py`, `tests/test_field_functions.py`,
`tests/test_field_fusion.py`, `tests/test_field_dynamics.py`,
`tests/test_field_model.py` — 30 tests, all passing (full suite
253/253). Confirmed: at the default `T=1`, `RoutingGateFunction` correctly
gets zero gradient (`r`'s movement only matters for a step that doesn't
exist yet); at `T=2` it gets a real one — a deliberate regression test,
not a bug tolerated.

**Measured, not estimated:** `n_cells=128`, `R=256`, `T=1`, `batch=32`,
CPU: **24 ms/training-step** — versus CellV1.1 sparse's ~168 ms/step at a
comparable scale (and that was at `T=6`, this is `T=1`). A short training
smoke test (15 AdamW steps) decreased loss monotonically with no
divergence.

**Not yet done:** no experiment has been run — this is unit-tested for
"matches the specified math and produces valid, bounded output," not
validated for task performance, the way `v1_001_dynamic_groups` did for
CellV1.1/1.2. Global (cross-region) field communication is designed in
the original conversation but not implemented — this pass is local-field-
only, one step. `evidence`/`uncertainty`'s noisier convergence (vs.
`mu`/`r_bar`) is flagged, not resolved.

## 14. CellV1.3.1 — Global communication for the field

**Status: implemented, unit-tested, one comparison sweep run under two
protocols. A real, unresolved reliability question found at scale.**
CellV1.3 (§13) shipped local-field-only; global (cross-region)
communication was designed in the original conversation but deferred.
Added here on top of the frozen local field (`field_fusion.py`,
untouched), via an additive `use_global` constructor flag — `False`
executes the identical local-only lines §13 already had.

### The mechanism (`global_field.py`)

Reuses dense CellV1's send/need/query-key design (§5 Part III) rather
than inventing a new one — a cell's send gate ("how useful is my
information to others") and need gate ("how much outside information do
I currently need") are the same shape/role as before, but retrieval is
**exact linear attention** (Katharopoulos et al. 2020, "Transformers are
RNNs"), not the field's RFF approximation: `phi(x) = ELU(x) + 1 > 0`
always, so `K^G(i,j) := phi(q_i)^T phi(k_j)` *is* the compatibility
kernel by definition — no random projection, no variance, none of the
convergence-with-R questions §13's local field spent most of its
debugging on. `O(n_cells * global_dim)`, `global_dim` (`d_g`, 8-16)
independent of `n_cells`. Exact self-removal (a receiver's raw reduction
provably includes, and then has subtracted, its own contribution) —
same self-anchoring argument as §13's local field, but here it's exact
subtraction of an exactly-known algebraic identity, not anchoring an
approximation to a known constant.

Fused into the existing local proposal via `fusion.py::precision_fusion`
— the same rule dense/sparse CellV1 already uses for its own
self+local+global fuse (§5 Part IV) — over a learned per-source gate,
before handing the result to the *unmodified* `WriteGateFunction` as its
proposal argument. The need gate scales the global proposal's evidence
(`e_global * need`) rather than adding a new `WriteGateFunction` input —
functionally equivalent (low need → weak evidence → the existing
precision-weighted fuse discounts it) without touching a function this
line of work has kept frozen since §12.

### A new task, built to actually require this

`dynamic_groups`'s existing cross-group term pairs groups by spatial
center (`k1`) — close enough to each object's own encoded features that
a wide local field could plausibly shortcut it without any real
non-local channel. `dynamic_groups_global`
(`src/data/synthetic/dynamic_groups.py`) pairs groups by `argmax`/
`argmin` of their *aggregate values* `h_k = tanh(sum_j v_j)` instead —
content only knowable after comparing every group's sum against every
other's, which a similarity-based local field cannot shortcut by
construction. 5 new tests in `tests/test_dynamic_groups_data.py`.

### What was found

At the field's original comparison scale (`n_cells=128`,
`dynamic_groups`, `R=256`, fixed 1500 steps, 3 seeds), `field_local_global_t2`
(R²=0.7872±0.0146) edges out local-only `field_t2` (0.7784±0.0043),
`field_t1` (0.7787±0.0094), and `cellv0.1` (0.7662±0.0145) — a modest,
real advantage for the global channel, no collapse in this final
recorded run.

Extending to a 4-level complexity sweep (`dynamic_groups_global`,
easy/medium/hard/very_hard = 24/48/96/192 objects) under the original
fixed-1500-step protocol found real single-seed collapses — attributed
(per `global_field.py::GlobalSendFunction`'s docstring, written
contemporaneously) to an unlearned, loud global channel getting a full
vote in the fused proposal from step 0; fixed by initializing `send`/
`need`'s bias near-off (`init_bias=-2.0`, matching `WriteGateFunction`'s
existing convention) instead of the unbiased default.
`diagnose_global_collapse.py` was written to isolate undertraining vs.
RFF-resolution vs. a deeper redesign as the explanation for one specific
collapse (`very_hard`/seed=0, test R²=0.224 vs. local-only 0.781 and
CellV0.1 0.816 at the time) — no saved output from running it was found
in this working tree, so its three-way conclusion is not independently
confirmed by this entry.

The fixed-step protocol itself was then flagged as confounded (a
24-object and a 192-object problem don't converge on the same
optimization timescale) and replaced with convergence-based training
(early stopping, `R` raised to 512). Under that corrected protocol, R²
stays close across `cellv0.1`/`field_t2`/`field_local_global_t2` at every
level — **except `field_t2` (local-only, no global) at `very_hard`/seed=0,
which converged to R²=0.0667** against seed 1/2's 0.79/0.82. The
instability recurred under the corrected protocol and higher `R`, and
this time in the *local-only* variant — not explained by the "global
channel amplifies it" framing that motivated `diagnose_global_collapse.py`,
and not fixed by the send/need init change (a global-only mechanism).
**This is open.** Full numbers: `docs/research_log.md`'s CellV1.3.1 entry.

### Implementation map

| Module | Contents |
|---|---|
| `global_field.py` | `elu_feature_map`, `GlobalSendFunction`, `GlobalNeedFunction`, `GlobalQueryKey`, `linear_global_belief_field` |
| `field_dynamics.py` | `FieldRefinementStep` gained `use_global`/`global_dim`; local-only path unchanged when `False` |
| `field_model.py` | `SelfOrganizingRefinementField` passes `use_global`/`global_dim` through |

Tests: `tests/test_global_field.py` (8 tests: validity, finite gradients,
exact self-exclusion for an isolated cell, output sensitivity to other
cells, need=0 independence, permutation equivariance, no quadratic
scaling, gradients reaching every new parameter) — full suite 274/274.
**Gap:** `field_dynamics.py`'s module docstring claims a
"`tests/test_field_local_global_ablation.py` zero-regression check"
verifying `use_global=False` is byte-identical to pre-change behavior —
no file or test by that name exists in this repository. The claim should
be backed by an actual test or removed from the docstring; not fixed as
part of this write-up.

## 15. CellV1.4 — Learned Association Field

**Status: implemented, unit-tested, run through the same 3-seed/4-level
convergence sweep as the field variant it replaces. Currently the more
reliable of the two field-style CellV1 variants.** CellV1.3's local field
(§13) spent `R=256`-`512` random Fourier features approximating a
Gaussian kernel chosen ahead of time over a routing coordinate `r` — and
§14 found that even at `R=512` this approximation is not fully
scale-stable (one seed still collapsed at `very_hard`). The user's
redesign, in their own words: *"We shouldn't spend hundreds of dimensions
accurately approximating a similarity function that we chose. The
network should learn the similarity function itself."*

### The mechanism (`learned_association.py`, `association_dynamics.py`)

```text
phi_i = softplus(F_assoc(mu_i, e_i, u_i, z_i))     (batch, n, D)
K_ij  = phi_i . phi_j
```

One small shared function; the dot product of its output *is* the
association kernel by construction, not an estimate of one — no
variance, no "does this converge as `D` grows" question. `D`
(`assoc_dim`, default 32) is independent of `n_cells` and an order of
magnitude below the field's `R=512`. No routing coordinate `r`, no
bandwidth/mass functions, no mean-shift — state stays plain
`BeliefCellV1 (mu, e, u, z)`, the same state dense/sparse CellV1 already
use, so (unlike `FieldCellState`) no wrapper dataclass is needed.
Aggregation reuses the same precision-weighted, scale-stable reduction
pattern as every prior CellV1 variant (population-wide reductions once,
each receiver reads them through its own `phi_i`, `O(n_cells * D)`), with
exact (not anchored-approximate) self-removal since `phi` is learned, not
an RFF estimate of some other target kernel. `z`'s update is **not** a
kernel-weighted average of neighbors (the field's own docstring already
flags that as the wrong move for an identity field, not a numerical
detail) — a learned, gated function of `(z, local content summary)`,
same role `(r, r_bar)` played for the field variant.

Global communication (§14) is reused **completely unmodified** — same
`GlobalSendFunction`/`GlobalNeedFunction`/`GlobalQueryKey`/
`linear_global_belief_field`/`precision_fusion`/`WriteGateFunction` — per
the user's explicit instruction to keep the global mechanism and feed it
this fusion's local output instead of the field's.

### What was found

A single-seed sanity check (`check_association_field.py`, the user's
explicit "narrow check, not another huge sweep") against
`field_local_global_t2` (R=512) at matched params: `easy` — field
R²=0.8152/600 steps/11.9s vs. association R²=0.8069/200 steps/3.24s;
`very_hard` — field R²=0.8136/2300 steps/80.7s vs. association
R²=0.8205/900 steps/16.1s. Comparable-or-better accuracy in roughly a
third the steps and a fifth the wall-clock, on this one seed.

The fuller 3-seed, 4-level convergence sweep
(`run_complexity_scaling_association.py`, `cellv0.1` vs.
`association_local_global_t2` only — the field variants are retired from
active development per that script's own docstring) confirms this isn't
a single-seed artifact: R² tracks `cellv0.1` within its own seed noise at
every level (easy 0.7896±0.0124 vs. 0.7880±0.0219; medium 0.8405±0.0071
vs. 0.8448±0.0073; hard 0.8287±0.0092 vs. 0.8292±0.0096; very_hard
0.8211±0.0004 vs. 0.8191±0.0052), and **no collapse on any of the 12
(level, seed) combinations** — `very_hard`'s ±0.0004 is the tightest seed
variance of any model/level combination recorded in this experiment
line. Steps-to-convergence is mixed, not uniformly faster than the
single-seed check suggested: fewer steps than `cellv0.1` at `easy` (967
vs. 2300), comparable-to-slightly-more at `hard`/`very_hard` (1267 vs.
967; 1367 vs. 1233) — the "3-9x fewer steps" figure from the single-seed
check does not hold up unchanged and should not be quoted as the
headline. Per-step wall-clock is consistently lower than the field
variant it replaced (e.g. `very_hard`: ~18ms/step vs. ~37ms/step at
R=512), though the ~2x gap is smaller than `D=32` vs. `R=512`'s ~16x
feature-count ratio would suggest — fixed costs (global communication,
write-gate, semantic update) are shared and don't shrink. Full numbers:
`docs/research_log.md`'s CellV1.4 entry.

### Implementation map

| Module | Contents |
|---|---|
| `learned_association.py` | `AssociationFunction`, `learned_local_association_fusion`, `AssociationSemanticUpdateFunction` |
| `association_dynamics.py` | `AssociationRefinementStep`, `AssociationRefinementCore` |
| `association_model.py` | `LearnedAssociationField` — encoder → core → decoder, unwrapped (no `r_0` to compute, unlike `FieldEncoder`) |

Tests: `tests/test_learned_association.py` (8 tests), full suite
274/274. **Not yet done:** an adaptive per-input continue/refine gate
(`T1` vs. `T2` decided from task loss rather than a fixed `num_steps`) is
named in the module's own docstring as deliberately deferred ("I would
not do another huge sweep") — `AssociationRefinementCore` still takes a
fixed `num_steps`, a known gap against the fuller spec, not a bug.

## 16. CellV1.5 (proposed, NOT FINALIZED) — Persistent structural substrate with dynamic functional gating

**Status: proposed across a multi-turn design conversation, same day as
§14/§15. Not finalized — several open questions listed below. No code
written for this section.** Captured here per the same convention CellV1
itself started with (§17's 2026-09-01 entry: "proposal captured... not
finalized, eleven open questions listed").

### Motivation

Every CellV1 variant so far (§1-§15) answers "who should this cell talk
to right now" by *recomputing the entire candidate search from scratch,
every refinement step* — dense pairwise scoring (§5), LSH candidate pools
(§11), a continuous kernel field (§13), or the exact learned kernel
(§15). The user's diagnosis, prompted by a design discussion the user had
outside this session and then relayed here: this treats "dynamic
functional organization" as "dynamically rebuild the wiring," which
doesn't match how cell-assembly theory describes cortical organization —
a relatively persistent synaptic substrate, with functional assemblies
emerging from *which parts of that substrate are currently active*, not
from redrawing the substrate itself. Reference: cell assemblies as
distributed, overlapping populations whose functional participation
changes with activity (Buzsáki 2010, "Neural syntax," and the
"synapsemble" framing of dynamically-changing effective-synapse
constellations it draws on).

### Core decomposition

```text
w_ij    -- structural connectivity: persistent, sparse, slowly learned
a_ij(t) -- functional connectivity: input-dependent, recomputed every step
c_ij(t) = w_ij * a_ij(t)   -- effective communication strength
```

`w_ij` says "this pair *can* communicate" (changes over training, not
within a forward pass). `a_ij(t)` says "should they, right now" (changes
every refinement step, same role §15's `a_ij` already plays). The
architectural claim: functional assemblies should be able to emerge from
which *existing* edges are currently active, without ever materializing
or scoring an `(n_cells, n_cells)` structure.

### What's specified

- **Cell state:** unchanged — `BeliefCellV1 = (mu, e, u, z)` (§3), same
  as every variant since §1.
- **`a_ij(t)`:** reuse §15's `AssociationFunction` unmodified — `phi_i =
  softplus(F_assoc(mu_i, e_i, u_i, z_i))`, evaluated only for cells with
  an existing structural edge (not the full population), `a_ij(t) =
  phi_i . phi_j` restricted to that edge.
- **`w_ij` — RESOLVED (2026-09-03): implicit, address-derived, never a
  stored per-edge `Parameter`.** Every persistent cell has a learned
  **structural address** `s_i in R^{d_s}` — an `(n_cells, d_s)` parameter
  table, one row per cell slot. Unlike `z`, `s_i` does **not** depend on
  the current input or belief state: it is a fixed per-cell identity,
  trained by ordinary backprop, the same for every example/batch —
  "who this cell structurally is," not "what it currently believes."
  For an existing directed edge `(i, j)`:

  ```text
  q_i = W_out s_i,  k_j = W_in s_j      (shared learned linear maps, no bias)
  w_ij = tanh(q_i . k_j / sqrt(d_s))
  ```

  Storage is only the sparse edge topology (an `(E, 2)` index list) plus
  non-parameter structural metadata (utility, age — open question 1
  below); `w_ij` itself is never stored, only recomputed from
  `s_i`/`s_j`/`W_out`/`W_in` for whichever edges currently exist, every
  forward pass — the same "shared function over the current structure,
  not a per-pair free parameter" convention every other CellV1 mechanism
  already follows (§2's rule against `BeliefLayer`-style permanent
  per-pair parameters extends cleanly from routing to the structural
  graph itself). This is also what resolves the growing/shrinking-
  parameter-tensor problem the first draft of this section flagged:
  growth needs no weight initialization scheme (a new edge's `w_ij`
  is just the formula evaluated on that pair, immediately well-defined),
  and pruning has no optimizer state to reconcile (there was never a
  per-edge optimizer slot to begin with) — it's exactly a row deletion
  from the edge-index list.

  Compute: project every cell's address once per forward pass —
  `q_all = S W_out^T`, `k_all = S W_in^T`, `O(n_cells * d_s^2)` — then
  gather `q_i`/`k_j` for the `E` existing edges and dot them, `O(E *
  d_s)`. No `(n_cells, n_cells)` tensor at any point, matching the same
  budget §11's LSH candidate search and §15's association kernel already
  hold themselves to.
- **Structural plasticity is occasional, not per-step.** Rewiring happens
  at discrete structural-plasticity events during training (a schedule,
  not every forward pass); between events the edge topology is fixed —
  `s_i`, `W_out`, `W_in`, and every other parameter still train normally
  every step, so `w_ij` for existing edges keeps changing even without a
  plasticity event, just not *which* edges exist.
- **Growth, at a structural-plasticity event:**
  1. Compute every cell's `phi_i` via the existing shared
     `AssociationFunction` (§15) — the *learned* representation, not a
     fixed/random one.
  2. Build or update an approximate-nearest-neighbor / maximum-inner-
     product-search (ANN/MIPS) index over those `phi` vectors.
  3. For each cell, retrieve a small candidate pool via approximate
     search in that index — **the ANN/LSH mechanism is only the search
     algorithm over the learned `phi`-space; it is explicitly not itself
     the growth criterion** (the user was explicit: fixed-random-LSH
     similarity must not be mistaken for the biological/learned
     criterion — contrast with §11, where LSH candidate-filtering *was*
     the whole mechanism, on a fixed/unlearned hash).
  4. Run an *exact* learned growth score only on that small candidate
     pool — combining association compatibility with accumulated
     activity/utility statistics (exact formula: open, see below).
  5. Add only the highest-scoring proposed edges to the edge-index list —
     no weight to initialize; `w_ij` is immediately defined by the two
     cells' current structural addresses (resolved above).
- **Pruning:** maintain a running per-edge utility statistic (based on
  the edge's contribution to loss/gradient, optionally co-activity);
  periodically remove persistently low-utility edges from the edge-index
  list (resolved above: nothing else to clean up).
- **Explicit constraints from the user, verbatim in spirit:** never
  materialize a dense `(n_cells, n_cells)` matrix or parameter anywhere;
  no fixed number of groups; no permanently designated local/global
  cells (consistent with §1's original CellV1 hypothesis, which already
  ruled this out for routing — this extends the same principle to the
  structural graph itself).

### Open questions (not to be resolved by the agent)

1. **Utility-statistic formula.** "Contribution to loss/gradient,
   optionally co-activity" is a description, not a formula — e.g. an EMA
   of `|dL/dw_ij|`, of the message/`c_ij` magnitude actually carried, a
   measure of `a_ij(t)` co-activity over recent steps, or some
   combination, and with what decay/window.
2. **Growth-score formula.** How association compatibility (`phi_i .
   phi_j`) and the accumulated utility/activity statistics combine into
   one score for ranking candidates — a fixed combination (e.g. product,
   sum) or a small shared learned function in the style of `F_need`/
   `F_offer` (§5 Part III)?
3. **Aggregation formula.** Does `learned_local_association_fusion`
   (§15) carry over unchanged, just with its sums restricted to each
   cell's structural neighbor set instead of the whole population (same
   math, smaller domain), or does `c_ij(t)` change the fusion formula
   itself?
4. **Edge budget and bootstrap topology.** Target average degree `k`
   (analogous to §11's `pool_size`, §13's `R`), and what a freshly
   encoded cell population's structural graph looks like *before* the
   first plasticity event — empty (bootstrapped entirely by early growth
   events) or seeded with one initial growth pass before training starts?
5. **Plasticity schedule.** What "occasionally" means concretely (every
   `K` optimizer steps? every `K` refinement steps `T`? epoch
   boundaries?), and how many edges are added/pruned per event.
6. **Relationship to existing local/global routing (§5 Parts I & III)
   and the global field (§14).** The user's original framing suggested
   local-vs-global should simply fall out of short- vs. long-range edges
   in *one* substrate, not a separate mechanism — does this proposal
   replace §5/§14's routing split entirely, or run alongside it?

### Relationship to prior variants

Builds directly on §15's `AssociationFunction` for `a_ij(t)` — not a new
kernel. Reframes, rather than replaces, the question §11 (LSH candidate
pools) and §13 (RFF field) each answered: those found "who is dynamically
similar right now" from scratch every step; this asks "who is my
*persistent* structural neighbor, and how strongly are we currently
talking." §11's ANN-style candidate search returns as a sub-component
(step 3 above) but now searches learned `phi`-space for a slow structural
decision, not a per-step routing decision.

## 17. Revision history

- 2026-09-01: Proposal captured from a design conversation with the user —
  self-organizing, input-dependent belief graph; no fixed
  layers/compartments; `CellV1 = (mu, e, u, k)` with `k` as a new
  association/key field. Not finalized — eleven open questions listed.
- 2026-09-02: User fully specified the mechanism (semantic-distance
  association, `sparsemax`/entmax-style sparsification with a null option,
  mutual local graph, reused Method E fusion, need/offer-gated query/key
  global routing, self+local+global fuse, semantic-address update),
  grounded in a literature search (DGCNN, Routing Transformer, Slot
  Attention, capsules, Growing Neural Gas, RIMs, Global Workspace — §0),
  and asked for it to be implemented. Implemented in
  `src/models/architecture_v1/` (`k` renamed `z` throughout, matching the
  user's own notation switch in this turn). Resolved nine of the eleven
  original open questions; encoding/decoding remain implementation-choice
  defaults (§6); staging relative to "isolation of variables" remains open
  (§9). A gradient-blowup bug in the mutual-association `sqrt` was caught
  and fixed during implementation (§7).
- 2026-09-02 (later same day): User identified that dense CellV1's
  `O(n_cells^2)` routing is a scaling dead end, grounded in Reformer
  (LSH attention, `O(n log n)`) and Routing Transformer (online-clustering
  attention, `O(n^1.5)`) as precedent for content-dependent sparse
  routing, and specified CellV1.1 — the same cell/update math, LSH-based
  candidate pre-filtering before the expensive association scoring.
  Implemented in `lsh.py`/`sparse_routing.py`/`sparse_dynamics.py`/
  `sparse_model.py` (§11); dense (§1-§10) kept unmodified as "CellV1 Dense
  Reference." Validated exact numerical consistency with dense at small
  `n_cells` (candidate pool covering everyone) and measured real,
  widening speedup with scale (2.15x at 128 cells, 12.00x at 512).
- 2026-09-02 (later still): `v1_001_dynamic_groups`'s first real run
  found CellV1 (dense and sparse both) flatlining near `R^2=0` on train
  *and* test while parameter-matched `mlp`/`cellv0.1` learned fine. An
  isolation diagnostic (dense vs. sparse, train R² recorded) ruled out
  routing/generalization and pointed at the recurrent state-write itself:
  every cell was fully overwritten by the fused proposal every step, with
  no mechanism to resist it -- the recurrent/message-passing analogue of
  GNN oversmoothing. Fixed with CellV1.2 (§12): a learned per-channel
  write gate (`shared_functions.py::WriteGateFunction`), `next = (1 -
  beta) * old + beta * proposal`, bias-initialized near `0.12` so cells
  start mostly self-preserving. Confirmed: train R² `0.016 -> 0.72`, test
  `0.008 -> 0.73`, no new overfitting gap.
- 2026-09-03: User identified CellV1.1's discrete LSH candidate search as
  an engineering choice imposed on the architecture ("make 8 groups,"
  "keep 6 neighbors") rather than something learned, and specified
  CellV1.3, the Self-Organizing Refinement Field (§13): local structure
  as density modes of a continuous, learned kernel field (mean-shift),
  with routing position `r` self-organizing via a gated mean-shift step
  and semantic identity `z` updated by a learned gate rather than
  reconstructed as a density average. Grounded in differentiable mean-
  shift, kernelized linear attention (Performer/FAVOR+), and L0/hard-
  concrete stochastic gates as precedent. Implementation surfaced and
  fixed a real chain of bugs before landing on a working design (wrong
  kernel family, a stabilizer that didn't cancel, a hidden `O(n*R^2)`
  term, signed-kernel division blowups) -- resolved by self-anchoring the
  local density on the exactly-known `K(i,i)=1` rather than any further
  epsilon patching. `field_dynamics.py`/`field_model.py` implemented at
  the user's stated defaults (`R=256`, `T=1`); no experiment run yet.
- 2026-09-03 (later same day): User asked for CellV1.3's deferred global
  communication, specified as a reuse of dense CellV1's send/need/
  query-key design (§5 Part III) via exact linear attention (Katharopoulos
  et al. 2020), plus a new task (`dynamic_groups_global`, argmax/argmin
  group-value pairing) built specifically to require it, since the
  existing spatial-pairing task could be shortcut by a wide local field.
  Implemented (§14: `global_field.py`, `field_dynamics.py`/`field_model.py`'s
  `use_global` flag). A real single-seed collapse was found and traced to
  an unlearned, loud global channel voting from step 0; fixed via a
  near-off gate-bias init. A *different*, unresolved collapse (local-only
  field, `very_hard`/seed=0, under the corrected convergence protocol at
  R=512) remains open. `diagnose_global_collapse.py` was written to
  isolate the cause of the first collapse; no saved output from it was
  found in this working tree.
- 2026-09-03 (later still): User identified that even `R=512` doesn't make
  CellV1.3's RFF-approximated kernel fully scale-stable (§14's open
  finding), and specified CellV1.4: stop approximating a chosen similarity
  function and learn one directly. Implemented (§15:
  `learned_association.py`, `association_dynamics.py`,
  `association_model.py`) -- `phi_i = softplus(F_assoc(mu_i,e_i,u_i,z_i))`,
  `K_ij = phi_i . phi_j`, exact by construction. Global communication
  (§14) reused unmodified per explicit instruction. A single-seed sanity
  check against the field variant, then a 3-seed/4-level convergence
  sweep, found accuracy matching CellV0.1 within seed noise at every
  level and no collapse across any of 12 (level, seed) combinations --
  currently the more reliable of the two field-style CellV1 variants.
- 2026-09-03 (later still): User relayed a design conversation proposing
  a pivot for Hypothesis B, grounded in cell-assembly theory (Buzsáki):
  stop dynamically rebuilding the graph every step (§1-§15's shared
  approach) and instead separate a persistent, sparse, slowly-learned
  structural connectivity `w_ij` from a dynamic functional gate `a_ij(t)`
  (reusing §15's `AssociationFunction`), with `c_ij(t) = w_ij * a_ij(t)`.
  Across two follow-up turns the user specified structural plasticity as
  occasional (not per-step) growth/pruning: new-edge candidates proposed
  via approximate nearest-neighbor search over the *learned* `phi`-space
  (ANN/LSH as an indexing algorithm only, never the growth criterion
  itself), an exact learned growth score run only on that small candidate
  pool, and existing edges pruned by a running utility statistic. Captured
  as CellV1.5 (§16) -- explicitly proposed, not finalized; several open
  questions listed (utility/growth-score formulas, how `w_ij` is
  parameterized given edges are added/removed at runtime, the aggregation
  formula's exact domain, edge budget/bootstrap topology, plasticity
  schedule, and the relationship to existing local/global routing). No
  code written.
- 2026-09-03 (later still): User resolved §16's most consequential open
  question -- `w_ij`'s parameterization -- with a full formula: every
  cell gets a persistent, input-independent learned structural address
  `s_i`, and `w_ij = tanh((W_out s_i) . (W_in s_j) / sqrt(d_s))` for an
  existing edge, computed on demand rather than stored. Only the sparse
  edge-index list plus non-parameter metadata (utility, age) persists;
  there is no per-edge `Parameter`, so growth needs no weight
  initialization and pruning is a plain row deletion -- the earlier
  concern about a growing/shrinking parameter tensor colliding with
  optimizer state doesn't arise. `O(n_cells * d_s^2 + E * d_s)`, no
  `(n_cells, n_cells)` tensor at any point. Recorded in §16's "What's
  specified" (moved out of open questions); six questions remain open
  (utility/growth-score formulas, aggregation domain, edge budget/
  bootstrap topology, plasticity schedule, relationship to existing
  local/global routing). Still no code written.
