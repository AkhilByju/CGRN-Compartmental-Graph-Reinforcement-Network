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

## 14. Revision history

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
