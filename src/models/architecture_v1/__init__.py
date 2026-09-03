"""ArchitectureV1 (working name "Dynamic Belief Graph") -- implementation of
the CellV1 synthesis proposed and specified by the user on 2026-09-02
(`docs/architecture_v1.md`).

Unlike `architecture_v0/`, this package has no per-module `NotImplementedError`
gate: the user's design conversation resolved essentially every open
question `docs/architecture_v1.md` originally listed (association
dimensionality, the semantic-distance/sparsification/routing formulas, the
reused scale-stable-precision fusion, the three-source self/local/global
fuse, the semantic-address update). See `docs/architecture_v1.md` for the
full spec and its revision history, and `docs/research_log.md`
("Architecture V1 proposed") for the design-conversation context.

A few pieces were never pinned down by the user because they're glue code
any implementation needs regardless of the belief-mechanism math (how raw
input features become the initial `N`-cell population; how a
variable-topology population is read out into a fixed-size prediction) --
`encoder.py` and `decoder.py` each document the specific default chosen
and flag it explicitly as an implementation choice, not a specified
formula, per `docs/architecture_v1.md` §6 (open questions 9-10).

Modules:

- `cell.py`      -- `BeliefCellV1`, the `(mu, evidence, uncertainty, z)` state.
- `routing.py`   -- `LocalAssociation`, `GlobalRouting`: the dynamic,
                     input-dependent sparse graph construction (§1, §3).
- `shared_functions.py` -- the small shared MLPs (`F_msg`, `F_need`,
                     `F_offer`, `F_z`) reused identically by every cell/step.
- `fusion.py`    -- `precision_fusion`: the scale-stable-precision fusion
                     formula (identical to
                     `src.models.architecture_v0.integration`'s Method E /
                     "CellV0.1"), reproduced here (not imported) because V1
                     additionally needs the returned content-weights
                     `alpha` for the semantic-address update, and operates
                     over a dynamically computed association matrix rather
                     than a learned dense `(w_ij, g_ij)` pair.
- `dynamics.py`  -- `DynamicBeliefGraphStep`: one refinement step (local
                     association -> local fusion -> need/offer -> global
                     routing -> global fusion -> self/local/global fuse ->
                     semantic-address update), shared across all `T` steps.
- `encoder.py`   -- input features -> initial `N`-cell population.
- `decoder.py`   -- final `N`-cell population -> task output.
- `model.py`     -- `DynamicBeliefGraph`: encoder -> `T` shared refinement
                     steps -> decoder.
"""
