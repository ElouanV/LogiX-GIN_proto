---
name: refactor
description: Behaviour-preserving refactoring of the LogiX-GIN research codebase — remove duplicated code, move logic into OOP modules, thin out entry-point scripts, clean the repo — while proving numbers do not change. Use when the user asks to refactor, deduplicate, restructure, modularize, clean up the repo, "factor out", merge the *_node / models_proto / PLEX copies, or when a new feature would otherwise copy-paste existing code.
---

# Refactor (behaviour-preserving)

A refactor in a research repo is only acceptable if **every result it can produce stays
identical**. The workflow below makes that checkable, commits in small reversible steps,
and never mixes refactoring with behaviour changes.

Read `CLAUDE.md` first (conventions, invariants, reproducibility and git rules). Use the
`logix-gin` interpreter: `~/miniconda3/envs/logix-gin/bin/python`.

## 1. Scope and map

1. Name the target: which files, which duplication or structural problem, and what the end
   state looks like (the module, the class, the public API).
2. Measure duplication instead of guessing:
   ```bash
   diff -u a.py b.py | grep -c '^[+-][^+-]'          # differing lines between twins
   grep -rn "def <name>\|class <name>" --include=*.py .  # every definition of a symbol
   ```
3. Find every caller of what will move: scripts, `tests/`, `nbs/*.ipynb` (grep the JSON),
   `scripts/*.sh` and `PLEX/`. Also pickled or `torch.save`d objects: a class that moves
   module breaks `torch.load` on full-model pickles. State dicts are fine.
4. Present the plan to the user (what moves where, which callers change, which checks prove
   equivalence) and wait for approval if it touches upstream reference code (`models/`,
   original `train_*.py`) or more than a few files.

Known hotspots in this repo (re-measure, they may have changed):
- `models_proto/tell.py` and `models_proto/model_node.py` are byte-identical copies of
  `models/`. `models_proto/model.py` differs from `models/model.py` by the `init_eps`
  removal only. Make these imports, not copies.
- `train_*.py` vs `train_*_node.py` and `optimize_*.py` vs `optimize_*_node.py`: large twins
  with the graph-level vs node-level task as the main difference.
- `PLEX/proto.py` is a copy of `models_proto/proto.py` with only the import changed. PLEX is
  a separate repo, so ask before coupling them.
- `__pycache__/*.pyc` files are tracked in git and `.gitignore` doesn't cover them.

## 2. Capture a baseline before touching code

Write a throwaway equivalence script **in the session scratchpad**, not in the repo. It
must pin the current behaviour:
- **Model outputs:** build each affected model with fixed args, load a real checkpoint from
  `results*/` (or `torch.manual_seed` weights), run the forward pass on a fixed batch, and
  save outputs plus intermediate `phi_in`, `reg_loss` and `entropy_loss`.
- **State-dict keys and shapes:** checkpoints must still load after the refactor.
- **Training determinism:** a short run of each affected entry point (1-3 epochs, one seed,
  small dataset like MUTAG). Save its losses and metrics. Set seeds and
  `torch.backends.cudnn.deterministic = True`, or compare on CPU.
- **Rule extraction**, if touched: the `latent_logic` / `unpack_rules` / `rule_eval`
  output for one run.
- Also run the existing test suite:
  `~/miniconda3/envs/logix-gin/bin/python -m unittest discover tests -v`.

Keep it cheap (green-compute rule): seconds to a few minutes, never a full training.

## 3. Refactor in small commits

- Work on a branch `refactor/<name>` (see Git workflow in CLAUDE.md).
- One logical step per commit, e.g. "replace models_proto/tell.py copy with import". After
  each step, rerun the baseline script and the tests. Outputs must match exactly
  (`torch.equal`), or to float tolerance if the op order legitimately changed. Say which.
- Commit messages carry no Claude attribution (CLAUDE.md).
- Design rules:
  - Shared logic moves into a package module (`models/`, `models_proto/`, `utils/`). Entry
    scripts end up as argparse plus calls.
  - Variants that differ by a parameter take the parameter (`task='graph'|'node'`) or share
    a base class with small overrides. Don't use a flag-riddled god function.
  - Keep public names that other code or checkpoints rely on: `GINTELL`, `.convs`, `.fc`,
    `reg_loss`, `entropy_loss`, `phi_in`. Keep CLI flags and their defaults. Keep output
    paths.
  - Don't collapse the three prototype variants into one class (project decision).
  - Delete the old copy in the same commit that redirects its callers. Don't leave
    re-export shims unless an external caller (notebook, PLEX) needs one, and if so say so.
- **Never mix in behaviour changes.** A bug found while refactoring is reported and fixed
  in its own separate commit, which notes which results it invalidates.

## 4. Clean up and report

- Remove what the refactor made dead: unused imports, functions, files and stale `.pyc`.
  Delete the scratchpad baseline script. Nothing temporary goes into the repo.
- If a check belongs in the test suite long-term (e.g. a checkpoint-loading
  compatibility test), add it to `tests/` as a proper test.
- Report to the user:
  - what moved where
  - the lines removed vs added
  - which equivalence checks passed (exact or tolerance)
  - callers you couldn't verify (notebooks are only grepped, not run)
  - anything left for later
