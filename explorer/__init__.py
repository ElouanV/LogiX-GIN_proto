"""LogiX-GIN explorer: an interactive view of a trained (prototype) LogiX-GIN.

Kept apart from the training / performance code on purpose: nothing here changes a model,
a run directory or a result. It only reads a checkpoint and shows its *symbolic* reading
(the same one interp_metrics.py scores), so every number on screen is an exact property
of the rules, not an approximation of the network.

    explorer/reader.py   SymbolicReader: the model card (architecture, literals and their
                         step functions, unit thresholds and weights, rule counts,
                         prototypes, head) and the trace of one graph (every count,
                         literal, margin, prototype match and the decision)
    explorer/bundle.py   static export: card + graph index + traces in one JSON, or one
                         self-contained HTML file
    explorer/server.py   local server for the web app, loads runs on demand
    explorer/web/        the web app (plain JS + SVG, no dependency, works offline)

    python -m explorer serve --root results_final                # browse every run
    python -m explorer serve --run <seed dir>                     # one run
    python -m explorer export <seed dir> --split test --html out.html
"""
