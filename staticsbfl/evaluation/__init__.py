"""The final evaluation, as data.

``accuracy``  the prediction results of the four phases: reading, filtering
              (predictor, model type, metric, target, population), the best
              case with its baselines, the overview, per-phase highlights
``costs``     what the whole run cost: training, scoring, table, fitting --
              light and heavy apart
``runtime``   the runtime comparison: measured records, per-tool and
              per-design aggregates, speed-ups, amortization, fidelity

Read only, and nothing here prints or draws: :mod:`staticsbfl.log.evaluation`
renders these results to the console and log, :mod:`staticsbfl.viz.html_report`
to an HTML page, and ``staticsbfl/cli/evaluate.py`` ties them together.
"""
