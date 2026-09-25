# Screenshot targets

1. **parameter_budget.png** — terminal output of
   `python scripts/count_parameters.py --config configs/model_nova.yaml`
   (shows the PASS status and the full breakdown table)
2. **training_curves.png** — `results/plots/loss_curves.png` opened full-screen
3. **benchmark_status.png** — terminal or file view of
   `results/benchmarks/lm_eval_report.md` (shows the real NOT RUN status,
   not a placeholder — this is intentional and should be shown, not hidden)
4. **web_demo.png** — browser window with `web/index.html` open, a prompt
   filled in, and a generated response visible
5. **tests_passing.png** — terminal output of `pytest -q` showing all tests green

Take these AFTER re-running `scripts/reproduce.py` in the final submission
environment (not only in this development container).
