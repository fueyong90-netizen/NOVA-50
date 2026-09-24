# Demo Video Script (2-5 min) — en anglais, sans chiffres non mesurés

0:00 — Problem
"Large language models need massive compute. Can a model under 50 million
trainable parameters still show meaningful language and reasoning ability?"

0:20 — NOVA-50 concept
"NOVA-50 explores this with a from-scratch causal language model, compared
head-to-head against a standard baseline transformer, under a strict
parameter budget."

0:50 — Architecture
Show `model/layers.py` / `model/transformer.py` and the parameter budget
report (`results/parameter_report.md`) — call out the automatic budget
checker and its PASS/FAIL test.

1:20 — Training
Show `training/train.py` running live (CPU), the JSONL logger output, and
mention hardware auto-detection.

1:50 — Results
Show the REAL loss curves in `results/plots/loss_curves.png`, and state
clearly on-screen: "trained on a synthetic demo corpus in a CPU-only
environment — full-scale results pending real data + GPU."

2:30 — Live demo
Open `web/index.html` with `inference/server.py` running, type a prompt,
show the generated text and the "Trainable parameters: X / 50,000,000" badge.

3:20 — Innovation
Mention the parameter-sharing ablation and its measured (not assumed) result.

4:00 — Conclusion
"NOVA-50's pipeline is fully functional end-to-end and honestly documented —
next step is scaling it up with real data and compute."
