# Demo Video Script (2-5 min, English)

**0:00-0:20 — Problem**
"Large language models need massive compute. Track 01 asks a different
question: how much language and reasoning ability can a model show under a
strict 50-million-parameter budget?"

**0:20-0:50 — Research question**
"NOVA-50 explores this with a from-scratch causal language model, built to
be directly comparable to a standard baseline Transformer using the exact
same training code."

**0:50-1:20 — Architecture**
Show `model/layers.py` / `model/transformer.py`, then
`results/parameter_report.md` — highlight the automatic PASS/FAIL budget
check and its enforcing test.

**1:20-1:50 — Training**
Show `training/train.py` running live, the curriculum scheduler switching
phases mid-run (visible in the console output), and the JSONL logger.

**1:50-2:40 — Benchmark results**
Be explicit on screen: "Official HellaSwag/ARC-Easy/PIQA/WinoGrande and
WikiText-103 results are NOT RUN in our development environment — network
access to Hugging Face was blocked. Here's the real error we captured, and
the exact command to reproduce this on a normal machine." Then show the real
toy-corpus ablation results (matched-baseline comparison, parameter-sharing,
curriculum) from `docs/ABLATIONS.md`.

**2:40-3:20 — Live inference demo**
Open `web/index.html` with `inference/server.py` running, type a prompt,
show the generated text and the "Trainable parameters: X / 50,000,000" badge.

**3:20-4:00 — Innovation**
Mention the parameter-sharing result (3/3 seeds) and the fair matched-size
baseline comparison.

**4:00-4:30 — Reproducibility / conclusion**
"Every script here is ready to run on a GPU machine with normal internet
access — that's the only thing standing between this repository and full
competition-scale results."
