# lm-evaluation-harness Report — Track 01 official benchmarks

- Status: **NOT RUN**
- Checkpoint: `experiments/nova/toy_run/best.ckpt`
- Model config: `configs/model_nova_tiny_demo.yaml`
- Tasks requested: hellaswag
- lm-eval version: 0.4.13
- Date: 2026-09-24T21:12:09.704671Z
- Command: `python evaluation/run_lm_eval.py --checkpoint experiments/nova/toy_run/best.ckpt --model_config configs/model_nova_tiny_demo.yaml`

**NOT RUN.** Reason: lm-evaluation-harness is installed and the model wrapper is implemented (see evaluation/run_lm_eval.py), but task data download failed. Task data for hellaswag/arc_easy/piqa/winogrande is served from the Hugging Face Hub, which is not reachable from this container's network egress allow-list (see docs/COMPETITION_RULES.md). Re-run this exact script on a machine with normal internet access to get real results.

Captured error: `FileNotFoundError: Couldn't find any data file at /home/claude/NOVA-50/Rowan/hellaswag. Couldn't find 'Rowan/hellaswag' on the Hugging Face Hub either: LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your conn`
