#!/usr/bin/env python3
"""
Competition Readiness Audit — checks that the required artifacts exist and
that the mandatory automated checks (parameter budget, tests) actually pass.

This checks PRESENCE and mechanical PASS/FAIL of scripts — it does NOT and
cannot certify that official benchmark numbers exist, because (as of this
run) they don't: HellaSwag/ARC-Easy/PIQA/WinoGrande/WikiText-103 results are
NOT RUN in this environment (see docs/COMPETITION_RULES.md). This script
reports that honestly as WARNING/FAIL rather than skipping the check.

Exit code is non-zero if any MANDATORY requirement is missing/failing.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

RESULTS = []  # (name, status, detail, mandatory)


def check_file(name, path, mandatory=True):
    ok = (ROOT / path).exists()
    RESULTS.append((name, "PASS" if ok else "FAIL", str(path), mandatory))


def check_command(name, cmd, mandatory=True, timeout=120):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        status = "PASS" if r.returncode == 0 else "FAIL"
        detail = (r.stdout[-300:] + r.stderr[-300:]).strip().replace("\n", " ")
        RESULTS.append((name, status, detail, mandatory))
        return r.returncode == 0
    except Exception as e:
        RESULTS.append((name, "FAIL", str(e), mandatory))
        return False


def check_benchmark_status(name, path):
    p = ROOT / path
    if not p.exists():
        RESULTS.append((name, "NOT RUN", f"{path} does not exist yet", False))
        return
    try:
        data = json.loads(p.read_text())
        status = data.get("status", "UNKNOWN")
        mark = "WARNING" if status == "NOT RUN" else ("PASS" if status == "RUN" else "FAIL")
        RESULTS.append((name, mark, f"status={status} ({path})", False))
    except Exception as e:
        RESULTS.append((name, "FAIL", str(e), False))


def main():
    # --- Presence checks (mandatory artifacts) ---
    check_file("Model config exists (baseline)", "configs/model_baseline.yaml")
    check_file("Model config exists (NOVA-50)", "configs/model_nova.yaml")
    check_file("Training config exists", "configs/train_toy_nova.yaml")
    check_file("Parameter budget checker exists", "scripts/count_parameters.py")
    check_file("Benchmark runner exists (lm-eval)", "evaluation/run_lm_eval.py")
    check_file("WikiText-103 evaluator exists", "evaluation/wikitext103.py")
    check_file("Benchmark contamination check exists", "evaluation/benchmark_contamination.py")
    check_file("Dataset manifest exists (toy)", "data/raw/MANIFEST.json")
    check_file("Production dataset docs exist", "docs/PRODUCTION_DATASET.md")
    check_file("README exists", "README.md")
    check_file("Technical report exists", "docs/TECHNICAL_REPORT.md")
    check_file("Built With exists", "submission/built_with.md")
    check_file("Devpost description exists", "submission/devpost_description.md")
    check_file("Demo script exists", "submission/demo_script.md")
    check_file("Screenshots plan exists", "submission/screenshots_plan.md")
    check_file("Judging alignment exists", "submission/judging_alignment.md")

    # --- AI disclosure ---
    readme_text = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    built_with_text = (ROOT / "submission/built_with.md").read_text() if (ROOT / "submission/built_with.md").exists() else ""
    disclosed = ("AI Assistance Disclosure" in readme_text or "AI ASSISTANCE DISCLOSURE" in readme_text.upper()) and \
                ("claude" in built_with_text.lower() or "chatgpt" in built_with_text.lower() or "ai" in built_with_text.lower())
    RESULTS.append(("AI-assisted development disclosure present", "PASS" if disclosed else "FAIL",
                     "README.md + submission/built_with.md", True))

    # --- Mechanical checks ---
    check_command("Parameter budget: baseline PASS",
                  [PY, "scripts/count_parameters.py", "--config", "configs/model_baseline.yaml"])
    check_command("Parameter budget: NOVA-50 PASS",
                  [PY, "scripts/count_parameters.py", "--config", "configs/model_nova.yaml"])
    check_command("Security scan: no secrets", [PY, "scripts/security_scan.py"])
    check_command("Tests pass (pytest -q)", [PY, "-m", "pytest", "-q"], timeout=300)

    # --- Official benchmark status (honest, non-blocking WARNING if NOT RUN) ---
    check_benchmark_status("Official lm-eval benchmarks (HellaSwag/ARC-Easy/PIQA/WinoGrande)",
                            "results/benchmarks/lm_eval_results.json")
    check_benchmark_status("WikiText-103 perplexity", "results/benchmarks/wikitext103_report.json")

    # --- Report ---
    print("=" * 70)
    print("COMPETITION READINESS REPORT")
    print("=" * 70)
    mandatory_fail = False
    for name, status, detail, mandatory in RESULTS:
        tag = "[MANDATORY]" if mandatory else "[info]     "
        print(f"{status:<8} {tag} {name}")
        if detail:
            print(f"         -> {detail[:120]}")
        if mandatory and status not in ("PASS",):
            mandatory_fail = True

    print("=" * 70)
    if mandatory_fail:
        print("RESULT: One or more MANDATORY requirements FAILED.")
        sys.exit(1)
    else:
        print("RESULT: All mandatory requirements PASS. "
              "Non-mandatory items (official benchmarks) may still be NOT RUN — "
              "see docs/COMPETITION_RULES.md and FINAL_COMPETITION_READINESS.md.")
        sys.exit(0)


if __name__ == "__main__":
    main()
