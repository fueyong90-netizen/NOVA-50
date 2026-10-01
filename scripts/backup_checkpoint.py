#!/usr/bin/env python3
"""Bundles the current best checkpoint + its config + logs into ONE zip file,
easy to find and download from Kaggle's Output panel (top level of /kaggle/working,
not buried in experiments/.../). Run this by hand every ~30-60 min during a long
training session, and download the resulting zip to your own computer.

Usage: python scripts/backup_checkpoint.py --run_dir experiments/nova/production_run
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="/kaggle/working/NOVA50_CHECKPOINT_BACKUP.zip")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    if not out_path.parent.exists():  # not on Kaggle (e.g. local test) -> write next to the repo
        out_path = Path("NOVA50_CHECKPOINT_BACKUP.zip")

    files = [p for p in [run_dir / "best.ckpt", run_dir / "last.ckpt",
                         run_dir / "run_summary.json", run_dir / "train_log.jsonl",
                         run_dir / "EXPERIMENT_INFO.json"] if p.exists()]
    if not (run_dir / "best.ckpt").exists():
        print(f"!!! No best.ckpt in {run_dir} yet — nothing to back up (training hasn't reached its first eval).")
        return

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f"{run_dir.name}/{f.name}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"Backup written: {out_path} ({size_mb:.1f} MB)")
    print("Download it now from the Output panel (top level of /kaggle/working) — "
          "this file will be ERASED if the Kaggle session restarts, same as everything else.")


if __name__ == "__main__":
    main()
