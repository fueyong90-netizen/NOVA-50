#!/usr/bin/env python3
"""Puts a checkpoint backup (made by scripts/backup_checkpoint.py on a previous,
now-dead Kaggle session) back in place so training/train.py's `resume: true`
picks it up and continues from the right step -- instead of restarting at step 0.

Usage (after re-uploading NOVA50_CHECKPOINT_BACKUP.zip as a Kaggle Input):
  python scripts/restore_checkpoint.py \
      --backup_zip /kaggle/input/my-backup/NOVA50_CHECKPOINT_BACKUP.zip \
      --run_dir experiments/nova/production_run
"""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup_zip", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.backup_zip) as z:
        names = z.namelist()
        for n in names:
            target = run_dir / Path(n).name
            with z.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"Restored into {run_dir}: {[Path(n).name for n in names]}")

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        print(f"This checkpoint was at step {s.get('steps_completed')}/{s.get('steps_configured')}, "
              f"val_loss={s.get('best_val_loss')}. Re-running training/train.py with resume: true "
              f"on the SAME model_config/train_shard will continue from there.")
    else:
        print("No run_summary.json in this backup — training/train.py will still resume from "
              "last.ckpt's step count, just without the extra context.")


if __name__ == "__main__":
    main()
