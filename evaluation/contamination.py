#!/usr/bin/env python3
"""Vérifie l'absence de fuite train/test : aucun exemple de test ne doit avoir
le même hash de texte qu'un exemple de train (voir data/scripts/split_and_shard.py)."""
import argparse
import json
import sys
from pathlib import Path


def load_hashes(path):
    hashes = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            hashes.add(rec["hash"])
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    args = ap.parse_args()

    train_h, val_h, test_h = load_hashes(args.train), load_hashes(args.val), load_hashes(args.test)

    leak_train_val = train_h & val_h
    leak_train_test = train_h & test_h
    leak_val_test = val_h & test_h

    print(f"train: {len(train_h)} hash uniques")
    print(f"val:   {len(val_h)} hash uniques")
    print(f"test:  {len(test_h)} hash uniques")
    print(f"Fuite train<->val:  {len(leak_train_val)}")
    print(f"Fuite train<->test: {len(leak_train_test)}")
    print(f"Fuite val<->test:   {len(leak_val_test)}")

    ok = not (leak_train_val or leak_train_test or leak_val_test)
    print("STATUS:", "OK — aucune fuite détectée" if ok else "FAIL — fuite détectée")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
