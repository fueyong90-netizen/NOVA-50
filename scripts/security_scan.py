#!/usr/bin/env python3
"""Scans the repository for likely secrets (API keys, tokens, passwords,
private keys, .env values). Returns non-zero if anything suspicious is found."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "AWS Access Key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Generic API key assignment": re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-./+]{16,}['\"]"),
    "Password assignment": re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"]{4,}['\"]"),
    "Private key block": re.compile(r"-----BEGIN (RSA|OPENSSH|EC|DSA|PRIVATE) KEY-----"),
    "Slack token": re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
SKIP_SUFFIXES = {".png", ".npy", ".ckpt", ".pt", ".model", ".zip"}
ALLOW_FILES = {".env.example"}  # documented placeholder file, no real secrets


def scan_file(path: Path):
    findings = []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return findings
    for label, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append((label, m.group()[:40]))
    return findings


def main():
    all_findings = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.name in ALLOW_FILES:
            continue
        findings = scan_file(path)
        for label, snippet in findings:
            all_findings.append((str(path.relative_to(ROOT)), label, snippet))

    if all_findings:
        print("SECURITY SCAN: POTENTIAL SECRETS FOUND\n")
        for f, label, snippet in all_findings:
            print(f"  [{label}] {f}: {snippet!r}")
        print(f"\nSTATUS: FAIL ({len(all_findings)} finding(s))")
        sys.exit(1)
    else:
        print("SECURITY SCAN: no obvious secrets found.")
        print("STATUS: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
