#!/usr/bin/env python3
"""
API locale minimale (stdlib uniquement, pas de dépendance à Flask/FastAPI)
pour servir le modèle NOVA-50 à la démo web (web/index.html).

Usage:
    python inference/server.py --checkpoint experiments/nova/toy_run/best.ckpt \
        --model_config configs/model_nova_tiny_demo.yaml --port 8000
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import yaml
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint

STATE = {}


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/info":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(STATE["info"]).encode())
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("prompt", "")
        max_new_tokens = int(body.get("max_new_tokens", 30))
        temperature = float(body.get("temperature", 0.8))
        top_k = int(body.get("top_k", 40))

        sp, model = STATE["sp"], STATE["model"]
        ids = sp.encode(prompt, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long)

        t0 = time.time()
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_k=top_k, eos_token_id=sp.eos_id())
        elapsed = time.time() - t0
        n_new = out.shape[1] - idx.shape[1]
        text = sp.decode(out[0].tolist())

        payload = {
            "text": text,
            "tokens_generated": n_new,
            "seconds": round(elapsed, 4),
            "tokens_per_second": round(n_new / max(elapsed, 1e-6), 1),
            "model": STATE["info"]["model_name"],
            "total_trainable_parameters": STATE["info"]["total_trainable_parameters"],
        }
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, fmt, *args):
        pass  # silence les logs par défaut (déjà loggé côté appelant si besoin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    counts = model.count_parameters()
    STATE["model"] = model
    STATE["sp"] = sp

    def load_benchmark_status(path):
        p = Path(path)
        if not p.exists():
            return {"status": "NOT RUN", "reason": "not executed yet"}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"status": "NOT RUN", "reason": "report unreadable"}

    STATE["info"] = {
        "model_name": "NOVA-50 (demo checkpoint — trained from scratch)",
        "total_trainable_parameters": counts["total_trainable"],
        "budget": 50_000_000,
        "context_length": cfg.max_seq_len,
        "vocab_size": cfg.vocab_size,
        "note": "Demo checkpoint trained on a synthetic toy corpus, for pipeline demonstration.",
        "benchmarks": {
            "lm_eval": load_benchmark_status(ROOT / "results/benchmarks/lm_eval_results.json"),
            "wikitext103": load_benchmark_status(ROOT / "results/benchmarks/wikitext103_report.json"),
        },
    }

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"NOVA-50 inference server at http://localhost:{args.port}  (GET /info, POST /generate)")
    server.serve_forever()


if __name__ == "__main__":
    main()
