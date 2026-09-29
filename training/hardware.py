"""Détection matérielle (section 11 du brief) et adaptation raisonnable de la
configuration d'entraînement (batch size, précision, workers)."""
import torch


def detect_hardware() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # Native bf16 needs compute capability >= 8.0 (Ampere+). T4 (7.5) and P100 (6.0)
        # only "emulate" bf16, which is slow, so we treat them as fp16 devices.
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
        "cpu_count": torch.get_num_threads(),
    }
    return info


def choose_device_and_precision(info: dict, requested_device: str = "auto"):
    if requested_device == "auto":
        device = "cuda" if info["cuda_available"] else "cpu"
    else:
        device = requested_device

    if device == "cuda":
        precision = "bf16" if info["bf16_supported"] else "fp16"
        recommended_batch_size = 32
        grad_accum = 1
    else:
        precision = "fp32"  # CPU: no reliable gain from fp16/bf16 autocast everywhere
        recommended_batch_size = 4  # CPU mode: small batch to stay responsive
        grad_accum = 8

    # DataLoader workers: never request more than (visible CPUs - 1). On a
    # single-core box (like some sandboxes), forcing >0 workers causes process-spawn
    # overhead that can dwarf actual training time or even appear to hang.
    recommended_num_workers = max(0, min(4, info["cpu_count"] - 1))

    return {
        "device": device,
        "precision": precision,
        "recommended_batch_size": recommended_batch_size,
        "recommended_grad_accum": grad_accum,
        "recommended_num_workers": recommended_num_workers,
    }
