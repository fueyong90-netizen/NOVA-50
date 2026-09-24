"""Détection matérielle (section 11 du brief) et adaptation raisonnable de la
configuration d'entraînement (batch size, précision, workers)."""
import torch


def detect_hardware() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
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
        precision = "fp32"  # CPU: pas de gain fiable avec fp16/bf16 autocast partout
        recommended_batch_size = 4  # CPU mode: petit batch pour rester rapide
        grad_accum = 8

    return {
        "device": device,
        "precision": precision,
        "recommended_batch_size": recommended_batch_size,
        "recommended_grad_accum": grad_accum,
    }
