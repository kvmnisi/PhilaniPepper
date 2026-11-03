"""Small helpers shared by the model-backed engines. Imported lazily."""


def ml_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def pick_device(preference="auto"):
    import torch

    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    import torch

    return torch.float16 if device == "cuda" else torch.float32
