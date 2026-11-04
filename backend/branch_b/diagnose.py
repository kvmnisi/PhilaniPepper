"""Find out why Branch B produces NaN/infinite scores.

    python -m backend.branch_b.diagnose                 # uses ./InkubaLMAdapter or INKUBALM_ADAPTER
    INKUBALM_DEVICE=cpu python -m backend.branch_b.diagnose

1. Reads the adapter file directly (no base model needed) and counts tensors
   containing NaN or infinity. Damaged weights mean the adapter must be retrained.
2. Loads InkubaLM + adapter with the backend's settings; the load-time check
   says whether the adapter, or the base model on this device/precision, is at fault.
3. Scores a few sentences and prints the top intents.
"""

import json
import sys
from pathlib import Path

SAMPLES = ["Sawubona", "ngicela usizo", "ikuphi i-great hall", "ngitshele isikhathi"]


def adapter_weight_report(adapter_dir):
    """{tensor name: problem} for tensors containing NaN or inf; raises if no weights file."""
    import torch

    folder = Path(adapter_dir)
    if (folder / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        tensors = load_file(str(folder / "adapter_model.safetensors"))
    elif (folder / "adapter_model.bin").exists():
        tensors = torch.load(str(folder / "adapter_model.bin"), map_location="cpu")
    else:
        raise FileNotFoundError("No adapter_model.safetensors or adapter_model.bin in %s" % folder)
    problems = {}
    for name, tensor in tensors.items():
        t = tensor.float()
        if torch.isnan(t).any():
            problems[name] = "NaN"
        elif torch.isinf(t).any():
            problems[name] = "infinity"
    return problems, len(tensors)


def main():
    from backend.branch_b.branch import BranchBUnusable, InkubaLMClassifier
    from backend.config import settings

    adapter = settings.inkubalm_adapter
    if not adapter:
        print("No adapter found: put it in ./InkubaLMAdapter or set INKUBALM_ADAPTER.")
        return 1
    print("Adapter:", adapter)
    info_path = Path(adapter) / "training_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        losses = [h.get("train_loss") for h in info.get("history", [])]
        print("Training losses per epoch:", losses)
        if any(isinstance(l, str) or (isinstance(l, float) and l != l) for l in losses):
            print("  -> training loss was NaN: the adapter was damaged during training.")

    print("\n1. Adapter weights")
    problems, total = adapter_weight_report(adapter)
    if problems:
        print("   DAMAGED: %d of %d tensors contain NaN/inf, e.g. %s" % (len(problems), total, next(iter(problems))))
        print("   Retrain the adapter with the latest Colab notebook. Nothing else will fix this.")
        return 2
    print("   OK: all %d tensors are finite" % total)

    print("\n2. Loading InkubaLM + adapter (device=%s, dtype=%s)" % (settings.inkubalm_device, settings.inkubalm_dtype))
    classifier = InkubaLMClassifier(settings)
    try:
        classifier.load()
    except BranchBUnusable as error:
        print("   FAILED:", error)
        return 3
    print("   OK on device=%s, dtype=%s" % (classifier.device, classifier.dtype_name))

    print("\n3. Scoring sample sentences")
    for text in SAMPLES:
        try:
            _, scores = classifier.classify(text)
        except Exception as error:  # report and continue
            print("   %-24s ERROR %s" % (text, error))
            continue
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print("   %-24s %s" % (text, ", ".join("%s %.2f" % kv for kv in top)))
    print("\nBranch B is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
