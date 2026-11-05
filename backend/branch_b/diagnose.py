"""Check that Branch B (InkubaLM) loads and scores on this machine.

    python -m backend.branch_b.diagnose
    INKUBALM_DEVICE=cpu python -m backend.branch_b.diagnose

1. Loads InkubaLM with the backend's settings and checks its outputs are finite.
2. Scores a few sentences and prints the top intents and the time taken.
"""

import sys
import time

SAMPLES = ["Sawubona", "ngicela usizo", "ikuphi i-great hall", "ngitshele isikhathi",
           "vula i student self service", "ngubani ofundisa i-computer vision"]


def main():
    import transformers

    from backend.branch_b.branch import BranchBUnusable, InkubaLMClassifier
    from backend.config import settings

    print("transformers", transformers.__version__)
    print("\n1. Loading %s (device=%s, dtype=%s)" % (settings.inkubalm_model, settings.inkubalm_device,
                                                     settings.inkubalm_dtype))
    classifier = InkubaLMClassifier(settings)
    try:
        classifier.load()
    except BranchBUnusable as error:
        print("   FAILED:", error)
        return 3
    mode = ("LoRA adapter %s" % classifier.adapter) if classifier.adapter else (
        "untrained, %d examples in the prompt" % len(classifier.examples))
    print("   OK on device=%s, dtype=%s, %s" % (classifier.device, classifier.dtype_name, mode))

    print("\n2. Scoring sample sentences")
    for text in SAMPLES:
        start = time.perf_counter()
        try:
            _, scores = classifier.classify(text)
        except Exception as error:  # report and continue
            print("   %-40s ERROR %s" % (text, error))
            continue
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print("   %-40s %s  (%.1f s)" % (text, ", ".join("%s %.2f" % kv for kv in top), time.perf_counter() - start))
    print("\nBranch B is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
