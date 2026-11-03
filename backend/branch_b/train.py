"""Fine-tune InkubaLM-0.4B with LoRA for Branch B intent classification.

Run on a GPU (e.g. Google Colab) from the repository root:

    pip install -r backend/requirements-ml.txt
    python -m backend.branch_b.train \\
        --data data/branch_b_train.csv \\
        --templates data/branch_b_templates.csv \\
        --exclude data/intent_dataset.csv \\
        --out models/inkubalm_intent_lora

Then serve it:  BRANCH_B_ENGINE=inkubalm INKUBALM_ADAPTER=models/inkubalm_intent_lora

Inputs (CSV, UTF-8):
  --data       text,intent[,verified]       one verified isiZulu request per row
  --templates  template,intent[,verified]   requests with placeholders, expanded
               with every value from data/: {course} (course names and aliases),
               {code}, {weekday}, {location}, {contact}. One verified template
               such as a "class" phrasing with {course} covers all 11 courses.
  --exclude    the held-out test set (data/intent_dataset.csv, the recordings):
               any training text matching a test transcript is dropped, so
               Branch B is never trained on what it is tested on.

Only rows with verified = yes are used unless --allow-unverified is given.
Training uses exactly the prompt and label scoring of the live backend
(branch_b/prompt.py, branch_b/scoring.py). Outputs in --out: the LoRA adapter,
metrics.json (validation accuracy / macro-F1 per epoch) and training_info.json.
Validation figures come from held-out training phrasings; report test results
from `python -m backend.evaluation.batch data/intent_dataset.csv`.
"""

import argparse
import csv
import itertools
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from backend.branch_b.prompt import PROMPT_TEMPLATE, encode_example
from backend.evaluation.metrics import classification_report
from backend.nlu.intents import intent_labels
from backend.nlu.slots import normalise_text

PLACEHOLDER = re.compile(r"\{(course|code|weekday|location|contact)\}")
# Standard isiZulu weekday names, as used by the slot extractor (backend/nlu/slots.py).
ZULU_WEEKDAYS = ["uMsombuluko", "uLwesibili", "uLwesithathu", "uLwesine", "uLwesihlanu", "uMgqibelo", "ngeSonto"]
LORA_CANDIDATES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --- data -------------------------------------------------------------------

def _verified(row):
    return (row.get("verified") or "").strip().lower() in ("yes", "y", "true", "1")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def slot_values(timetable):
    """Values substituted into template placeholders, drawn from data/*.json."""
    courses = []
    for course in timetable.courses:
        for alias in [course["name"]] + course.get("aliases", []):
            if alias.lower() not in [c.lower() for c in courses]:
                courses.append(alias)
    locations = []
    for location in timetable.locations:
        locations.extend([location["name"]] + location.get("aliases", [])[:2])
    return {
        "course": courses,
        "code": [c["code"] for c in timetable.courses],
        "weekday": ZULU_WEEKDAYS,
        "location": locations + ["MSL004", "MSL005", "MSL006"],
        "contact": [alias for c in timetable.contacts for alias in c.get("aliases", [])[:2]],
    }


def expand_templates(rows, values, max_per_template=60, seed=13):
    """Fill placeholders; long expansions are sampled so no template dominates."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        template = row["template"].strip()
        names = PLACEHOLDER.findall(template)
        if not names:
            out.append({"text": template, "intent": row["intent"].strip()})
            continue
        combos = list(itertools.product(*[values[name] for name in names]))
        if len(combos) > max_per_template:
            combos = rng.sample(combos, max_per_template)
        for combo in combos:
            text = template
            for name, value in zip(names, combo):
                text = text.replace("{%s}" % name, value, 1)
            out.append({"text": text, "intent": row["intent"].strip()})
    return out


def prepare(data_rows, template_rows, exclude_rows, values, allow_unverified=False):
    """Return (examples, report). Raises ValueError on unknown labels."""
    report = {"skipped_unverified": 0, "dropped_test_overlap": 0, "duplicates": 0}
    rows = []
    for row in data_rows:
        if not allow_unverified and not _verified(row):
            report["skipped_unverified"] += 1
            continue
        rows.append({"text": (row.get("text") or "").strip(), "intent": (row.get("intent") or "").strip()})
    kept = [r for r in template_rows if allow_unverified or _verified(r)]
    report["skipped_unverified"] += len(template_rows) - len(kept)
    rows += expand_templates(kept, values)

    unknown = sorted({r["intent"] for r in rows} - set(intent_labels()))
    if unknown:
        raise ValueError("Unknown intent labels (see data/intents.json): %s" % ", ".join(unknown))

    test_texts = {normalise_text(r.get("reference_transcript") or r.get("text") or "") for r in exclude_rows}
    test_texts.discard("")
    seen, examples = set(), []
    for row in rows:
        key = normalise_text(row["text"])
        if not key:
            continue
        if key in test_texts:
            report["dropped_test_overlap"] += 1
        elif key in seen:
            report["duplicates"] += 1
        else:
            seen.add(key)
            examples.append(row)
    report["examples"] = len(examples)
    report["per_intent"] = dict(sorted(Counter(e["intent"] for e in examples).items()))
    return examples, report


def stratified_split(examples, val_fraction, seed):
    """Per-intent split; intents with fewer than 3 examples stay in training."""
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for example in examples:
        by_intent[example["intent"]].append(example)
    train, val = [], []
    for intent in sorted(by_intent):
        items = by_intent[intent][:]
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction)) if len(items) >= 3 else 0
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train)
    return train, val


def build_features(tokenizer, examples, max_length):
    """Token ids with the loss only on the label tokens (prompt tokens -> -100)."""
    features = []
    for example in examples:
        ids, prompt_len = encode_example(tokenizer, example["text"], example["intent"])
        ids = ids[:max_length]
        labels = [-100] * min(prompt_len, len(ids)) + ids[prompt_len:]
        features.append({"input_ids": ids, "labels": labels[: len(ids)]})
    return features


def collate(batch, pad_id, torch):
    width = max(len(f["input_ids"]) for f in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, f in enumerate(batch):
        n = len(f["input_ids"])
        input_ids[row, :n] = torch.tensor(f["input_ids"])
        labels[row, :n] = torch.tensor(f["labels"])
        attention[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


# --- model ------------------------------------------------------------------

def lora_targets(model):
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [n for n in LORA_CANDIDATES if n in names]
    if not found:
        raise ValueError("No standard projection layers found; pass --target-modules. Seen: %s" % sorted(names)[:40])
    return found


def evaluate(model, tokenizer, examples, device, pad_id, torch):
    from backend.branch_b.scoring import score_labels

    labels = intent_labels()
    model.eval()
    y_true, y_pred = [], []
    for example in examples:
        scores = score_labels(model, tokenizer, example["text"], labels, device, pad_id, torch)
        y_true.append(example["intent"])
        y_pred.append(max(scores, key=scores.get))
    model.train()
    seen = set(y_true) | set(y_pred)
    return classification_report(y_true, y_pred, labels=[l for l in labels if l in seen])


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args, train_set, val_set, report):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    device = pick_device(torch)
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.float32)
    targets = args.target_modules.split(",") if args.target_modules else lora_targets(model)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, target_modules=targets))
    model.print_trainable_parameters()
    model.to(device).train()

    features = build_features(tokenizer, train_set, args.max_length)
    steps_per_epoch = (len(features) + args.batch_size - 1) // args.batch_size
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps_per_epoch * args.epochs),
                                                steps_per_epoch * args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(features)
        started, total = time.time(), 0.0
        for start in range(0, len(features), args.batch_size):
            batch = {k: v.to(device) for k, v in collate(features[start:start + args.batch_size], pad_id, torch).items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        entry = {"epoch": epoch, "train_loss": total / max(1, steps_per_epoch), "seconds": round(time.time() - started, 1)}
        if val_set:
            metrics = evaluate(model, tokenizer, val_set, device, pad_id, torch)
            entry.update(val_accuracy=metrics["accuracy"], val_macro_f1=metrics["macro_f1"])
        history.append(entry)
        print(json.dumps(entry))
        score = entry.get("val_macro_f1", -entry["train_loss"])
        if best is None or score > best:
            best = score
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print("  saved best adapter to", out)

    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    (out / "training_info.json").write_text(json.dumps({
        "base_model": args.base_model, "prompt_template": PROMPT_TEMPLATE, "labels": intent_labels(),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": targets},
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "train_examples": len(train_set), "val_examples": len(val_set), "data_report": report,
        "note": "Validation figures are on held-out training phrasings, not the test recordings.",
    }, indent=2, ensure_ascii=False))
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="CSV with text,intent[,verified]")
    parser.add_argument("--templates", help="CSV with template,intent[,verified]")
    parser.add_argument("--exclude", help="held-out test CSV whose transcripts must not be trained on")
    parser.add_argument("--out", default="models/inkubalm_intent_lora")
    parser.add_argument("--base-model", default="lelapa/InkubaLM-0.4B")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=None, help="comma-separated; default: detected projections")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="prepare and report the data without training")
    args = parser.parse_args(argv)
    if not args.data and not args.templates:
        parser.error("give --data and/or --templates")

    from backend.dispatcher.timetable import get_timetable

    examples, report = prepare(
        read_csv(args.data) if args.data else [],
        read_csv(args.templates) if args.templates else [],
        read_csv(args.exclude) if args.exclude else [],
        slot_values(get_timetable()),
        allow_unverified=args.allow_unverified,
    )
    train_set, val_set = stratified_split(examples, args.val_fraction, args.seed)
    print(json.dumps(dict(report, train=len(train_set), val=len(val_set)), indent=2, ensure_ascii=False))
    thin = [i for i, n in report["per_intent"].items() if n < 5]
    if thin:
        print("Warning: fewer than 5 examples for: " + ", ".join(thin), file=sys.stderr)
    missing = sorted(set(intent_labels()) - set(report["per_intent"]))
    if missing:
        print("Warning: no examples at all for: " + ", ".join(missing), file=sys.stderr)
    if args.dry_run:
        return 0
    if not train_set:
        print("No training examples.", file=sys.stderr)
        return 1
    train(args, train_set, val_set, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
