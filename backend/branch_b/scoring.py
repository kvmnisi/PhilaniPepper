"""Label scoring shared by Branch B inference (branch.py) and training (train.py).

Every canonical label is scored as a continuation of the shared prompt; a
softmax over the summed label-token log-probabilities gives a probability per
label, so the prediction is always a valid label.
"""

import math

from backend.branch_b.prompt import encode_example


class InvalidScores(RuntimeError):
    """The model produced NaN or infinite label scores."""


def pad_batch(encoded, pad_id, torch):
    """encoded: [(ids, prompt_len)] -> input_ids, attention_mask, target_mask."""
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    target_mask = torch.zeros((len(encoded), width), dtype=torch.float32)
    for row, (ids, prompt_len) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention[row, : len(ids)] = 1
        target_mask[row, prompt_len: len(ids)] = 1.0
    return input_ids, attention, target_mask


def sequence_logprobs(model, encoded, device, pad_id, torch):
    """Summed log-probability of each sequence's label tokens, as Python floats."""
    input_ids, attention, target_mask = (t.to(device) for t in pad_batch(encoded, pad_id, torch))
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    token_lp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    # Select the label positions; never multiply by the mask. Padded positions can
    # hold -inf or NaN, and -inf * 0 = NaN would poison every label's score.
    keep = target_mask[:, 1:] > 0
    token_lp = torch.where(keep, token_lp, torch.zeros_like(token_lp))
    return token_lp.sum(dim=-1).tolist()


def stable_softmax(values):
    """Softmax in plain Python; raises InvalidScores on NaN or infinite input."""
    if not values or not all(math.isfinite(v) for v in values):
        raise InvalidScores("Branch B produced invalid scores (NaN or infinity).")
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _finite(values):
    return all(math.isfinite(v) for v in values)


def score_labels(model, tokenizer, text, labels, device, pad_id, torch):
    """Return {label: probability} for one transcript.

    Scores all labels in one padded batch; if that yields NaN/inf (some model
    code mishandles padding), scores each label on its own without padding.
    Raises InvalidScores if the scores are still invalid, rather than returning
    a meaningless prediction.
    """
    encoded = [encode_example(tokenizer, text, label) for label in labels]
    totals = sequence_logprobs(model, encoded, device, pad_id, torch)
    if not _finite(totals):
        totals = [sequence_logprobs(model, [one], device, pad_id, torch)[0] for one in encoded]
    if not _finite(totals):
        raise InvalidScores("Branch B produced invalid scores (NaN or infinity) for this input. "
                            "Load InkubaLM in float32 (INKUBALM_DTYPE=float32) and check the adapter.")
    return dict(zip(labels, stable_softmax(totals)))
