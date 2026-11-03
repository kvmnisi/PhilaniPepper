"""Label scoring shared by Branch B inference (branch.py) and training (train.py).

Every canonical label is scored as a continuation of the shared prompt; a
softmax over the summed label-token log-probabilities gives a probability per
label, so the prediction is always a valid label.
"""

from backend.branch_b.prompt import encode_example


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


def score_labels(model, tokenizer, text, labels, device, pad_id, torch):
    """Return {label: probability} for one transcript."""
    encoded = [encode_example(tokenizer, text, label) for label in labels]
    input_ids, attention, target_mask = (t.to(device) for t in pad_batch(encoded, pad_id, torch))
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    token_lp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    label_scores = (token_lp * target_mask[:, 1:]).sum(dim=-1)
    return dict(zip(labels, torch.softmax(label_scores, dim=0).tolist()))
