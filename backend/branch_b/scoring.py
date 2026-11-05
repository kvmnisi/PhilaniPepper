"""Branch B label scoring with InkubaLM.

The prompt (prompt.py) shows InkubaLM labelled requests and ends with the new
request followed by "Intent:". Every intent has a label word; the model's
log-probability of each label word coming next, compared across all intents,
gives a probability per intent. The prediction is therefore always a valid label.

Fast path: if the label words start with different tokens, one forward pass
scores every intent. Otherwise each label word is scored as a full continuation.

Calibration (Zhao et al., 2021, "Calibrate Before Use"): the same prompt with a
content-free request ("N/A", "") measures how much the model prefers some
label words whatever the request says. Scores are divided by that baseline.
"""

import math

from backend.branch_b.prompt import build_prompt, encode_label

CONTENT_FREE = ("N/A", "")


class InvalidScores(RuntimeError):
    """The model produced NaN or infinite label scores."""


def _finite(values):
    return bool(values) and all(math.isfinite(v) for v in values)


def stable_softmax(values):
    """Softmax in plain Python; raises InvalidScores on NaN or infinite input."""
    if not _finite(values):
        raise InvalidScores("Branch B produced invalid scores (NaN or infinity).")
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def log_normalise(values):
    """log(softmax(values)), computed stably."""
    if not _finite(values):
        raise InvalidScores("Branch B produced invalid scores (NaN or infinity).")
    top = max(values)
    lse = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - lse for v in values]


# --- model calls ---------------------------------------------------------------

def forward_logits(model, input_ids):
    """Logits without a KV cache (InkubaLM's own model code differs across transformers versions)."""
    try:
        return model(input_ids=input_ids, use_cache=False).logits
    except TypeError:
        return model(input_ids=input_ids).logits


def next_token_logprobs(model, ids, token_ids, device, torch):
    """Log-probability of each of `token_ids` as the token after `ids` (one sequence, no padding)."""
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = forward_logits(model, input_ids)[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return [float(log_probs[t]) for t in token_ids]


def pad_batch(encoded, pad_id, torch):
    """encoded: [(ids, prompt_len)] -> right-padded input_ids and a mask over the continuation tokens."""
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    target_mask = torch.zeros((len(encoded), width), dtype=torch.float32)
    for row, (ids, prompt_len) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        target_mask[row, prompt_len: len(ids)] = 1.0
    return input_ids, target_mask


def sequence_logprobs(model, encoded, device, pad_id, torch):
    """Summed log-probability of each sequence's continuation tokens.

    Sequences are right-padded and the model is causal, so padding comes after
    every scored token and cannot change the scores; no attention mask is passed.
    """
    input_ids, target_mask = (t.to(device) for t in pad_batch(encoded, pad_id, torch))
    with torch.inference_mode():
        logits = forward_logits(model, input_ids).float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    token_lp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    # Select the scored positions; never multiply by the mask (-inf * 0 = NaN).
    keep = target_mask[:, 1:] > 0
    token_lp = torch.where(keep, token_lp, torch.zeros_like(token_lp))
    return token_lp.sum(dim=-1).tolist()


# --- planning --------------------------------------------------------------------

def continuations(tokenizer, prompt, words):
    """Prompt token ids, and the token ids each word adds after it (words that do not
    tokenise cleanly after the prompt are skipped)."""
    base = list(tokenizer(prompt).input_ids)
    tails = []
    for word in words:
        full = list(tokenizer(prompt + " " + word).input_ids)
        if len(full) > len(base) and full[: len(base)] == base:
            tails.append(tuple(full[len(base):]))
    return base, tails


def _common_prefix(seqs):
    prefix = []
    for column in zip(*seqs):
        if len(set(column)) != 1:
            break
        prefix.append(column[0])
    return tuple(prefix)


def single_step_plan(tails_by_label):
    """(lead, [one token per label]) when a single next token tells every intent apart.

    `lead` holds tokens every label word starts with (e.g. a lone space token);
    they are appended to the prompt. Returns None when no such token exists.
    """
    if not tails_by_label or any(not tails for tails in tails_by_label):
        return None
    firsts = [tails[0] for tails in tails_by_label]
    lead = _common_prefix(firsts)[: min(len(t) for t in firsts) - 1]
    used, tokens = set(), []
    for tails in tails_by_label:
        pick = next((t[len(lead)] for t in tails
                     if len(t) > len(lead) and t[: len(lead)] == lead and t[len(lead)] not in used), None)
        if pick is None:
            return None
        used.add(pick)
        tokens.append(pick)
    return lead, tokens


# --- scorer ----------------------------------------------------------------------

class LabelScorer:
    def __init__(self, model, tokenizer, labels, label_words, examples, device, pad_id, torch,
                 calibrate=True, max_tokens=None):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.labels = list(labels)
        self.label_words = label_words
        self.examples = list(examples)
        self.device, self.pad_id = device, pad_id
        self.calibrate = calibrate
        self.max_tokens = max_tokens
        self._baselines = {}

    def _prompt(self, text):
        """The prompt for `text`, dropping example rounds from the front if it is too long."""
        examples = self.examples
        while True:
            prompt = build_prompt(text, examples)
            tails = []
            for label in self.labels:
                base, label_tails = continuations(self.tokenizer, prompt, self.label_words[label])
                if not label_tails:
                    raise RuntimeError("Branch B label words for %s cannot be tokenised after the prompt." % label)
                tails.append(label_tails)
            longest = len(base) + max(len(t[0]) for t in tails)
            if not self.max_tokens or longest <= self.max_tokens or not examples:
                return base, tails
            examples = examples[len(self.labels):]

    def raw(self, text):
        """(method key, log-probabilities normalised over the intents)."""
        base, tails = self._prompt(text)
        plan = single_step_plan(tails)
        if plan:
            lead, tokens = plan
            values = next_token_logprobs(self.model, base + list(lead), tokens, self.device, self.torch)
            key = ("next", lead, tuple(tokens))
        else:
            encoded = [(base + list(t[0]), len(base)) for t in tails]
            values = sequence_logprobs(self.model, encoded, self.device, self.pad_id, self.torch)
            if not _finite(values):   # retry one at a time, without padding
                values = [sequence_logprobs(self.model, [one], self.device, self.pad_id, self.torch)[0]
                          for one in encoded]
            key = ("full", tuple(t[0] for t in tails))
        if not _finite(values):
            raise InvalidScores("Branch B produced invalid scores (NaN or infinity). "
                                "Load InkubaLM in float32 (INKUBALM_DTYPE=float32) on INKUBALM_DEVICE=cpu.")
        return key, log_normalise(values)

    def baseline(self, key):
        """Mean intent probabilities for content-free requests, as logs (None if unavailable)."""
        if key not in self._baselines:
            runs = [values for k, values in (self.raw(cf) for cf in CONTENT_FREE) if k == key]
            if runs:
                mean = [sum(math.exp(v[i]) for v in runs) / len(runs) for i in range(len(self.labels))]
                self._baselines[key] = [math.log(max(p, 1e-12)) for p in mean]
            else:
                self._baselines[key] = None
        return self._baselines[key]

    def score(self, text):
        """{intent: probability} for one transcript."""
        key, values = self.raw(text)
        if self.calibrate:
            base = self.baseline(key)
            if base:
                values = [v - b for v, b in zip(values, base)]
        return dict(zip(self.labels, stable_softmax(values)))


class FineTunedScorer:
    """Scores every canonical label (all its tokens and eos) as the continuation of
    the fine-tune prompt; a softmax over the labels gives the probabilities."""

    def __init__(self, model, tokenizer, labels, device, pad_id, eos_id, torch):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.labels = list(labels)
        self.device, self.pad_id, self.eos_id = device, pad_id, eos_id

    def score(self, text):
        encoded = [encode_label(self.tokenizer, text, label, self.eos_id) for label in self.labels]
        totals = sequence_logprobs(self.model, encoded, self.device, self.pad_id, self.torch)
        if not _finite(totals):   # retry one at a time, without padding
            totals = [sequence_logprobs(self.model, [one], self.device, self.pad_id, self.torch)[0] for one in encoded]
        if not _finite(totals):
            raise InvalidScores("Branch B produced invalid scores (NaN or infinity). "
                                "Use transformers 4.46.x and INKUBALM_DTYPE=float32.")
        return dict(zip(self.labels, stable_softmax(totals)))
