"""Prompt format shared by Branch B training and inference.

The LoRA fine-tune of InkubaLM-0.4B must be trained on exactly this format,
built with `encode_example`:
    build_prompt(<isiZulu transcript>) + " " + <intent label> + <eos>
Inference scores every canonical label as a continuation of the prompt, so
the output is always a valid label and comes with a probability.
"""

PROMPT_TEMPLATE = "### isiZulu command:\n{text}\n### Intent:"


def build_prompt(text):
    return PROMPT_TEMPLATE.format(text=" ".join(text.split()))


def encode_example(tokenizer, text, label):
    """Token ids for prompt + label + eos, and the number of prompt tokens.

    Use this in the LoRA training script too: the loss should cover only the
    tokens after `prompt_len`, and inference scores exactly those tokens.
    """
    prompt = build_prompt(text)
    prompt_len = len(tokenizer(prompt).input_ids)
    ids = tokenizer(prompt + " " + label).input_ids + [tokenizer.eos_token_id]
    return ids, prompt_len
