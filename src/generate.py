import torch
from .dataset import tokenize_text, detokenize

def generate_text(
    model,
    vocab,
    seed_text,
    max_new_tokens=30,
    temperature=0.7,
    seq_len=15,
    device="cpu",
    top_k=50
):
    """
    Autoregressively generates next words given a seed text and temperature.

    Args:
        model: Trained PyTorch recurrent model.
        vocab: Vocabulary object.
        seed_text: String prompt to initiate generation.
        max_new_tokens: Number of words to generate.
        temperature: Softmax temperature (> 0). Higher = more random, Lower = more deterministic.
        seq_len: Expected input sequence length.
        device: 'cpu' or 'cuda'.
        top_k: Top-k filtering to prevent sampling rare or nonsensical tokens.

    Returns:
        full_text: String containing seed prompt + generated text.
        generated_tokens: List of newly generated word tokens.
    """
    model.eval()
    model.to(device)

    seed_tokens = tokenize_text(seed_text)
    if len(seed_tokens) == 0:
        seed_tokens = ["the"]

    current_ids = vocab.encode(seed_tokens)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use last seq_len tokens (pad on the left if shorter)
            if len(current_ids) < seq_len:
                padded = [vocab.pad_idx] * (seq_len - len(current_ids)) + current_ids
            else:
                padded = current_ids[-seq_len:]

            x_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            logits, _ = model(x_tensor) # (1, vocab_size)
            logits = logits[0]          # (vocab_size)

            # Avoid sampling special PAD or UNK tokens if possible
            logits[vocab.pad_idx] = -1e9
            logits[vocab.unk_idx] = -1e9

            if temperature <= 0.05:
                # Greedy selection
                next_id = torch.argmax(logits).item()
            else:
                scaled_logits = logits / max(temperature, 1e-5)

                # Optional top-k filtering
                if top_k is not None and top_k > 0 and top_k < len(scaled_logits):
                    top_vals, _ = torch.topk(scaled_logits, top_k)
                    min_val = top_vals[-1]
                    scaled_logits[scaled_logits < min_val] = -1e9

                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

            generated_ids.append(next_id)
            current_ids.append(next_id)

    generated_tokens = vocab.decode(generated_ids)
    all_tokens = seed_tokens + generated_tokens
    full_text = detokenize(all_tokens)

    return full_text, generated_tokens
