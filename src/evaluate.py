import math
from collections import Counter
import torch
import torch.nn as nn

def evaluate_model_metrics(model, val_loader, criterion, device="cpu"):
    """
    Computes validation loss, perplexity, top-1 accuracy, and top-5 accuracy.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size

            # Top-1 accuracy
            preds = torch.argmax(logits, dim=-1)
            top1_correct += (preds == y).sum().item()

            # Top-5 accuracy
            _, top5_preds = torch.topk(logits, k=5, dim=-1)
            top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=-1).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20)) # Prevent math overflow
    top1_acc = (top1_correct / total_tokens) * 100.0
    top5_acc = (top5_correct / total_tokens) * 100.0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc
    }


def compute_diversity_metrics(tokens):
    """
    Computes lexical diversity and repetition metrics:
    - distinct_1: ratio of unique unigrams to total unigrams
    - distinct_2: ratio of unique bigrams to total bigrams
    - repetition_rate_3: percentage of 3-grams that appear more than once
    """
    if len(tokens) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}

    # Distinct-1
    distinct_1 = len(set(tokens)) / len(tokens)

    # Distinct-2
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    distinct_2 = (len(set(bigrams)) / len(bigrams)) if len(bigrams) > 0 else 0.0

    # 3-gram Repetition
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if len(trigrams) > 0:
        counts = Counter(trigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        repetition_rate = repeated / len(trigrams)
    else:
        repetition_rate = 0.0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_rate": repetition_rate
    }
