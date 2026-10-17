import os
import re
import json
import urllib.request
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

GUTENBERG_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
RAW_DATA_PATH = "data/sherlock_holmes.txt"
CLEAN_DATA_PATH = "data/sherlock_holmes_selected.txt"
INFO_PATH = "data/dataset_info.json"

def download_and_prepare_dataset(max_words=45000, max_vocab=2500, min_freq=2):
    """
    Downloads The Adventures of Sherlock Holmes from Project Gutenberg,
    extracts the opening complete detective cases (e.g., 'A Scandal in Bohemia'
    and 'The Red-Headed League'), tokenizes the text, builds vocabulary,
    and saves dataset metadata.
    """
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH) or os.path.getsize(RAW_DATA_PATH) < 10000:
        print(f"Downloading raw text from {GUTENBERG_URL} ...")
        req = urllib.request.Request(GUTENBERG_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8-sig")

        start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", content, re.IGNORECASE)
        end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", content, re.IGNORECASE)

        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(content)
        raw_text = content[start_idx:end_idx].strip()

        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            f.write(raw_text)
    else:
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()

    # Select the opening corpus (~45,000 words covering complete stories)
    all_words = raw_text.split()
    selected_words = all_words[:max_words]
    selected_text = " ".join(selected_words)

    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(selected_text)

    # Tokenize
    tokens = tokenize_text(selected_text)

    vocab = Vocabulary(max_vocab_size=max_vocab, min_freq=min_freq)
    vocab.build_vocab(tokens)

    # Compute dataset characteristics
    char_count = len(selected_text)
    token_count = len(tokens)
    unique_words = len(set(tokens))
    avg_word_length = sum(len(t) for t in tokens if t.isalnum()) / max(1, sum(1 for t in tokens if t.isalnum()))

    info = {
        "dataset_name": "The Adventures of Sherlock Holmes (Selected Stories)",
        "author": "Sir Arthur Conan Doyle",
        "source": "Project Gutenberg (eBook #1661)",
        "source_url": GUTENBERG_URL,
        "total_characters": char_count,
        "total_tokens": token_count,
        "vocabulary_size": len(vocab),
        "raw_unique_tokens": unique_words,
        "average_word_length": round(avg_word_length, 2),
        "stories_included": [
            "I. A Scandal in Bohemia",
            "II. The Red-Headed League",
            "III. A Case of Identity"
        ]
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"Dataset prepared successfully: {token_count} tokens, {len(vocab)} vocab words.")
    return selected_text, vocab, tokens, info

download_dataset = download_and_prepare_dataset



def tokenize_text(text):
    """
    Standard lowercased word and punctuation tokenization.
    Keeps alphanumeric tokens and punctuation (.,!?;:) as distinct items.
    """
    return re.findall(r"\w+|[.,!?;:]", text.lower())


def detokenize(tokens):
    """Reconstructs text from tokens with proper punctuation attachment."""
    punctuation = {".", ",", "!", "?", ";", ":"}
    text = ""
    for tok in tokens:
        if tok in punctuation or text == "" or text.endswith("\n"):
            text += tok
        else:
            text += " " + tok
    return text


class Vocabulary:
    """Vocabulary mapper with special PAD and UNK tokens."""
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size=2500, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, tokens):
        counts = Counter(tokens)
        sorted_words = [w for w, c in counts.most_common() if c >= self.min_freq]

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        vocab_words = special_tokens + sorted_words[: self.max_vocab_size - len(special_tokens)]

        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for i, w in enumerate(vocab_words)}
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]

    def encode(self, tokens):
        return [self.word2idx.get(w, self.unk_idx) for w in tokens]

    def decode(self, indices):
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]

    def __len__(self):
        return len(self.word2idx)


class AutoregressiveTextDataset(Dataset):
    """
    Autoregressive sliding-window dataset.
    Input: [w_t, w_{t+1}, ..., w_{t+seq_len-1}]
    Target: w_{t+seq_len}
    """
    def __init__(self, token_ids, seq_len=12, stride=3):
        self.seq_len = seq_len
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - seq_len, stride):
            self.inputs.append(token_ids[i : i + seq_len])
            self.targets.append(token_ids[i + seq_len])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_data_loaders(seq_len=12, stride=3, batch_size=64, max_vocab=2500, val_split=0.15):
    """Convenience function to get train and validation loaders."""
    selected_text, vocab, tokens, info = download_and_prepare_dataset(max_vocab=max_vocab)
    token_ids = vocab.encode(tokens)

    dataset = AutoregressiveTextDataset(token_ids, seq_len=seq_len, stride=stride)
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab, info
