import os
import time
import math
import torch
import torch.nn as nn
from .evaluate import evaluate_model_metrics

def train_recurrent_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=0.002,
    device="cpu",
    save_path=None,
    model_name="Model"
):
    """
    Unified training loop for comparable training across all 4 recurrent architectures.
    Tracks loss, perplexity, top-1 accuracy, top-5 accuracy, and training time per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "model_name": model_name,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "val_top1_acc": [],
        "val_top5_acc": [],
        "epoch_times": [],
        "total_training_time": 0.0,
        "best_val_loss": float("inf"),
        "best_epoch": 0
    }

    print(f"\n{'='*60}")
    print(f"Starting Training: {model_name} ({epochs} Epochs | Device: {device})")
    print(f"{'='*60}")

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in recurrent networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = y_batch.size(0)
            running_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss = running_loss / total_train_samples
        epoch_duration = time.time() - epoch_start

        # Evaluate on validation set
        val_metrics = evaluate_model_metrics(model, val_loader, criterion, device=device)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_perplexity"].append(val_metrics["perplexity"])
        history["val_top1_acc"].append(val_metrics["top1_acc"])
        history["val_top5_acc"].append(val_metrics["top5_acc"])
        history["epoch_times"].append(epoch_duration)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val PPL: {val_metrics['perplexity']:.2f} | "
            f"Val Top-1: {val_metrics['top1_acc']:.2f}% | "
            f"Val Top-5: {val_metrics['top5_acc']:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint saving
        if val_metrics["loss"] < history["best_val_loss"]:
            history["best_val_loss"] = val_metrics["loss"]
            history["best_epoch"] = epoch
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                    "model_name": model_name
                }, save_path)

    total_time = time.time() - total_start_time
    history["total_training_time"] = total_time

    print(f"\nFinished Training {model_name} in {total_time:.2f}s. Best Val Loss: {history['best_val_loss']:.4f} at Epoch {history['best_epoch']}.")
    return history
