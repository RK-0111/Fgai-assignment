import os
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_loss_curves(histories, save_path="outputs/plots/loss_comparison.png"):
    """Plots training and validation loss for all 4 models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "Simple RNN": "#e74c3c",
        "Bidirectional RNN": "#3498db",
        "LSTM": "#2ecc71",
        "GRU": "#9b59b6"
    }

    for name, hist in histories.items():
        c = colors.get(name, "black")
        ax1.plot(hist["epochs"], hist["train_loss"], label=name, color=c, linewidth=2, marker='o', markersize=4)
        ax2.plot(hist["epochs"], hist["val_loss"], label=name, color=c, linewidth=2, marker='s', markersize=4)

    ax1.set_title("Training Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Validation Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curves saved to {save_path}")


def plot_perplexity_curves(histories, save_path="outputs/plots/perplexity_comparison.png"):
    """Plots validation perplexity comparison across epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    colors = {
        "Simple RNN": "#e74c3c",
        "Bidirectional RNN": "#3498db",
        "LSTM": "#2ecc71",
        "GRU": "#9b59b6"
    }

    for name, hist in histories.items():
        c = colors.get(name, "black")
        plt.plot(hist["epochs"], hist["val_perplexity"], label=name, color=c, linewidth=2.2, marker='o', markersize=4)

    plt.title("Validation Perplexity vs. Epochs (Lower is Better)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Perplexity (exp(Loss))", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Perplexity curves saved to {save_path}")


def plot_metrics_bar(summary_df, save_path="outputs/plots/metrics_bar_chart.png"):
    """Bar chart comparing quantitative metrics across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    models = summary_df["Model"].tolist()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    # 1. Validation Perplexity (Lower is better)
    axes[0, 0].bar(models, summary_df["Val Perplexity"], color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("Validation Perplexity (Lower is Better)", fontweight="bold")
    axes[0, 0].set_ylabel("PPL")
    for i, v in enumerate(summary_df["Val Perplexity"]):
        axes[0, 0].text(i, v + max(summary_df["Val Perplexity"])*0.02, f"{v:.1f}", ha="center", fontweight="bold")

    # 2. Validation Top-1 Accuracy (%)
    axes[0, 1].bar(models, summary_df["Val Top-1 Acc (%)"], color=colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_title("Validation Top-1 Accuracy % (Higher is Better)", fontweight="bold")
    axes[0, 1].set_ylabel("Top-1 Accuracy (%)")
    for i, v in enumerate(summary_df["Val Top-1 Acc (%)"]):
        axes[0, 1].text(i, v + 0.3, f"{v:.2f}%", ha="center", fontweight="bold")

    # 3. Validation Top-5 Accuracy (%)
    axes[1, 0].bar(models, summary_df["Val Top-5 Acc (%)"], color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Validation Top-5 Accuracy % (Higher is Better)", fontweight="bold")
    axes[1, 0].set_ylabel("Top-5 Accuracy (%)")
    for i, v in enumerate(summary_df["Val Top-5 Acc (%)"]):
        axes[1, 0].text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

    # 4. Total Training Time (s)
    axes[1, 1].bar(models, summary_df["Training Time (s)"], color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Training Time (Seconds)", fontweight="bold")
    axes[1, 1].set_ylabel("Seconds")
    for i, v in enumerate(summary_df["Training Time (s)"]):
        axes[1, 1].text(i, v + 0.5, f"{v:.1f}s", ha="center", fontweight="bold")

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Metrics bar chart saved to {save_path}")


def plot_temperature_analysis(temp_df, save_path="outputs/plots/temperature_diversity_chart.png"):
    """Visualizes the effect of temperature on Distinct-1 and Repetition Rate."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    models = temp_df["Model"].unique()
    colors = {
        "Simple RNN": "#e74c3c",
        "Bidirectional RNN": "#3498db",
        "LSTM": "#2ecc71",
        "GRU": "#9b59b6"
    }

    for m in models:
        sub = temp_df[temp_df["Model"] == m].sort_values("Temperature")
        c = colors.get(m, "black")
        ax1.plot(sub["Temperature"], sub["Distinct-1"], marker="o", label=m, color=c, linewidth=2)
        ax2.plot(sub["Temperature"], sub["Repetition Rate"], marker="s", label=m, color=c, linewidth=2)

    ax1.set_title("Lexical Diversity (Distinct-1) vs. Temperature", fontweight="bold")
    ax1.set_xlabel("Sampling Temperature")
    ax1.set_ylabel("Distinct-1 (Unique / Total Tokens)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Repetition Rate vs. Temperature (Lower is Better)", fontweight="bold")
    ax2.set_xlabel("Sampling Temperature")
    ax2.set_ylabel("Trigram Repetition Rate")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Temperature diversity chart saved to {save_path}")
