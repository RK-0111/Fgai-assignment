import os
import json
import torch
import pandas as pd

from src.dataset import get_data_loaders
from models.recurrent_models import create_all_models, count_parameters
from src.train import train_recurrent_model
from src.generate import generate_text
from src.evaluate import compute_diversity_metrics
from src.plots import (
    plot_loss_curves,
    plot_perplexity_curves,
    plot_metrics_bar,
    plot_temperature_analysis
)

def run_experiment(epochs=8, lr=0.002, batch_size=64, seq_len=12, stride=3):
    print("\n" + "="*70)
    print("Foundations of Generative AI (ICT4403) - Assignment 2")
    print("Comparative Autoregressive Text Generation: Simple RNN vs BiRNN vs LSTM vs GRU")
    print("="*70 + "\n")

    os.makedirs("outputs/saved_models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}\n")

    # 1. Dataset Loading
    print("1. Preparing and Tokenizing Dataset...")
    train_loader, val_loader, vocab, dataset_info = get_data_loaders(
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        max_vocab=2500
    )
    vocab_size = len(vocab)
    print(f"Dataset: {dataset_info['dataset_name']}")
    print(f"Vocabulary Size: {vocab_size} | Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}\n")

    # 2. Instantiate Models
    print("2. Instantiating Models under Comparable Experimental Conditions...")
    models = create_all_models(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.2)

    param_counts = {name: count_parameters(m) for name, m in models.items()}
    for name, p_count in param_counts.items():
        print(f" - {name:<18}: {p_count:,} trainable parameters")

    # 3. Train all Models
    print("\n3. Training All Models under Comparable Experimental Setup...")
    histories = {}
    summary_rows = []

    for name, model in models.items():
        save_path = f"outputs/saved_models/{name.lower().replace(' ', '_')}.pt"
        hist = train_recurrent_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            save_path=save_path,
            model_name=name
        )
        histories[name] = hist

        # Final best metrics
        best_epoch = hist["best_epoch"] - 1
        summary_rows.append({
            "Model": name,
            "Parameters": param_counts[name],
            "Train Loss": round(hist["train_loss"][best_epoch], 4),
            "Val Loss": round(hist["val_loss"][best_epoch], 4),
            "Val Perplexity": round(hist["val_perplexity"][best_epoch], 2),
            "Val Top-1 Acc (%)": round(hist["val_top1_acc"][best_epoch], 2),
            "Val Top-5 Acc (%)": round(hist["val_top5_acc"][best_epoch], 2),
            "Training Time (s)": round(hist["total_training_time"], 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("outputs/metrics_summary.csv", index=False)
    print("\n" + "="*70)
    print("QUANTITATIVE EVALUATION SUMMARY TABLE:")
    print("="*70)
    print(summary_df.to_string(index=False))

    # 4. Generate Visualizations
    print("\n4. Generating Comparative Training Plots...")
    plot_loss_curves(histories, "outputs/plots/loss_comparison.png")
    plot_perplexity_curves(histories, "outputs/plots/perplexity_comparison.png")
    plot_metrics_bar(summary_df, "outputs/plots/metrics_bar_chart.png")

    # 5. Temperature Sampling and Qualitative Generation Analysis
    print("\n5. Running Autoregressive Generation across Temperatures (T = 0.3, 0.7, 1.2)...")
    seed_prompts = [
        "it was a dark and",
        "holmes looked at the",
        "i have no doubt that"
    ]
    temperatures = [0.3, 0.7, 1.2]

    generation_records = []
    temp_analysis_rows = []

    for name, model in models.items():
        # Load best weights
        save_path = f"outputs/saved_models/{name.lower().replace(' ', '_')}.pt"
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        for temp in temperatures:
            distinct_1_list = []
            distinct_2_list = []
            rep_list = []

            for seed in seed_prompts:
                generated_text, tokens = generate_text(
                    model=model,
                    vocab=vocab,
                    seed_text=seed,
                    max_new_tokens=25,
                    temperature=temp,
                    seq_len=seq_len,
                    device=device
                )

                div_metrics = compute_diversity_metrics(tokens)
                distinct_1_list.append(div_metrics["distinct_1"])
                distinct_2_list.append(div_metrics["distinct_2"])
                rep_list.append(div_metrics["repetition_rate"])

                generation_records.append({
                    "model": name,
                    "temperature": temp,
                    "seed": seed,
                    "generated_text": generated_text,
                    "distinct_1": round(div_metrics["distinct_1"], 3),
                    "distinct_2": round(div_metrics["distinct_2"], 3),
                    "repetition_rate": round(div_metrics["repetition_rate"], 3)
                })

            avg_d1 = sum(distinct_1_list) / len(distinct_1_list)
            avg_d2 = sum(distinct_2_list) / len(distinct_2_list)
            avg_rep = sum(rep_list) / len(rep_list)

            temp_analysis_rows.append({
                "Model": name,
                "Temperature": temp,
                "Distinct-1": round(avg_d1, 3),
                "Distinct-2": round(avg_d2, 3),
                "Repetition Rate": round(avg_rep, 3)
            })

    # Save generation records and temperature summary
    with open("outputs/generated_samples.json", "w", encoding="utf-8") as f:
        json.dump(generation_records, f, indent=4)

    temp_df = pd.DataFrame(temp_analysis_rows)
    temp_df.to_csv("outputs/temperature_analysis.csv", index=False)

    plot_temperature_analysis(temp_df, "outputs/plots/temperature_diversity_chart.png")

    print("\n" + "="*70)
    print("TEMPERATURE SAMPLING DIVERSITY & REPETITION ANALYSIS:")
    print("="*70)
    print(temp_df.to_string(index=False))

    print("\nGenerated samples saved to outputs/generated_samples.json")
    print("Pipeline execution completed successfully!\n")

if __name__ == "__main__":
    run_experiment(epochs=8, lr=0.002, batch_size=64, seq_len=12, stride=3)
