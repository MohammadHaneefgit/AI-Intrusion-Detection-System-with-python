# eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(train_df, test_df, out_dir="outputs/eda"):
    # Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Print dataset shapes
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # -----------------------------------------------------------
    # SAVE SUMMARY STATS (CSV + MARKDOWN TABLE)
    # -----------------------------------------------------------
    summary = train_df.describe().transpose()

    # Save CSV
    csv_path = os.path.join(out_dir, "train_summary_stats.csv")
    summary.to_csv(csv_path)
    print("Saved train_summary_stats.csv at:", os.path.abspath(csv_path))

    # Save Markdown manually
    md_path = os.path.join(out_dir, "train_summary_stats.md")
    with open(md_path, "w") as f:
        f.write("# Train Summary Statistics (Readable Table)\n\n")
        f.write("| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
        f.write("|---------|-------|------|-----|-----|-----|-----|-----|-----|\n")

        for feature, row in summary.iterrows():
            f.write(
                f"| {feature} | "
                f"{row['count']:.2f} | "
                f"{row['mean']:.4f} | "
                f"{row['std']:.4f} | "
                f"{row['min']:.4f} | "
                f"{row['25%']:.4f} | "
                f"{row['50%']:.4f} | "
                f"{row['75%']:.4f} | "
                f"{row['max']:.4f} |\n"
            )

    print("Saved train_summary_stats.md at:", os.path.abspath(md_path))

    # -----------------------------------------------------------
    # CLASS DISTRIBUTION (TRAIN VS TEST)
    # -----------------------------------------------------------
    if "label" in train_df.columns:

        train_pct = train_df["label"].value_counts(normalize=True).sort_index() * 100
        test_pct = test_df["label"].value_counts(normalize=True).sort_index() * 100

        train_pct = train_pct.reindex([0, 1], fill_value=0)
        test_pct = test_pct.reindex([0, 1], fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#4CAF50", "#F44336"]  # green = normal, red = attack

        # Train
        sns.barplot(
            x=train_pct.index,
            y=train_pct.values,
            hue=train_pct.index,
            palette=colors,
            legend=False,
            ax=axes[0]
        )
        axes[0].set_title("TRAIN SET: Normal (0) vs Attack (1) – Percentage")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Percentage (%)")

        for i, v in enumerate(train_pct.values):
            axes[0].text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=10)

        # Test
        sns.barplot(
            x=test_pct.index,
            y=test_pct.values,
            hue=test_pct.index,
            palette=colors,
            legend=False,
            ax=axes[1]
        )
        axes[1].set_title("TEST SET: Normal (0) vs Attack (1) – Percentage")
        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Percentage (%)")

        for i, v in enumerate(test_pct.values):
            axes[1].text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "class_counts.png"))
        plt.close()
        print("Saved class distribution chart.")

    # -----------------------------------------------------------
    # ATTACK TYPE DISTRIBUTION
    # -----------------------------------------------------------
    if "attack_type" in train_df.columns:

        attack_pct = train_df["attack_type"].value_counts(normalize=True) * 100

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=attack_pct.index,
            y=attack_pct.values,
            hue=attack_pct.index,
            palette="viridis",
            legend=False
        )

        plt.xticks(rotation=45, ha="right")
        plt.title("Attack Type Distribution (Training Set) – Percentage")
        plt.xlabel("Attack Type")
        plt.ylabel("Percentage (%)")

        for i, v in enumerate(attack_pct.values):
            plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "attack_type_distribution.png"))
        plt.close()
        print("Saved attack type distribution chart.")

    # -----------------------------------------------------------
    # CORRELATION HEATMAP (FIRST 25 NUMERIC FEATURES)
    # -----------------------------------------------------------
    numeric = train_df.select_dtypes(include=["number"])
    cols = numeric.columns[:25]
    corr = numeric[cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (subset of 25 features)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()
    print("Saved correlation heatmap.")
