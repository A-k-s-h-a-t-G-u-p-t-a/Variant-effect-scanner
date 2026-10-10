import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.score import AAS


def plot_scan(matrix, sequence, out_path, start=1, end=None, title=None):
    end = end or len(sequence)
    sub = matrix[start - 1:end].T
    fig, ax = plt.subplots(figsize=(max(12, (end - start) * 0.18), 5))
    sns.heatmap(sub, cmap="RdBu", center=0, vmin=-10, vmax=2,
                yticklabels=AAS,
                xticklabels=[f"{sequence[i]}{i + 1}" for i in range(start - 1, end)],
                cbar_kws={"label": "log-ratio (negative = damaging)"}, ax=ax)
    ax.set_xlabel("position")
    ax.set_ylabel("mutant amino acid")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
