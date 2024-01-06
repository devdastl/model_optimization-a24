import matplotlib.pyplot as plt
import pandas as pd

def save_plot(stats):
    for model_id in stats.keys():
        latency = stats[model_id]["latency"]
        duration = stats[model_id]["duration"]
        memory = stats[model_id]["memory"]
        quant = ["AWQ", "GPTQ"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for i, (data, title, ylabel, color) in enumerate(
            zip(
                [latency, duration, memory],
                ["Latency", "Duration", "Memory"],
                ["( ms/token)", "(s)", "(GB)"],
                ["red", "green", "blue"],
            )
        ):
            ax = axs[i]
            a = ax.bar(quant, data, color=color)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.bar_label(a, label_type="edge")

        plt.suptitle(f"Parameters for {model_id} Model")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.text(0.5, 0.00, "Quantization", ha="center", va="center", fontsize=12)

        plt.savefig(f"plot-{model_id}.png", bbox_inches="tight", pad_inches=0)
