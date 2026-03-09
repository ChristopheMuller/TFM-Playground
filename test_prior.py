#%%

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tfmplayground.priors.dataloader import TabICLPriorDataLoader

def plot_classification_samples(X_data, y_data, title, batch_size=4):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(min(batch_size, 4)):
        ax = axes[i]
        X_i = X_data[i, :, :2]
        y_i = y_data[i, :].astype(int)
        unique_classes = np.unique(y_i)
        for class_idx in unique_classes:
            mask = (y_i == class_idx)
            if np.any(mask):
                color = colors[class_idx % len(colors)]
                count = np.sum(mask)
                ax.scatter(X_i[mask, 0], X_i[mask, 1],
                          color=color, alpha=0.8, s=40,
                          label=f'C{class_idx} ({count})')
                centroid = np.mean(X_i[mask, :2], axis=0)
                ax.scatter(centroid[0], centroid[1],
                          color=color, s=100, marker='X', alpha=1.0)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"Synthetic Sample {i + 1} ({len(unique_classes)} classes)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    for i in range(batch_size, len(axes)):
        fig.delaxes(axes[i])
    plt.suptitle(f"{title}", fontsize=16)
    plt.tight_layout()
    plt.show()

def explore_priors(priors):
    for prior_type in priors:
        try:
            loader = TabICLPriorDataLoader(
                num_steps=1,
                batch_size=4,
                num_datapoints_min=10,
                num_datapoints_max=50,
                min_features=2,
                max_features=2,
                max_num_classes=3,
                device=torch.device('cpu'),
                prior_type=prior_type
            )
            batch = next(iter(loader))
            X_data = batch["x"].cpu().numpy()
            y_data = batch["y"].cpu().numpy()
            print(f"Generated X shape: {X_data.shape}")
            print(f"Generated y shape: {y_data.shape}")
            plot_classification_samples(X_data, y_data, title=f"TabICL Prior Generation ({prior_type})")
        except Exception as e:
            print(f"Could not load prior {prior_type}: {e}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--priors', nargs='+', default=['mix_scm'])
    args = parser.parse_args()
    explore_priors(args.priors)