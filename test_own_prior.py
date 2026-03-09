import torch
import matplotlib.pyplot as plt
from tfmplayground.priors.dataloader import PriorDataLoader

def get_dummy_scm_batch(batch_size: int, seq_len: int, num_features: int) -> dict:
    x = torch.zeros(batch_size, seq_len, num_features)
    for i in range(num_features):
        x[:, :, i] = torch.randn(batch_size, seq_len)
        if i > 0:
            weights = torch.randn(batch_size, i)
            x[:, :, i] += torch.einsum('bsj,bj->bs', x[:, :, :i], weights)
    
    y_weights = torch.randn(batch_size, num_features)
    y = torch.einsum('bsj,bj->bs', x, y_weights) + 0.1 * torch.randn(batch_size, seq_len)
    
    single_eval_pos = seq_len // 2
    
    return {
        "x": x,
        "y": y,
        "target_y": y,
        "single_eval_pos": single_eval_pos
    }

if __name__ == "__main__":
    device = torch.device("cpu")
    
    dummy_loader = PriorDataLoader(
        get_batch_function=get_dummy_scm_batch,
        num_steps=10,
        batch_size=16,
        num_datapoints_max=100,
        num_features=10,
        device=device
    )

    for step, batch in enumerate(dummy_loader):
        print(f"--- Batch {step + 1} ---")
        print(f"X shape: {batch['x'].shape}")
        print(f"y shape: {batch['y'].shape}")
        print(f"Evaluation position (Train/Test split): {batch['single_eval_pos']}")

        x_np = batch['x'][0].cpu().numpy()
        y_np = batch['y'][0].cpu().numpy()
        num_feats = x_np.shape[1]

        number_of_cols = 4
        number_of_rows = (num_feats + number_of_cols - 1) // number_of_cols
        fig, axes = plt.subplots(number_of_rows, number_of_cols, figsize=(4 * number_of_cols, 4 * number_of_rows))
        axes = axes.flatten()
        for i in range(num_feats):
            axes[i].scatter(x_np[:, i], y_np)
            axes[i].set_xlabel(f"Dimension {i}")
            axes[i].set_ylabel("Output")
            axes[i].set_title(f"Dimension {i} vs Output")

        plt.tight_layout()
        plt.show()
        break