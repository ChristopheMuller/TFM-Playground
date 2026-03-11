#%%

import numpy as np
import matplotlib.pyplot as plt
from tabicl import TabICLRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_dag(n_nodes, edge_prob=0.3):
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes):
        parent = np.random.randint(0, i)
        adj[parent, i] = 1
        for j in range(i):
            if np.random.rand() < edge_prob:
                adj[j, i] = 1
    
    weights = np.random.uniform(0.5, 1.5, size=(n_nodes, n_nodes)) * (np.random.choice([-1, 1], size=(n_nodes, n_nodes)))
    return adj, weights

def sample_scm(adj, weights, n_samples, intervention=None):
    n_nodes = adj.shape[0]
    data = np.zeros((n_samples, n_nodes))
    for i in range(n_nodes):
        if intervention is not None and i in intervention:
            data[:, i] = intervention[i]
        else:
            parents = np.where(adj[:, i])[0]
            if len(parents) > 0:
                parent_data = data[:, parents]
                linear_comb = parent_data @ weights[parents, i]
                nonlinear_comb = np.tanh(parent_data) @ (weights[parents, i] * 0.5)
                data[:, i] = linear_comb + nonlinear_comb + np.random.normal(0, 0.1, n_samples)
            else:
                data[:, i] = np.random.normal(0, 1.0, n_samples)
    return data

def plot_dag(adj, seen_indices=None, unseen_index=None, title="Causal DAG"):
    n_nodes = adj.shape[0]
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n_nodes):
        color = 'lightgreen'
        if i == unseen_index:
            color = 'crimson'
        elif seen_indices is not None and i in seen_indices:
            color = 'gold'
            
        ax.scatter(*pos[i], s=1200, c=color, edgecolors='black', zorder=2)
        ax.text(pos[i][0], pos[i][1], f"X{i}", ha='center', va='center', fontweight='bold')
        
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj[i, j]:
                ax.annotate("", xy=pos[j], xytext=pos[i],
                            arrowprops=dict(arrowstyle="-|>", lw=2, color='black', 
                                          shrinkA=25, shrinkB=25, connectionstyle="arc3,rad=0.1"))
    
    if seen_indices is not None or unseen_index is not None:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Observational', markerfacecolor='lightgreen', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Possible Seen Interv.', markerfacecolor='gold', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Target Interv. (Pred)', markerfacecolor='crimson', markersize=10, markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title)
    ax.set_axis_off()
    plt.show()

def add_interventional_flags(X, intervention_dict, n_nodes):
    n_features = n_nodes - 1
    flags = np.zeros((X.shape[0], n_features))
    if intervention_dict:
        for node_idx in intervention_dict:
            if node_idx < n_features:
                flags[:, node_idx] = 1.0
    return np.hstack([X, flags])

#%%
# shared parameters
n_nodes = 7
context_size = 800
test_size = 200
target_idx = n_nodes - 1
feature_indices = list(range(n_nodes - 1))

# generate shared DAG
adj, weights = generate_dag(n_nodes)
plot_dag(adj, title="Initial Causal DAG")

#%% 
# Experiment 1: Varying Ratio of a Single Intervention in Context
print("--- Experiment 1: Impact of Single Intervention Ratio ---")
interv_idx_1 = np.random.randint(0, n_nodes - 1)
interv_val_1 = 2.5

test_data_exp1 = sample_scm(adj, weights, test_size, intervention={interv_idx_1: interv_val_1})
X_test_exp1 = add_interventional_flags(test_data_exp1[:, feature_indices], {interv_idx_1: interv_val_1}, n_nodes)
y_test_exp1 = test_data_exp1[:, target_idx]

interv_ratios = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
results_mse_exp1 = []

for ratio in interv_ratios:
    n_int = int(context_size * ratio)
    n_obs = context_size - n_int
    
    obs_context = sample_scm(adj, weights, n_obs)
    int_context = sample_scm(adj, weights, n_int, intervention={interv_idx_1: interv_val_1})
    
    X_c = np.vstack([
        add_interventional_flags(obs_context[:, feature_indices], {}, n_nodes),
        add_interventional_flags(int_context[:, feature_indices], {interv_idx_1: interv_val_1}, n_nodes)
    ])
    y_c = np.concatenate([obs_context[:, target_idx], int_context[:, target_idx]])
    
    shuffle = np.random.permutation(len(y_c))
    model = TabICLRegressor()
    model.fit(X_c[shuffle], y_c[shuffle])
    mse = mean_squared_error(y_test_exp1, model.predict(X_test_exp1))
    results_mse_exp1.append(mse)
    print(f"Ratio: {ratio:.2f} | MSE: {mse:.4f}")

#%%
# Experiment 2: Generalization to Unseen Intervention vs Number of Seen Interventions
print("\n--- Experiment 2: Generalization to Unseen Intervention ---")
all_feature_nodes = list(range(n_nodes - 1))
np.random.shuffle(all_feature_nodes)

unseen_idx = all_feature_nodes[0]
pool_indices = all_feature_nodes[1:]
interv_val = 3.0

print(f"Target Unseen Intervention: do(X{unseen_idx})")
print(f"Pool for Context Interventions: {[f'do(X{i})' for i in pool_indices]}")

test_data_exp2 = sample_scm(adj, weights, test_size, intervention={unseen_idx: interv_val})
X_test_exp2 = add_interventional_flags(test_data_exp2[:, feature_indices], {unseen_idx: interv_val}, n_nodes)
y_test_exp2 = test_data_exp2[:, target_idx]

num_seen_interventions = list(range(len(pool_indices) + 1))
results_mse_exp2 = []

for k in num_seen_interventions:
    selected_indices = pool_indices[:k]
    
    if k > 0:
        n_int_total = context_size // 2
        n_obs = context_size - n_int_total
        n_per_int = n_int_total // k
        
        obs_context = sample_scm(adj, weights, n_obs)
        X_list = [add_interventional_flags(obs_context[:, feature_indices], {}, n_nodes)]
        y_list = [obs_context[:, target_idx]]
        
        for idx in selected_indices:
            int_data = sample_scm(adj, weights, n_per_int, intervention={idx: interv_val})
            X_list.append(add_interventional_flags(int_data[:, feature_indices], {idx: interv_val}, n_nodes))
            y_list.append(int_data[:, target_idx])
            
        X_c = np.vstack(X_list)
        y_c = np.concatenate(y_list)
    else:
        obs_context = sample_scm(adj, weights, context_size)
        X_c = add_interventional_flags(obs_context[:, feature_indices], {}, n_nodes)
        y_c = obs_context[:, target_idx]

    shuffle = np.random.permutation(len(y_c))
    model = TabICLRegressor()
    model.fit(X_c[shuffle], y_c[shuffle])
    mse = mean_squared_error(y_test_exp2, model.predict(X_test_exp2))
    results_mse_exp2.append(mse)
    print(f"Num Seen Interventions: {k} | MSE on Unseen: {mse:.4f}")

#%%
# Final Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(interv_ratios, results_mse_exp1, marker='o', color='teal', linewidth=2)
ax1.set_xlabel('Ratio of Interventional Data (Single Action)')
ax1.set_ylabel('MSE on Seen Intervention')
ax1.set_title(f'Experiment 1: Impact of Intervention Ratio (do(X{interv_idx_1}))')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2.plot(num_seen_interventions, results_mse_exp2, marker='s', color='crimson', linewidth=2)
ax2.set_xlabel('Number of Distinct Interventions in Context')
ax2.set_ylabel('MSE on Unseen Intervention')
ax2.set_title(f'Experiment 2: Generalization (Target: do(X{unseen_idx}))')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

# Highlighted DAG Plot for Experiment 2
plot_dag(adj, seen_indices=pool_indices, unseen_index=unseen_idx, 
         title=f"Experiment 2: Target do(X{unseen_idx}), Pool {[f'X{i}' for i in pool_indices]}")
