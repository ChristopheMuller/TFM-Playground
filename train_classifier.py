#%%


from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import TabICLPriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss
from torch.backends import mps
import torch

#%%

model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=32*6,
    mlp_hidden_size=256,
    num_layers=2,
    num_outputs=2,
)
criterion = CrossEntropyLoss()

#%%

# device = mps if mps.is_available() else "cpu"
device=get_default_device()
print(f"Using device: {device}")

#%%

prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_min=10,
    num_datapoints_max=50,
    min_features=2,
    max_features=3,
    device=device,
    max_num_classes=2,
    prior_type="mix_scm"
)

#%%

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=50,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)

#%%

from my_data import DrivenSpinSystem, EasyData

# spin_system = DrivenSpinSystem(n_features=2, n_spins=500, n_iterations=30, random_state=1234)
# X,y = spin_system.generate(n_samples=400)

easy_data = EasyData(n_features=2, random_state=1234)
X, y = easy_data.generate(n_samples=400)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.colorbar(label='Class Label')
plt.show()


#%%

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(0)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(0)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(0) 
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(0)

with torch.no_grad():
    
    y_preds = trained_model(
        X_train_tensor,
        y_train_tensor,
        X_test_tensor
    )


#%%

from sklearn.metrics import classification_report, confusion_matrix

y_pred_labels = torch.argmax(y_preds, dim=-1).cpu().numpy().flatten()
y_true_labels = y_test_tensor.cpu().numpy().flatten()

print(classification_report(y_true_labels, y_pred_labels))
print(confusion_matrix(y_true_labels, y_pred_labels))