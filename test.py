

# %%
import torch
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import TabICLPriorDataLoader
from tfmplayground.priors.dataloader import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss

# %%

device = get_default_device()
print(device) # mac m4 pro


# %%

model = NanoTabPFNModel(
    num_attention_heads=4,
    embedding_size=64,
    mlp_hidden_size=512,
    num_layers=4,
    num_outputs=2,
)
criterion = CrossEntropyLoss()
print(model)

# %%

device = get_default_device()
prior = TabICLPriorDataLoader(
    num_steps=16, 
    batch_size=2, 

    num_datapoints_min=256,
    num_datapoints_max=1024,

    min_features=2,
    max_features=4,

    max_num_classes=2,

    device=device,
    prior_type="mix_scm"
    )


next(prior)

# %%

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=8,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)


# %%
