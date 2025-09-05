import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP, CNN
from utils import load_data, run_epochs

epochs = 30
batch_size = 64
best_accuracy = 0.0
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = load_data(batch_size)
models = [MLP, CNN]

for actual_model in models: 
  model = actual_model().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  run_epochs(
    model=model,
    best_accuracy=best_accuracy,
    criterion=criterion,
    device=device,
    epochs=epochs,
    optimizer=optimizer,
    test_loader=test_loader,
    train_loader=train_loader
  )