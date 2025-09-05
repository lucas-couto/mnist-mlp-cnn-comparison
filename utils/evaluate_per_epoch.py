import torch
from .get_metrics import get_metrics

def evaluate_per_epoch(model, device, test_loader):
  model.eval()  # Modo de avaliação
  all_labels = []
  all_predictions = []
  
  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      
      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predicted.cpu().numpy())
  
  return get_metrics(all_labels, all_predictions)