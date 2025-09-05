import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def get_metrics(all_labels, all_predictions):
  all_labels = np.array(all_labels)
  all_predictions = np.array(all_predictions)

  # Calcula as métricas
  accuracy = (all_predictions == all_labels).mean() * 100
  precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
  recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
  f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
  conf_matrix = confusion_matrix(all_labels, all_predictions)

  metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': conf_matrix
  }

  return metrics
