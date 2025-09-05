import torch

def save_best_model(model, current_accuracy, best_accuracy):
  if current_accuracy > best_accuracy:
    best_accuracy = current_accuracy
    # Salva o estado do modelo
    torch.save(model.state_dict(), f'best_mnist_{model.__class__.__name__}_model.pth')
    print(f"  -> Melhor modelo salvo com acurácia de: {best_accuracy:.2f}%")
  
  return best_accuracy