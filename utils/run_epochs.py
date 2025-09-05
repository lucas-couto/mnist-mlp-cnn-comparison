from .save_best_model import save_best_model
from .plot_all_metrics import plot_all_metrics
from .model_otimization import model_otimization
from .evaluate_per_epoch import evaluate_per_epoch

def run_epochs(model, epochs, train_loader, test_loader, optimizer, device, criterion, best_accuracy):
  print("Iniciando o treinamento...")
  
  all_metrics = []
  
  for epoch in range(epochs):
    model.train()  # Seta o modelo para o modo de treinamento

    # Otimização do modelo
    loss = model_otimization(model, train_loader, device, criterion, optimizer)
    
    # Avaliação da performance por época
    metrics = evaluate_per_epoch(model, device, test_loader)
    print(f'Época [{epoch+1}/{epochs}], Perda de Treino: {loss.item():.2f}')
    print('--- Métricas de Validação ---')
    print(f'Acurácia: {metrics["accuracy"]:.2f}%')
    print(f'Precisão: {metrics["precision"]*100:.2f}%')
    print(f'Recall: {metrics["recall"]*100:.2f}%')
    print(f'F1-Score: {metrics["f1_score"]*100:.2f}%')
    print('-----------------------------')
    
    metrics['epoch'] = epoch
    metrics['train_loss'] = loss.item()
    all_metrics.append(metrics)
    # Salva modelo com a melhor acuracia.
    best_accuracy = save_best_model(model, metrics['accuracy'], best_accuracy)
  
  
  plot_all_metrics(all_metrics)
  
  print("Treinamento finalizado.")
  print(f"Acurácia final do melhor modelo: {best_accuracy:.2f}%")