def model_otimization(model, train_loader, device, criterion, optimizer):
  for images, labels in train_loader:
    # Passa a imagem pela rede (forward pass)
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    # Calcula a perda (loss)
    loss = criterion(outputs, labels)
    
    # Backpropagation e otimização
    optimizer.zero_grad()  # Zera os gradientes
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pesos
    
  return loss