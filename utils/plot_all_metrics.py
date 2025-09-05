import pandas as pd
import matplotlib.pyplot as plt

def plot_all_metrics(all_metrics):
    
    df = pd.DataFrame(all_metrics)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Criando a figura e os eixos
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plotando as métricas de performance (accuracy, precision, etc.)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Métrica (%)', color='tab:blue', fontsize=12)
    ax1.plot(df['epoch'], df['accuracy'], label='Acurácia', color='blue', marker='o')
    ax1.plot(df['epoch'], df['precision'], label='Precisão', color='green', marker='o')
    ax1.plot(df['epoch'], df['recall'], label='Recall', color='orange', marker='o')
    ax1.plot(df['epoch'], df['f1_score'], label='F1-Score', color='purple', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 100) # Definindo limites para métricas em porcentagem
    
    # Criando o segundo eixo para a perda de treinamento
    ax2 = ax1.twinx()
    ax2.set_ylabel('Perda de Treino', color='tab:red', fontsize=12)
    ax2.plot(df['epoch'], df['train_loss'], label='Perda de Treino', color='red', linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Título e legenda
    fig.tight_layout()
    plt.title('Evolução das Métricas de Validação e Perda de Treino', fontsize=16)
    
    # Unindo as legendas de ambos os eixos
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
    
    plt.show()
