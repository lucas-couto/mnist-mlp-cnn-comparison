# Comparação de CNN e MLP no MNIST

## 🤖 Visão Geral do Projeto

Este projeto demonstra a classificação de dígitos manuscritos utilizando o famoso dataset **MNIST**. O objetivo principal foi comparar o desempenho de duas arquiteturas de redes neurais amplamente utilizadas: uma **Rede Neural Multilayer Perceptron (MLP)** e uma **Rede Neural Convolucional (CNN)**.

O trabalho foi dividido em três etapas principais:

1.  **Pré-processamento de Dados:** Preparação e normalização do dataset MNIST.
2.  **Modelagem:** Construção e treinamento de uma MLP e de uma CNN separadamente.
3.  **Análise de Resultados:** Comparação da acurácia, precisão, recall, f-measure e da perda (loss) de ambos os modelos para entender suas vantagens e desvantagens.

## 🚀 Tecnologias e Ferramentas

- **Linguagem:** Python
- **Frameworks/Bibliotecas:**
  - `PyTorch`: Para o desenvolvimento dos modelos de deep learning.
  - `TorchVision`: Para adquirir e pré processar o dataset.
  - `Matplotlib`: Para a visualização dos gráficos de treinamento.
  - `scikit-learn`: Para métricas de avaliação.

## 📈 Resultados e Conclusão

A **CNN** superou a **MLP** em termos de acurácia, precisão e recall. O que reforça o poder das camadas convolucionais na extração de características espaciais de imagens. A arquitetura da CNN foi capaz de identificar padrões essenciais nos dados, levando a um desempenho superior na tarefa de classificação.

## ⚙️ Como Executar o Projeto

1.  **Clone o Repositório:**

    ```bash
    git clone https://github.com/lucas-couto/mnist-mlp-cnn-comparison.git
    cd mnist-mlp-cnn-comparison
    ```

2.  **Crie e Ative o Ambiente Conda:**

    É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto. Se você ainda não tem o Conda instalado, [baixe e instale a versão mais recente do Anaconda ou Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    ```bash
    # Cria o ambiente conda desse projeto
    conda env create -f environment.yml

    # Ativa o ambiente para trabalhar nele
    conda activate mnist-mlp-cnn-comparison
    ```

3.  **Execute o Script Principal:**

    Agora que o ambiente está pronto e as dependências instaladas, você pode rodar o script principal do projeto.

    ```bash
    python main.py
    ```

## 📂 Estrutura do Projeto

- `utils/`: Pasta contendo funções utilitárias.
- `main.py`: Arquivo principal.
- `requirements.yml`: Lista de todas as bibliotecas necessárias.
- `README.md`: Este arquivo.
- `models/`: Modelos de MLP e CNN.
- `best_mnist_MLP_model.pth`: Contém o modelo MLP com a melhor época salvo. **(Gerado após o treinamento)**
- `best_mnist_CNN_model.pth`: Contém o modelo CNN com a melhor época salvo. **(Gerado após o treinamento)**
- `data/`: Contém o dataset MNIST. **(Gerado após a execução do script)**

## ✒️ Autor

- Lucas Couto
