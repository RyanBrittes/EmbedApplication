import numpy as np
from collections import defaultdict
import random

# Corpus simples
corpus = "o gato está no telhado e o cachorro está no quintal".lower().split()

# Parâmetros
window_size = 2  # contexto em torno da palavra
embedding_dim = 5  # dimensão do vetor de embedding

# Construir vocabulário
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

#Criando pares
def generate_training_data(corpus, window_size):
    training_data = []

    for i, target in enumerate(corpus):
        target_idx = word2idx[target]

        # contexto à esquerda e à direita
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(corpus):
                context_word = corpus[j]
                context_idx = word2idx[context_word]
                training_data.append((target_idx, context_idx))

    return training_data

training_data = generate_training_data(corpus, window_size)
print("Exemplo de pares (entrada, contexto):")
for i in range(5):
    print(f"{idx2word[training_data[i][0]]} -> {idx2word[training_data[i][1]]}")

#Criando e treinando o modelo Skip-gram

# Inicializar pesos
W1 = np.random.rand(vocab_size, embedding_dim)
W2 = np.random.rand(embedding_dim, vocab_size)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# One-hot encoding
def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

# Treinamento
lr = 0.01
epochs = 5000

for epoch in range(epochs):
    loss = 0
    for target_idx, context_idx in training_data:
        x = one_hot(target_idx, vocab_size)  # vetor de entrada
        h = np.dot(W1.T, x)                  # camada escondida
        u = np.dot(W2.T, h)                  # camada de saída
        y_pred = softmax(u)                  # previsão

        y_true = one_hot(context_idx, vocab_size)

        # Cálculo do erro
        e = y_pred - y_true
        dW2 = np.outer(h, e)
        dW1 = np.outer(x, np.dot(W2, e))

        # Atualização dos pesos
        W1 -= lr * dW1
        W2 -= lr * dW2

        loss += -np.log(y_pred[context_idx] + 1e-9)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#Usando os embeddings
def get_embedding(word):
    idx = word2idx[word]
    return W1[idx]

print("\nVetor da palavra 'cachorro':")
print(get_embedding("cachorro"))
