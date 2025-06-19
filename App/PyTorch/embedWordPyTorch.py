import torch
import torch.nn as nn
import torch.optim as optim

# Dados
data = [
    ("gato", "animal"),
    ("cachorro", "animal"),
    ("tigre", "animal"),
    ("maçã", "fruta"),
    ("banana", "fruta"),
    ("laranja", "fruta")
]

# Construção dos vocabulários
word_to_idx = {word: i for i, (word, _) in enumerate(data)}
label_to_idx = {"animal": 0, "fruta": 1}

# Hiperparâmetros
vocab_size = len(word_to_idx) # Quantidade de palavras que iremos usar
embedding_dim = 4  # Você pode aumentar se quiser
num_classes = len(label_to_idx) # Classes, ou é animal ou é fruta

# Modelo
class WordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, word_idx):
        embeds = self.embeddings(word_idx)
        output = self.linear(embeds)
        return output

# Instanciação
model = WordClassifier(vocab_size, embedding_dim, num_classes)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Treinamento
for epoch in range(100):
    total_loss = 0
    for word, label in data:
        word_idx = torch.tensor([word_to_idx[word]])
        label_idx = torch.tensor([label_to_idx[label]])

        optimizer.zero_grad()
        outputs = model(word_idx)
        loss = criterion(outputs, label_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# 🔍 Visualizando os embeddings treinados
print("\nEmbeddings aprendidos:")
for word, idx in word_to_idx.items():
    vector = model.embeddings.weight[idx].detach().numpy()
    print(f"{word}: {vector}")
