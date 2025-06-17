import torch
import torch.nn as nn
import torch.optim as optim

# Dataset de treino com frases e rótulos
data = [
    ("eu amei o filme", "positivo"),
    ("este filme é ótimo", "positivo"),
    ("gostei muito do final", "positivo"),
    ("não gostei do filme", "negativo"),
    ("o final foi ruim", "negativo"),
    ("filme horrível", "negativo")
]

# Construindo o vocabulário com token <UNK>
vocab = set(word for sentence, _ in data for word in sentence.split())
word_to_idx = {"<UNK>": 0}
word_to_idx.update({word: i+1 for i, word in enumerate(vocab)})

label_to_idx = {"positivo": 0, "negativo": 1}

#  Hiperparâmetros
vocab_size = len(word_to_idx)
embedding_dim = 8
num_classes = 2

# Modelo
class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        embeds = self.embedding(inputs)           # [seq_len, embed_dim]
        sentence_vector = embeds.mean(dim=0)      # média dos embeddings das palavras
        output = self.classifier(sentence_vector) # classifica a frase
        return output

# Treinamento
model = SentenceClassifier(vocab_size, embedding_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loop de treino
for epoch in range(50):
    total_loss = 0
    for sentence, label in data:
        words = sentence.split()
        word_idxs = torch.tensor([
            word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words
        ], dtype=torch.long)

        label_idx = torch.tensor([label_to_idx[label]])

        optimizer.zero_grad()
        outputs = model(word_idxs)
        loss = criterion(outputs.unsqueeze(0), label_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Função para prever frases
def predict(sentence):
    words = sentence.split()
    idxs = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
    inputs = torch.tensor(idxs, dtype=torch.long)
    with torch.no_grad():
        outputs = model(inputs)
        predicted = torch.argmax(outputs).item()
        label = [k for k, v in label_to_idx.items() if v == predicted][0]
        print(f"Frase: '{sentence}' => Classe: {label}")

# Testando com frases fora do treinamento
predict("amei o filme")          # Esperado: positivo
predict("final horrível")        # Esperado: negativo
predict("este final é incrível") # Esperado: positivo (ou erro aceitável)
