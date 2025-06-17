import numpy as np
import math

# 1. Pré-processamento básico
def preprocess(text):
    text = text.lower()
    for char in ".,!?;:()[]{}<>\"'\\/|@#$%^&*-_=+`~":
        text = text.replace(char, '')
    return text.split()

# 2. Criar vocabulário
def build_vocabulary(corpus):
    vocab = []
    for text in corpus:
        for word in preprocess(text):
            if word not in vocab:
                vocab.append(word)
    return vocab

# 3. Term Frequency (TF)
def compute_tf(text, vocab):
    words = preprocess(text)
    tf_vector = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        tf_vector[i] = words.count(term) / len(words)
    return tf_vector

# 4. Inverse Document Frequency (IDF)
def compute_idf(corpus, vocab):
    N = len(corpus)
    idf = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        count = sum(1 for doc in corpus if term in preprocess(doc))
        idf[i] = math.log((N + 1) / (count + 1)) + 1  # Suavização
    return idf

# 5. TF-IDF
def compute_tfidf(text, vocab, idf):
    tf = compute_tf(text, vocab)
    tfidf = tf * idf
    return tfidf

# 6. Aplicação
corpus = [
    "O cachorro gosta de brincar.",
    "O cachorro gosta de brincar.",
    "O céu é azul e a luz é amarela."
]

vocab = build_vocabulary(corpus)
idf = compute_idf(corpus, vocab)

print("Vocabulário:", vocab)

print("\nVetores TF-IDF:")
for text in corpus:
    vec = compute_tfidf(text, vocab, idf)
    print(np.round(vec, 3))

vectors = [compute_tfidf(t, vocab, idf) for t in corpus]

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

# Exemplo: similaridade entre frase 0 e 1
sim = cosine_similarity(vectors[0], vectors[1])
print(f"\nSimilaridade entre frase 0 e 1: {sim:.3f}")
