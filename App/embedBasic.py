import math

# 1. Pré-processamento simples
#O texto recebido sera dividido por palavras, ignorando caracteres especiais
def preprocess(text):
    text = text.lower()
    for char in ".,!?;:()[]{}<>\"'\\/|@#$%^&*-_=+`~":
        text = text.replace(char, '')
    return text.split()


# 2. Construção de vocabulário
#Análise de cada palavra da lista de palavras e adição de cada palavra que não esteja adicionada na lista de vocabulário, sem repetições
def build_vocabulary(corpus):
    vocab = []
    for text in corpus:
        for word in preprocess(text):
            if word not in vocab:
                vocab.append(word)
    return vocab

# 3. Calcular TF
# Calculo para saber o quão frequente é a palavra no texto
def term_frequency(text, vocab):
    tokens = preprocess(text)
    tf = []
    for word in vocab:
        tf.append(tokens.count(word) / len(tokens))
    return tf


# 4. Calcular IDF
# Cálculo para saber o quanto uma palavra é rara no texto, penalizando palavras muito frequêntes
def inverse_document_frequency(corpus, vocab):
    idf = []
    N = len(corpus)
    for word in vocab:
        count = sum(1 for doc in corpus if word in preprocess(doc))
        idf.append(math.log((N + 1) / (count + 1)) + 1)  # Suavização
    return idf

# 5. Calcular TF-IDF
# Produto entre TF e IDF que irá prover pesos às palavras e quanto mais rara e frequente for, maior será seu valor
def tf_idf_vector(text, vocab, idf):
    tf = term_frequency(text, vocab)
    return [tf[i] * idf[i] for i in range(len(vocab))]

# 6. Exemplo de uso
corpus = [
    "O cachorro gosta de brincar.",
    "O gato não gosta de brincar.",
    "O cachorro e o gato são amigos."
]

# Vocabulário
vocab = build_vocabulary(corpus)

# IDF
idf = inverse_document_frequency(corpus, vocab)

# TF-IDF Vetores
vectors = []
for text in corpus:
    vector = tf_idf_vector(text, vocab, idf)
    vectors.append(vector)

# Mostrar resultados
print("Vocabulário:", vocab)
print("\nVetores TF-IDF:")
for vec in vectors:
    print([round(v, 3) for v in vec])
