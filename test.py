# chatbot.py
import os
import re
import torch
import torch.nn as nn


# Configuración
CORPUS_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\corpus"
SEQ_LEN = 128
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
SAVE_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"--", " ", text)  # Reemplazar guiones dobles por espacio
    text = re.sub(r"-", "", text)  # Reemplazar guiones por espacio en blanco
    text = re.sub(r"\?", " ? ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r"!", " ! ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r"¡", " ¡ ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r"¿", " ¿ ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r",", " , ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r"\.", " . ", text)  # Hacer que permanezcan algunos signos
    text = re.sub(r"[^a-záéíóúüñ0123456789,.¡¿?!\s]", "", text)  # Eliminar caracteres no alfabéticos
    text = re.sub(r"\s+", " ", text).strip()  # Eliminar espacios extra y tabulaciones
    return text


class SimpleTokenizer:
    def __init__(self, corpus):
        self.vocab = self.build_vocab(corpus)
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}

    def build_vocab(self, corpus):
        tokens = set()
        for text in corpus:
            tokens.update(text.split())
        tokens = sorted(tokens)
        tokens.insert(0, "<UNK>")  # Agregar el token <UNK> al vocabulario
        return tokens

    def encode(self, text):
        return [self.token_to_idx.get(t, self.token_to_idx["<UNK>"]) for t in text.split()]

    def decode(self, ids):
        return " ".join([self.idx_to_token.get(i, "<UNK>") for i in ids])

# Modelo de lenguaje simple (debe coincidir con train.py)
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

# Cargar el tokenizer
def load_tokenizer(corpus_dir):
    file_paths = [os.path.join(corpus_dir, fname) for fname in os.listdir(corpus_dir)]
    with open(file_paths[0], "r", encoding="utf-8") as f:
        corpus_text = preprocess_text(f.read())  # Aplicar preprocesamiento
    return SimpleTokenizer([corpus_text])

# Cargar el modelo entrenado
def load_model(model_path, vocab_size, embedding_dim, hidden_dim):
    model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

tokenizer = load_tokenizer(CORPUS_DIR)

# Verifica el tokenizer
print("Vocabulario:", tokenizer.vocab[:1000])  # Muestra las primeras 10 palabras del vocabulario
print("Texto codificado:", tokenizer.encode("La historia de argentina es"))