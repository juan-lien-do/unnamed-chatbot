# chatbot.py
import os
import re
import torch
import torch.nn as nn
import pickle

# Configuración
CORPUS_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\corpus"
SEQ_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
SAVE_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función de preprocesamiento (debe coincidir con train.py)
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

# Tokenizador básico (debe coincidir con train.py)
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
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Cargar el modelo entrenado
def load_model(model_path, vocab_size, embedding_dim, hidden_dim):
    model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Función para generar respuestas
def generate_response(model, tokenizer, prompt, max_length=50):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_length):
        with torch.no_grad():
            output = model(tokens)
        next_token = output.argmax(dim=-1)[:, -1].item()
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(tokens.squeeze().tolist())

# Chatbot por consola
def main():
    # Cargar el tokenizer
    tokenizer = load_tokenizer(os.path.join(SAVE_DIR, "tokenizer.pkl"))

    # Cargar el modelo entrenado
    model_path = os.path.join(SAVE_DIR, "model.pth")
    model = load_model(model_path, tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

    print("Rengoku: Hola! ¿En qué puedo ayudarte? (Escribe 'salir' para terminar)")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("Rengoku: ¡Hasta luego!")
            break
        response = generate_response(model, tokenizer, preprocess_text(user_input))
        print(f"Rengoku: {response}")
        print("\n")

if __name__ == "__main__":
    main()