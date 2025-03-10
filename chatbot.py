# chatbot.py
import os
import torch
import torch.nn as nn

# Configuración
CORPUS_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\corpus"
SEQ_LEN = 128  # Longitud de la secuencia de entrada es 128 en realidad
EMBEDDING_DIM = 256  # Dimensión de los embeddings
HIDDEN_DIM = 512  # Dimensión de la capa oculta
SAVE_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizador básico (debe coincidir con el usado en el entrenamiento)
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

# Modelo de lenguaje simple (debe coincidir con el usado en el entrenamiento)
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
        corpus_text = f.read()
    return SimpleTokenizer([corpus_text])

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
    tokenizer = load_tokenizer(CORPUS_DIR)

    # Cargar el modelo entrenado
    model_path = os.path.join(SAVE_DIR, "model.pth")
    model = load_model(model_path, tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

    print("Chatbot: Hola! ¿En qué puedo ayudarte? (Escribe 'salir' para terminar)")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("Chatbot: ¡Hasta luego!")
            break
        response = generate_response(model, tokenizer, user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()