import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

# Configuración
CORPUS_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\corpus"
BATCH_SIZE = 16
EPOCHS = 40  # era 10
LEARNING_RATE = 0.0004
SEQ_LEN = 256
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
SAVE_DIR = r"I:\desarrollo de sistemas\SLM prueba\03\model"

# Función de preprocesamiento
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

# Tokenizador básico
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

# Dataset personalizado
class TextDataset(Dataset):
    def __init__(self, file_paths, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = preprocess_text(f.read())  # Aplicar preprocesamiento
                tokens = self.tokenizer.encode(text)
                for i in range(0, len(tokens) - seq_len, seq_len):
                    self.data.append(tokens[i:i + seq_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# Modelo de lenguaje simple
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

# Cargar el corpus
file_paths = [os.path.join(CORPUS_DIR, fname) for fname in os.listdir(CORPUS_DIR)]
with open(file_paths[0], "r", encoding="utf-8") as f:
    corpus_text = preprocess_text(f.read())  # Aplicar preprocesamiento

tokenizer = SimpleTokenizer([corpus_text])
dataset = TextDataset(file_paths, tokenizer, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inicializar el modelo, la función de pérdida y el optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLanguageModel(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch[:, :-1])
        loss = criterion(output.transpose(1, 2), batch[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# Guardar el modelo
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))

# Guardar el tokenizer
with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
