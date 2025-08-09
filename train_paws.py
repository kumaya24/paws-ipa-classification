# python train_paws.py char
# python train_paws.py ipa

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from collections import defaultdict
import sys

class PAWSDatasetChar(Dataset):
    def __init__(self, tsv_file, sep_token="|"):
        self.df = pd.read_csv(tsv_file, sep="\t", header=None, names=["pair", "label"])
        self.sep_token = sep_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pair = self.df.iloc[idx]["pair"].replace(" ", "")
        label = int(self.df.iloc[idx]["label"])
        s1, s2 = pair.split(self.sep_token, maxsplit=1)
        s1_tokens = self.tokenize(s1)
        s2_tokens = self.tokenize(s2)
        return torch.tensor(s1_tokens, dtype=torch.long), torch.tensor(s2_tokens, dtype=torch.long), label

    def tokenize(self, text):
        return [ord(c) for c in text]

class PAWSDatasetIPA(Dataset):
    def __init__(self, tsv_file, vocab, sep_token="|"):
        self.df = pd.read_csv(tsv_file, sep="\t", header=None, names=["pair", "label"])
        self.sep_token = sep_token
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pair = self.df.iloc[idx]["pair"].strip()
        label = int(self.df.iloc[idx]["label"])
        s1, s2 = pair.split(self.sep_token, maxsplit=1)
        s1_tokens = self.tokenize(s1)
        s2_tokens = self.tokenize(s2)
        return torch.tensor(s1_tokens, dtype=torch.long), torch.tensor(s2_tokens, dtype=torch.long), label

    def tokenize(self, text):
        tokens = text.split()  # split IPA string by spaces
        return [self.vocab.get(token, 0) for token in tokens]  # 0 for unknown/pad

def build_ipa_vocab(tsv_file, sep_token="|"):
    df = pd.read_csv(tsv_file, sep="\t", header=None, names=["pair", "label"])
    vocab = defaultdict(lambda: len(vocab) + 1)  # start ids from 1
    for pair in df["pair"]:
        s1, s2 = pair.split(sep_token, maxsplit=1)
        for token in s1.split():
            vocab[token]
        for token in s2.split():
            vocab[token]
    return dict(vocab)

def collate_fn(batch):
    s1_batch, s2_batch, labels = zip(*batch)
    s1_padded = pad_sequence(s1_batch, batch_first=True, padding_value=0)
    s2_padded = pad_sequence(s2_batch, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return s1_padded, s2_padded, labels_tensor

class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size=65536, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def encode(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.encoder(emb)
        return hn[-1]

    def forward(self, s1, s2):
        h1 = self.encode(s1)
        h2 = self.encode(s2)
        h = torch.cat([h1, h2], dim=1)
        return self.classifier(h)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for s1, s2, labels in dataloader:
            s1, s2, labels = s1.to(device), s2.to(device), labels.to(device)
            outputs = model(s1, s2)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return acc

def train(train_tsv, test_tsv, mode="char", epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "char":
        train_dataset = PAWSDatasetChar(train_tsv)
        test_dataset = PAWSDatasetChar(test_tsv)
        vocab_size = 65536 
    elif mode == "ipa":
        vocab = build_ipa_vocab(train_tsv)
        train_dataset = PAWSDatasetIPA(train_tsv, vocab)
        test_dataset = PAWSDatasetIPA(test_tsv, vocab)
        vocab_size = len(vocab) + 1
    else:
        raise ValueError("Mode must be 'char' or 'ipa'")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SiameseLSTM(vocab_size=vocab_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for s1, s2, labels in train_loader:
            s1, s2, labels = s1.to(device), s2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(s1, s2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_model_{mode}.pth")

    print(f"Best test accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "char"

    train(
        train_tsv="char_ko_dev.tsv" if mode == "char" else "ipa_ko_dev.tsv",
        test_tsv="char_ko_test.tsv" if mode == "char" else "ipa_ko_test.tsv",
        mode=mode,
        epochs=5
    )
