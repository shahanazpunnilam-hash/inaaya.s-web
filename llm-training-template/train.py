import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


class SimpleLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_output, _ = self.lstm(embeddings)
        lstm_output = self.dropout(lstm_output)

        logits = self.fc(lstm_output)

        return logits


class Trainer:

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_input_ids in progress_bar:
            batch_input_ids = batch_input_ids.to(self.device)

            logits = self.model(batch_input_ids)

            logits_flat = logits[:, :-1, :].reshape(-1, self.model.vocab_size)
            targets_flat = batch_input_ids[:, 1:].reshape(-1)

            loss = self.criterion(logits_flat, targets_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_input_ids in val_loader:
                batch_input_ids = batch_input_ids.to(self.device)

                logits = self.model(batch_input_ids)

                logits_flat = logits[:, :-1, :].reshape(-1, self.model.vocab_size)
                targets_flat = batch_input_ids[:, 1:].reshape(-1)

                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def calculate_perplexity(self, loss):
        return np.exp(loss)


def load_data(processed_data_dir="processed_data", batch_size=32):
    data_dir = Path(processed_data_dir)

    print("📂 Loading processed data...")
    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")

    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)

    train_dataset = TensorDataset(torch.stack(train_data))
    val_dataset = TensorDataset(torch.stack(val_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Vocabulary size: {len(vocab)}")

    return train_loader, val_loader, vocab


def main():
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    Path(args.model_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    train_loader, val_loader, vocab = load_data(args.data_dir, args.batch_size)

    print("\n" + "=" * 50)
    print("CREATING MODEL")
    print("=" * 50)
    model = SimpleLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Embedding: {args.embedding_dim}")
    print(f"  Hidden: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")

    trainer = Trainer(model, device, learning_rate=args.learning_rate)

    start_epoch = 0
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1

    print("\n" + "=" * 50)
    print(f"TRAINING ({args.epochs} epochs)")
    print("=" * 50)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")

        train_loss = trainer.train_epoch(train_loader)
        train_perplexity = trainer.calculate_perplexity(train_loss)

        val_loss = trainer.evaluate(val_loader)
        val_perplexity = trainer.calculate_perplexity(val_loss)

        print(f"  Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")

        checkpoint_path = Path(args.model_dir) / f"model_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.model_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'vocab': vocab,
            }, best_path)
            print(f"  ✨ Best model saved!")

    print("\n" + "=" * 50)
    print("✅ Training complete!")
    print("=" * 50)
    print(f"\nNext step: Run evaluation with: python evaluate.py")


if __name__ == "__main__":
    main()
