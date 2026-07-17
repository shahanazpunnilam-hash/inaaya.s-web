import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

class TextDataProcessor:

    def __init__(self, data_dir="data", output_dir="processed_data", vocab_size=10000):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.vocab_size = vocab_size
        self.output_dir.mkdir(exist_ok=True)

        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"

    def load_text_files(self):
        print("📂 Loading text files...")
        texts = []

        if not self.data_dir.exists():
            print(f"❌ Error: {self.data_dir} directory not found!")
            print(f"Please create a '{self.data_dir}' folder and add text files to it.")
            return []

        text_files = list(self.data_dir.glob("*.txt"))

        if not text_files:
            print(f"⚠️  No .txt files found in {self.data_dir}/")
            print("Add text files to the data/ directory and try again.")
            return []

        for file_path in tqdm(text_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")

        print(f"✓ Loaded {len(texts)} text file(s)")
        return texts

    def clean_text(self, text):
        text = ' '.join(text.split())
        return text

    def build_vocabulary(self, texts):
        print("\n🔤 Building vocabulary...")

        all_tokens = []
        for text in tqdm(texts):
            tokens = text.split()
            all_tokens.extend(tokens)

        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.vocab_size - 4)

        vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        for idx, (token, count) in enumerate(most_common, start=4):
            vocab[token] = idx

        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"  Most common tokens: {[t for t, _ in most_common[:10]]}")

        return vocab

    def tokenize_text(self, text, vocab):
        tokens = text.split()
        token_ids = []

        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                token_ids.append(vocab[self.UNK_TOKEN])

        token_ids = [vocab[self.BOS_TOKEN]] + token_ids + [vocab[self.EOS_TOKEN]]

        return token_ids

    def create_sequences(self, token_ids, seq_length=128):
        sequences = []

        for i in range(0, len(token_ids) - seq_length, seq_length):
            seq = token_ids[i:i + seq_length]
            if len(seq) == seq_length:
                sequences.append(torch.tensor(seq, dtype=torch.long))

        return sequences

    def split_data(self, sequences, train_ratio=0.8, val_ratio=0.1):
        print(f"\n📊 Splitting data ({train_ratio:.0%} train, {val_ratio:.0%} val, {1-train_ratio-val_ratio:.0%} test)...")

        indices = torch.randperm(len(sequences))
        sequences = [sequences[i] for i in indices]

        train_size = int(len(sequences) * train_ratio)
        val_size = int(len(sequences) * val_ratio)

        train_data = sequences[:train_size]
        val_data = sequences[train_size:train_size + val_size]
        test_data = sequences[train_size + val_size:]

        print(f"✓ Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data, vocab):
        print(f"\n💾 Saving processed data to {self.output_dir}/...")

        torch.save(train_data, self.output_dir / "train.pt")
        torch.save(val_data, self.output_dir / "val.pt")
        torch.save(test_data, self.output_dir / "test.pt")

        with open(self.output_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f, indent=2)

        reverse_vocab = {v: k for k, v in vocab.items()}
        with open(self.output_dir / "reverse_vocab.json", 'w') as f:
            json.dump(reverse_vocab, f, indent=2)

        print(f"✓ Saved training data: {len(train_data)} sequences")
        print(f"✓ Saved validation data: {len(val_data)} sequences")
        print(f"✓ Saved test data: {len(test_data)} sequences")
        print(f"✓ Saved vocabulary with {len(vocab)} tokens")

    def process(self, seq_length=128):
        print("=" * 50)
        print("TEXT DATA PROCESSOR")
        print("=" * 50)

        texts = self.load_text_files()
        if not texts:
            return

        print("\n🧹 Cleaning text...")
        texts = [self.clean_text(text) for text in texts]
        total_chars = sum(len(text) for text in texts)
        print(f"✓ Total characters: {total_chars:,}")

        vocab = self.build_vocabulary(texts)

        print("\n🔢 Tokenizing texts...")
        all_sequences = []
        for text in tqdm(texts):
            token_ids = self.tokenize_text(text, vocab)
            sequences = self.create_sequences(token_ids, seq_length=seq_length)
            all_sequences.extend(sequences)

        print(f"✓ Created {len(all_sequences)} sequences of length {seq_length}")

        train_data, val_data, test_data = self.split_data(all_sequences)

        self.save_data(train_data, val_data, test_data, vocab)

        print("\n" + "=" * 50)
        print("✅ Dataset preparation complete!")
        print("=" * 50)
        print("\nNext step: Run training with: python train.py")


if __name__ == "__main__":
    processor = TextDataProcessor(
        data_dir="data",
        output_dir="processed_data",
        vocab_size=10000  # Adjust based on your data
    )

    processor.process(seq_length=128)  # Max sequence length for training
