import torch
import torch.nn as nn
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse


def load_model_and_vocab(model_path, device):
    print(f"📂 Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']

    from train import SimpleLanguageModel

    model = SimpleLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )

    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    print(f"✓ Loaded model with {len(vocab)} vocabulary")

    return model, vocab


def load_test_data(data_dir="processed_data"):
    data_path = Path(data_dir) / "test.pt"
    test_data = torch.load(data_path)

    print(f"✓ Loaded {len(test_data)} test sequences")

    return test_data


def evaluate_model(model, test_data, device, batch_size=32):
    print("\n📊 Evaluating on test set...")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = torch.stack(test_data[i:i+batch_size]).to(device)

        with torch.no_grad():
            logits = model(batch)

            logits_flat = logits[:, :-1, :].reshape(-1, model.vocab_size)
            targets_flat = batch[:, 1:].reshape(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch.size(0)

            predictions = torch.argmax(logits_flat, dim=1)
            correct += (predictions == targets_flat).sum().item()
            total += targets_flat.size(0)

    avg_loss = total_loss / len(test_data)
    accuracy = correct / total * 100
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity, accuracy


def show_sample_predictions(model, vocab, num_samples=5):
    print(f"\n🔮 Sample Predictions (first 5 tokens):")
    print("-" * 60)

    reverse_vocab = {v: k for k, v in vocab.items()}

    sample_input = torch.randint(4, len(vocab), (1, 20))

    model.eval()
    with torch.no_grad():
        logits = model(sample_input)

    for pos in range(min(5, logits.size(1))):
        actual_token_id = sample_input[0, pos + 1].item() if pos + 1 < sample_input.size(1) else None
        predicted_logits = logits[0, pos, :]
        top_k = torch.topk(predicted_logits, k=5)

        actual_word = reverse_vocab.get(actual_token_id, "?") if actual_token_id else "?"
        predicted_word = reverse_vocab.get(top_k.indices[0].item(), "?")

        print(f"\nPosition {pos}:")
        print(f"  Actual next token: {actual_word}")
        print(f"  Model predicts (top 5):")
        for rank, (logit, idx) in enumerate(zip(top_k.values, top_k.indices)):
            token = reverse_vocab.get(idx.item(), "?")
            print(f"    {rank + 1}. {token} (confidence: {logit:.3f})")


def generate_text(model, vocab, prompt_tokens, max_length=50, temperature=0.8):
    print(f"\n💭 Text Generation Example:")
    print("-" * 60)

    reverse_vocab = {v: k for k, v in vocab.items()}

    model.eval()
    tokens = prompt_tokens.clone()

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens.unsqueeze(0))
            next_logits = logits[0, -1, :] / temperature

            probs = torch.softmax(next_logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token])

            if next_token.item() == 3:
                break

    generated_text = []
    for token_id in tokens:
        word = reverse_vocab.get(token_id.item(), "?")
        if word not in ["<BOS>", "<PAD>"]:
            generated_text.append(word)

    print("Generated: " + " ".join(generated_text[:50]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained language model")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Path to processed data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    model, vocab = load_model_and_vocab(args.model, device)
    test_data = load_test_data(args.data_dir)

    test_loss, test_perplexity, accuracy = evaluate_model(
        model, test_data, device, args.batch_size
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")
    print(f"  (Lower is better. < 100 is good, < 50 is excellent)")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"  (% of correctly predicted next tokens)")

    show_sample_predictions(model, vocab)

    prompt = torch.randint(4, len(vocab), (10,))
    generate_text(model, vocab, prompt, max_length=30)

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)

    print("\n📖 How to interpret results:")
    print("  - Perplexity: How 'confused' the model is about the next word")
    print("    - 50-100: Good for a small model")
    print("    - 20-50: Very good")
    print("    - <20: Excellent")
    print("  - Accuracy: % of correctly predicted next tokens")
    print("  - Generated text: Check if it makes linguistic sense")


if __name__ == "__main__":
    main()
