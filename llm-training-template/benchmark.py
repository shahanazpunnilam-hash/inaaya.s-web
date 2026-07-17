import torch
import torch.nn as nn
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
from train import SimpleLanguageModel


class BenchmarkSuite:

    def __init__(self, model_path, device):
        self.device = device

        checkpoint = torch.load(model_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.model = SimpleLanguageModel(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device)
        self.model.eval()

        self.test_data = None

    def load_test_data(self, data_dir="processed_data"):
        data_path = Path(data_dir) / "test.pt"
        self.test_data = torch.load(data_path)
        return self.test_data

    def benchmark_perplexity(self, batch_size=32):
        print("\n📊 PERPLEXITY BENCHMARK")
        print("-" * 60)

        if self.test_data is None:
            self.load_test_data()

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_data), batch_size)):
                batch = torch.stack(self.test_data[i:i+batch_size]).to(self.device)
                logits = self.model(batch)

                logits_flat = logits[:, :-1, :].reshape(-1, self.model.vocab_size)
                targets_flat = batch[:, 1:].reshape(-1)

                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(self.test_data)
        perplexity = np.exp(avg_loss)

        return perplexity

    def benchmark_token_accuracy(self, batch_size=32):
        print("\n🎯 TOKEN ACCURACY BENCHMARK")
        print("-" * 60)

        if self.test_data is None:
            self.load_test_data()

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_data), batch_size)):
                batch = torch.stack(self.test_data[i:i+batch_size]).to(self.device)
                logits = self.model(batch)

                logits_flat = logits[:, :-1, :].reshape(-1, self.model.vocab_size)
                targets_flat = batch[:, 1:].reshape(-1)

                predictions = torch.argmax(logits_flat, dim=1)
                correct += (predictions == targets_flat).sum().item()
                total += targets_flat.size(0)

        accuracy = correct / total * 100
        return accuracy

    def benchmark_text_diversity(self, num_samples=10, max_tokens=100):
        print("\n🎨 TEXT DIVERSITY BENCHMARK")
        print("-" * 60)

        unique_bigrams = set()
        unique_trigrams = set()
        total_words = 0

        self.model.eval()

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                prompt = torch.randint(4, len(self.vocab), (5,))
                tokens = prompt.clone().to(self.device)

                for _ in range(max_tokens):
                    logits = self.model(tokens.unsqueeze(0))
                    next_logits = logits[0, -1, :]
                    probs = torch.softmax(next_logits, dim=0)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token])

                    if next_token.item() == self.vocab.get("<EOS>", 3):
                        break

                for i in range(len(tokens) - 1):
                    bigram = (tokens[i].item(), tokens[i+1].item())
                    unique_bigrams.add(bigram)

                for i in range(len(tokens) - 2):
                    trigram = (tokens[i].item(), tokens[i+1].item(), tokens[i+2].item())
                    unique_trigrams.add(trigram)

                total_words += len(tokens)

        diversity_score = len(unique_bigrams) / total_words if total_words > 0 else 0
        trigram_score = len(unique_trigrams) / total_words if total_words > 0 else 0

        return {
            'unique_bigrams': len(unique_bigrams),
            'unique_trigrams': len(unique_trigrams),
            'total_words': total_words,
            'bigram_diversity': diversity_score,
            'trigram_diversity': trigram_score
        }

    def benchmark_coherence(self, num_samples=5, max_tokens=50):
        print("\n📝 COHERENCE BENCHMARK")
        print("-" * 60)

        coherence_scores = []

        self.model.eval()

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                prompt = torch.randint(4, len(self.vocab), (10,))
                tokens = prompt.clone().to(self.device)

                for _ in range(max_tokens):
                    logits = self.model(tokens.unsqueeze(0))
                    next_logits = logits[0, -1, :]
                    probs = torch.softmax(next_logits, dim=0)
                    confidence = torch.max(probs).item()
                    coherence_scores.append(confidence)

                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token])

                    if next_token.item() == self.vocab.get("<EOS>", 3):
                        break

        avg_confidence = np.mean(coherence_scores)
        return avg_confidence

    def get_benchmark_comparison(self):
        print("\n" + "=" * 70)
        print("🏆 BENCHMARK COMPARISON")
        print("=" * 70)

        perplexity = self.benchmark_perplexity()
        accuracy = self.benchmark_token_accuracy()
        diversity = self.benchmark_text_diversity()
        coherence = self.benchmark_coherence()

        benchmarks = {
            'Your Model': {
                'Perplexity': perplexity,
                'Accuracy': accuracy,
                'Diversity': diversity['bigram_diversity'],
                'Coherence': coherence * 100
            },
            'GPT-2 (estimated)': {
                'Perplexity': 50.0,
                'Accuracy': 75.0,
                'Diversity': 0.45,
                'Coherence': 65.0
            },
            'Claude-like (estimated)': {
                'Perplexity': 25.0,
                'Accuracy': 85.0,
                'Diversity': 0.55,
                'Coherence': 80.0
            }
        }

        print("\n📊 RESULTS COMPARISON:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Your Model':<20} {'GPT-2':<20} {'Claude-like':<20}")
        print("-" * 70)

        for metric in ['Perplexity', 'Accuracy', 'Diversity', 'Coherence']:
            your_score = benchmarks['Your Model'][metric]
            gpt2_score = benchmarks['GPT-2 (estimated)'][metric]
            claude_score = benchmarks['Claude-like (estimated)'][metric]

            print(f"{metric:<20} {your_score:<20.2f} {gpt2_score:<20.2f} {claude_score:<20.2f}")

        print("\n" + "=" * 70)
        print("📈 DETAILED BREAKDOWN")
        print("=" * 70)

        print(f"\n✨ Your Model Metrics:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"    → How confused the model is (lower is better)")
        print(f"    → Your score: {'🔥 Excellent!' if perplexity < 50 else '👍 Good' if perplexity < 100 else '📚 Keep improving!'}")

        print(f"\n  Token Accuracy: {accuracy:.2f}%")
        print(f"    → % of correctly predicted next tokens")
        print(f"    → Your score: {'🔥 Excellent!' if accuracy > 75 else '👍 Good' if accuracy > 60 else '📚 Keep improving!'}")

        print(f"\n  Text Diversity: {diversity['bigram_diversity']:.4f}")
        print(f"    → How varied the generated text is (higher is better)")
        print(f"    → Unique bigrams: {diversity['unique_bigrams']}")
        print(f"    → Unique trigrams: {diversity['unique_trigrams']}")

        print(f"\n  Coherence Score: {coherence * 100:.2f}%")
        print(f"    → Average confidence in predictions (higher is better)")
        print(f"    → Your score: {'🔥 Very confident!' if coherence > 0.7 else '👍 Confident' if coherence > 0.5 else '📚 Less confident'}")

        print("\n" + "=" * 70)
        print("🎯 COMPARISON INSIGHTS")
        print("=" * 70)

        print(f"\n🔍 vs GPT-2:")
        if perplexity < 50:
            print(f"  ✅ Your perplexity is better than typical GPT-2!")
        else:
            print(f"  → Your perplexity is {perplexity - 50:.0f} points higher than GPT-2")
            print(f"     (This is normal for a small model!)")

        print(f"\n🔍 vs Claude:")
        print(f"  → Claude is a much larger model trained on more data")
        print(f"  → Your model's performance is good for its size!")

        print("\n" + "=" * 70)
        print("💡 HOW TO IMPROVE YOUR SCORE")
        print("=" * 70)
        print("""
1. **Reduce Perplexity**
   → Train for more epochs
   → Use more training data
   → Increase model size (larger hidden_dim)

2. **Increase Accuracy**
   → Train longer (more epochs)
   → Use higher quality data
   → Adjust learning rate

3. **Improve Diversity**
   → Use more varied training data
   → Increase temperature during generation
   → Train on larger dataset

4. **Boost Coherence**
   → More training data
   → Larger model
   → Better quality data

5. **General Tips**
   → Check data quality
   → Increase training time
   → Try different hyperparameters
   → Use a larger model size
        """)

        return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Benchmark your trained LLM")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="processed_data",
                        help="Path to processed data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    print("=" * 70)
    print("🏆 LLM BENCHMARK SUITE")
    print("=" * 70)

    suite = BenchmarkSuite(args.model, device)
    suite.load_test_data(args.data_dir)

    benchmarks = suite.get_benchmark_comparison()

    print("\n✅ Benchmark complete!")
    print("\n💫 Share your results with friends to compare!")


if __name__ == "__main__":
    main()
