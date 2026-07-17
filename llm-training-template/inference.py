import torch
import json
from pathlib import Path
import argparse
from train import SimpleLanguageModel


class TextGenerator:

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

    def tokenize_prompt(self, prompt_text):
        tokens = prompt_text.split()
        token_ids = []

        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<UNK>"])

        return torch.tensor([self.vocab["<BOS>"]] + token_ids, dtype=torch.long)

    def decode_tokens(self, token_ids):
        words = []
        for token_id in token_ids:
            word = self.reverse_vocab.get(token_id, "<UNK>")

            if word not in ["<BOS>", "<PAD>", "<UNK>"]:
                words.append(word)

        return " ".join(words)

    def generate_greedy(self, prompt_tokens, max_tokens=50):
        tokens = prompt_tokens.clone().to(self.device)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(tokens.unsqueeze(0))
                next_logits = logits[0, -1, :]

                next_token = torch.argmax(next_logits)

                tokens = torch.cat([tokens, next_token.unsqueeze(0)])

                if next_token.item() == self.vocab.get("<EOS>", 3):
                    break

        return tokens

    def generate_sampling(self, prompt_tokens, max_tokens=50, temperature=0.8, top_k=None):
        tokens = prompt_tokens.clone().to(self.device)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(tokens.unsqueeze(0))
                next_logits = logits[0, -1, :] / temperature

                if top_k:
                    top_k_logits, top_k_indices = torch.topk(next_logits, k=min(top_k, len(self.vocab)))
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits[top_k_indices] = top_k_logits

                probs = torch.softmax(next_logits, dim=0)
                next_token = torch.multinomial(probs, num_samples=1)

                tokens = torch.cat([tokens, next_token])

                if next_token.item() == self.vocab.get("<EOS>", 3):
                    break

        return tokens

    def generate_beam_search(self, prompt_tokens, max_tokens=50, beam_width=3):
        sequences = [(prompt_tokens.clone().to(self.device), 0.0)]

        with torch.no_grad():
            for step in range(max_tokens):
                candidates = []

                for tokens, log_prob in sequences:
                    logits = self.model(tokens.unsqueeze(0))
                    next_logits = logits[0, -1, :]

                    top_logits, top_indices = torch.topk(next_logits, k=beam_width)
                    top_probs = torch.log_softmax(top_logits, dim=0)

                    for prob, idx in zip(top_probs, top_indices):
                        new_tokens = torch.cat([tokens, idx.unsqueeze(0)])
                        new_log_prob = log_prob + prob.item()
                        candidates.append((new_tokens, new_log_prob))

                candidates.sort(key=lambda x: x[1], reverse=True)
                sequences = candidates[:beam_width]

                if all(seq[0][-1].item() == self.vocab.get("<EOS>", 3) for seq in sequences):
                    break

        best_tokens, _ = sequences[0]
        return best_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained LLM")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to continue from")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=focused, 1.0=normal, 2.0=creative)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Only sample from top K tokens")
    parser.add_argument("--method", type=str, default="sampling",
                        choices=["greedy", "sampling", "beam"],
                        help="Generation method")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print(f"\n📂 Loading model: {args.model}")
    generator = TextGenerator(args.model, device)
    print("✓ Model loaded\n")

    print(f"💭 Prompt: {args.prompt}")
    prompt_tokens = generator.tokenize_prompt(args.prompt)

    print(f"🔮 Generating ({args.method})...\n")

    if args.method == "greedy":
        generated_tokens = generator.generate_greedy(prompt_tokens, args.max_tokens)
    elif args.method == "beam":
        generated_tokens = generator.generate_beam_search(prompt_tokens, args.max_tokens)
    else:
        generated_tokens = generator.generate_sampling(
            prompt_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    generated_text = generator.decode_tokens(generated_tokens)

    print("=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)

    print(f"\n⚙️  Parameters:")
    print(f"  Method: {args.method}")
    if args.method == "sampling":
        print(f"  Temperature: {args.temperature}")
        if args.top_k:
            print(f"  Top-K: {args.top_k}")
    print(f"  Max tokens: {args.max_tokens}")

    print("\n💡 Try different temperatures:")
    print("  - 0.2: Very focused (may be repetitive)")
    print("  - 0.7: Good balance (recommended)")
    print("  - 1.5: Creative (may be nonsensical)")


if __name__ == "__main__":
    main()
