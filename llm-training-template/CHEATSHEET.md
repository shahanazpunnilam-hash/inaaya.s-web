# LLM Training Cheat Sheet

Quick reference for the most common commands.

## Setup (Do This Once)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put text files in data/ folder
# data/file1.txt
# data/file2.txt
# etc.
```

## The 4-Step Workflow

### Step 1: Prepare Data
```bash
python dataset_preparation.py
```
- Loads text files from `data/`
- Creates training/validation/test splits
- Saves to `processed_data/`
- Takes: 5 minutes

### Step 2: Train Model
```bash
# Basic
python train.py

# Custom
python train.py --epochs 20 --batch_size 16 --learning_rate 0.0005
```
- Trains neural network on your data
- Saves checkpoints and best model
- Takes: 30 min to 4 hours

### Step 3: Evaluate
```bash
python evaluate.py
```
- Tests model performance
- Shows metrics and sample predictions
- Takes: 5 minutes

### Step 4: Evaluate & Benchmark
```bash
python evaluate.py
```
- Tests model performance
- Takes: 5 minutes

### Step 5: Compare to ChatGPT/Claude
```bash
python benchmark.py
```
- Compares your model to large language models
- Shows detailed metrics and insights
- Takes: 5-10 minutes

### Step 6: Generate Text
```bash
python inference.py --prompt "Your text here"
```
- Generates text starting with your prompt
- Takes: 1 minute

## Common Commands

### Training Variations

**Fast training (for testing)**
```bash
python train.py --epochs 3 --batch_size 8 --embedding_dim 64 --hidden_dim 128
```

**Slow but good quality**
```bash
python train.py --epochs 50 --batch_size 32 --embedding_dim 256 --hidden_dim 512
```

**Resume from checkpoint**
```bash
python train.py --epochs 20 --checkpoint models/model_epoch_5.pt
```

### Text Generation Variations

**Creative text**
```bash
python inference.py --prompt "Once upon a time" --temperature 1.5 --max_tokens 100
```

**Focused/predictable**
```bash
python inference.py --prompt "Once upon a time" --temperature 0.3 --max_tokens 50
```

**Using beam search (slower, better quality)**
```bash
python inference.py --prompt "Once upon a time" --method beam --max_tokens 100
```

**Greedy decoding (fastest)**
```bash
python inference.py --prompt "Once upon a time" --method greedy --max_tokens 100
```

## Parameters Quick Reference

### Dataset Preparation
- No parameters (uses defaults)

### Training
| Parameter | Default | What It Does | Try These |
|-----------|---------|--------------|-----------|
| `--epochs` | 10 | How many times through data | 5, 10, 20, 50 |
| `--batch_size` | 32 | Examples per step | 8, 16, 32, 64 |
| `--learning_rate` | 0.001 | How fast it learns | 0.0001, 0.0005, 0.001 |
| `--embedding_dim` | 128 | Word vector size | 64, 128, 256 |
| `--hidden_dim` | 256 | Neural net size | 128, 256, 512 |
| `--num_layers` | 2 | LSTM layers | 1, 2, 3 |
| `--dropout` | 0.3 | Prevent overfitting | 0.1, 0.3, 0.5 |

### Inference
| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--prompt` | (required) | Starting text |
| `--temperature` | 0.8 | Randomness (0.1-2.0) |
| `--max_tokens` | 50 | How much to generate |
| `--method` | sampling | greedy/sampling/beam |
| `--top_k` | None | Only sample top K tokens |

## Troubleshooting

### Problem: Training is slow
**Solution 1**: Use smaller model
```bash
python train.py --embedding_dim 64 --hidden_dim 128 --batch_size 8
```
**Solution 2**: Use less data or fewer epochs
```bash
python train.py --epochs 3
```

### Problem: "Out of memory"
**Solution**: Reduce batch size
```bash
python train.py --batch_size 4
```

### Problem: Loss not decreasing
**Solution 1**: Increase learning rate
```bash
python train.py --learning_rate 0.01
```
**Solution 2**: Add more data (more text files)
**Solution 3**: Train longer
```bash
python train.py --epochs 50
```

### Problem: Model generates gibberish
**Solution 1**: Train more
```bash
python train.py --epochs 30
```
**Solution 2**: Use more data
**Solution 3**: Make model bigger
```bash
python train.py --hidden_dim 512 --embedding_dim 256
```

### Problem: File not found errors
**Solution**: Check you're in the right directory
```bash
# You should see:
# - dataset_preparation.py
# - train.py
# - evaluate.py
# - inference.py
# - data/
# - models/
# - processed_data/
```

## File Locations

```
llm-training-template/
├── data/                    ← YOUR TEXT FILES GO HERE
├── processed_data/          ← AUTO-CREATED (train/val/test data)
├── models/                  ← AUTO-CREATED (saved models)
│   └── best_model.pt        ← USE THIS FOR INFERENCE
└── *.py                     ← Scripts to run
```

## Quick Metrics Guide

After training, `evaluate.py` shows:

**Perplexity**: How confused the model is
- < 20: Excellent
- 20-50: Very good
- 50-100: Good
- > 200: Needs work

**Accuracy**: % of correct next-word predictions
- > 70%: Great
- 50-70%: Good
- < 50%: Keep training

**Loss**: Raw error metric
- Should go down during training
- If not: fix learning rate or data

## Experiments to Try

1. **Different datasets**
   - Try Wikipedia vs. novels vs. your own writing
   - See how results differ

2. **Parameter tuning**
   - Change learning rate: 0.0001 → 0.01
   - Change model size: 128 → 1024
   - See what improves results

3. **Generation strategies**
   - Greedy vs. sampling vs. beam search
   - Different temperatures
   - See quality differences

4. **Domain-specific models**
   - Train on Shakespeare → generate poetry
   - Train on code → generate code
   - Train on emails → generate emails

## Resources

- **Full Guide**: See `README.md`
- **Step-by-Step**: See `GETTING_STARTED.md`
- **Script Documentation**: Each `.py` file has comments explaining code

## Command History

Keep track of what works:

```bash
# What I tried:
python train.py --epochs 20 --batch_size 32 --learning_rate 0.0005
# Result: Perplexity 45.2, Accuracy 62%
# Good! Keep these settings.

python train.py --epochs 5 --learning_rate 0.01
# Result: Loss not decreasing, learning rate too high
# Don't use this.
```

---

**Happy training!** 🚀
