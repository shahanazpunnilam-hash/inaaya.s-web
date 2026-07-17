# Getting Started with LLM Training

Complete step-by-step guide for your first LLM training!

## 🎯 5-Minute Overview

Training a small LLM requires 3 main things:

1. **Text Data** - Hundreds of text files or thousands of sentences
2. **Python Setup** - Install required libraries
3. **Run Scripts** - 5 command-line commands

That's it! Here's the flow:

```
Raw Text Data
     ↓
[1] Prepare & Tokenize (5 min)
     ↓
Training Dataset
     ↓
[2] Train Model (1-4 hours)
     ↓
Trained Model
     ↓
[3] Evaluate (5 min)
     ↓
Results & Insights
     ↓
[4] Benchmark vs ChatGPT/Claude (5 min)
     ↓
Comparison Scores
     ↓
[5] Generate Text (1 min)
```

---

## 🛠️ Step 0: Setup Your Environment

### Option A: Windows/Mac/Linux with Python

**1. Check Python version**
```bash
python --version
# Should be 3.8 or higher
```

**2. Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

This installs:
- `torch` - Deep learning framework
- `transformers` - Pre-built models
- `numpy`, `pandas` - Data handling
- Other utilities

### Option B: Google Colab (Free GPU!)

If you don't have a computer with good specs:

1. Go to https://colab.research.google.com
2. Upload this folder to Colab
3. Run cells instead of command line

**Pro tip**: Google Colab gives you free GPU access for 12 hours at a time!

---

## 📊 Step 1: Prepare Your Data (5 minutes)

### Before you start:
- Decide on your dataset (see options below)
- Place text files in the `data/` folder

### Text File Formats Supported:
- `.txt` files (plain text)
- Multiple files in one folder
- Any language (English, Spanish, etc.)

### Example Datasets to Try

#### Option 1: Use the Sample Data
```bash
# Already in data/sample.txt
# Good for learning, but too small for a good model
```

#### Option 2: Download Free Data
**Wikipedia Dump** (most popular)
- Go to https://dumps.wikimedia.org/enwiki/
- Download a small dump (e.g., articles subset)
- Extract and copy .txt files to `data/`

**Project Gutenberg** (classic books)
- Go to https://www.gutenberg.org/
- Download public domain books
- Save as .txt files in `data/`

**ArXiv Papers** (scientific papers)
- Go to https://arxiv.org/
- Use `arxiv-download` tool to get abstracts
- Save to `data/`

**Your Own Data**
- Collect blog posts, news articles, etc.
- Save each article as a .txt file
- Put in `data/` folder

### Data Size Guidelines

For the included code:
- **Minimum**: 10 MB (about 100,000 words) - very small model
- **Recommended**: 100 MB - 1 GB (1-10 million words) - good quality
- **Ideal**: 1-10 GB+ - excellent quality

**Rule of thumb**: 1 MB of text ≈ 200 words

### Run Data Preparation

Once you have text files in `data/`:

```bash
python dataset_preparation.py
```

**What this does:**
1. Loads all .txt files from `data/`
2. Cleans the text (removes extra spaces, etc.)
3. Splits into train/validation/test (80/10/10)
4. Creates a vocabulary (list of unique words)
5. Converts words to numbers (tokenization)
6. Saves to `processed_data/`

**Output files created:**
- `processed_data/train.pt` - 80% of data for training
- `processed_data/val.pt` - 10% for validation
- `processed_data/test.pt` - 10% for testing
- `processed_data/vocab.json` - Word mappings

**Progress output:**
```
==================================================
TEXT DATA PROCESSOR
==================================================
📂 Loading text files...
✓ Loaded 1 text file(s)
🧹 Cleaning text...
✓ Total characters: 50,000
🔤 Building vocabulary...
✓ Vocabulary size: 5,000
🔢 Tokenizing texts...
✓ Created 1,200 sequences of length 128
...
✅ Dataset preparation complete!
```

---

## 🏋️ Step 2: Train Your Model (1-4 hours)

The training script will:
1. Load your processed data
2. Build a neural network
3. Train it on your data
4. Save checkpoints every epoch
5. Track progress with loss metrics

### Basic Training

```bash
python train.py
```

This uses default settings:
- 10 epochs
- Batch size 32
- Learning rate 0.001

### Customized Training

```bash
python train.py \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --embedding_dim 256 \
  --hidden_dim 512
```

### Parameters Explained

**Training-related:**
- `--epochs` (default: 10) - How many times to go through data
- `--batch_size` (default: 32) - How many examples per step
- `--learning_rate` (default: 0.001) - How fast it learns

**Model architecture:**
- `--embedding_dim` (default: 128) - Size of word vectors
- `--hidden_dim` (default: 256) - Size of neural network
- `--num_layers` (default: 2) - Number of LSTM layers
- `--dropout` (default: 0.3) - Prevent overfitting

### Tweaking for Your Hardware

**If training is too slow:**
```bash
# Use smaller model
python train.py --batch_size 8 --embedding_dim 64 --hidden_dim 128
```

**If running out of memory:**
```bash
# Reduce batch size significantly
python train.py --batch_size 4
```

**For faster training (if you have GPU):**
```bash
# The script auto-detects GPU
# No special flags needed!
python train.py --batch_size 64  # Larger batches are faster
```

### What You'll See

During training:

```
Epoch 1/10
Training: 100%|████████| 125/125 [02:34<00:00,  1.24s/it]
  Loss: 4.892 | Perplexity: 132.45
  Val Loss: 4.234 | Val Perplexity: 68.92
  ✨ Best model saved!

Epoch 2/10
Training: 100%|████████| 125/125 [02:33<00:00,  1.23s/it]
  Loss: 4.123 | Perplexity: 61.32
  Val Loss: 3.876 | Val Perplexity: 48.21
  ...
```

### Understanding the Metrics

**Loss**: How wrong the model's predictions are
- Should decrease over time
- If not decreasing: try higher learning rate or more data

**Perplexity**: How "confused" the model is
- About the predicted next word
- Lower is better (target: < 50)
- Related to loss by: perplexity = e^loss

**Early stopping tip**:
- If loss plateaus, you can stop early
- Or increase learning rate and continue

### Saved Files

After training:

```
models/
├── model_epoch_1.pt      ← Checkpoint after epoch 1
├── model_epoch_2.pt      ← Checkpoint after epoch 2
├── best_model.pt         ← Best model overall (use this!)
└── ...
```

The `best_model.pt` has the lowest validation loss.

---

## 📈 Step 3: Evaluate Your Model (5 minutes)

Test how well your model learned:

```bash
python evaluate.py
```

This will:
1. Load your best trained model
2. Test on the test set (data it never saw)
3. Calculate performance metrics
4. Show sample predictions

### What You'll See

```
==================================================
MODEL EVALUATION
==================================================
📊 Evaluating on test set...

==================================================
RESULTS
==================================================
Test Loss: 3.456
Test Perplexity: 31.82
  (Lower is better. < 100 is good, < 50 is excellent)
Test Accuracy: 62.34%
  (% of correctly predicted next tokens)

🔮 Sample Predictions (first 5 tokens):
Position 0:
  Actual next token: fox
  Model predicts (top 5):
    1. fox (confidence: 2.145)
    2. dog (confidence: 1.932)
    3. cat (confidence: 1.234)
    ...
```

### Interpreting Results

**Perplexity < 50**: Your model is learning well!
**Perplexity 50-100**: Still decent, try:
- More training data
- More epochs
- Larger model

**Perplexity > 200**: Something's wrong, check:
- Data quality (clean text files)
- Data size (need more?)
- Learning rate (too high?)

**Low accuracy (< 40%)**: This is normal for small models. Don't worry!

---

## 🏆 Step 4: Benchmark Against ChatGPT/Claude (5-10 minutes)

After training, compare your model against larger language models:

```bash
python benchmark.py
```

**What this does:**
1. Measures your model's perplexity
2. Calculates token accuracy
3. Tests text diversity
4. Checks coherence
5. Compares to GPT-2 and Claude-like models

**Example output:**
```
📊 RESULTS COMPARISON:
Metric               Your Model           GPT-2                Claude-like
Perplexity           45.23                50.0                 25.0
Accuracy             62.34%               75.0%                85.0%
Diversity            0.42                 0.45                 0.55
Coherence            58.21%               65.0%                80.0%
```

**What the metrics mean:**
- **Perplexity**: How confused the model is (lower = better)
- **Accuracy**: % of correct next-token predictions
- **Diversity**: How varied generated text is
- **Coherence**: Model's confidence in predictions

This helps you understand your model's performance relative to state-of-the-art models!

---

## 🔮 Step 5: Generate Text! (1 minute)

Use your trained model to generate text:

```bash
python inference.py --prompt "Once upon a time"
```

**Example output:**
```
GENERATED TEXT
======================================================================
Once upon a time there was a young girl in the forest. She lived with
her family near a big tree. The tree was very old and tall. The girl
played under the tree every day. She made friends with the forest animals
...
```

### Fun Experiments

**More creative (higher temperature):**
```bash
python inference.py --prompt "The future is" --temperature 1.5
```

**More focused (lower temperature):**
```bash
python inference.py --prompt "The future is" --temperature 0.3
```

**Longer text:**
```bash
python inference.py --prompt "In a galaxy far far away" --max_tokens 200
```

**Different generation methods:**
```bash
# Greedy (deterministic, may repeat)
python inference.py --prompt "Hello" --method greedy

# Beam search (slower but higher quality)
python inference.py --prompt "Hello" --method beam
```

### Temperature Explained

Temperature controls randomness:

- **0.1**: Very focused, deterministic
  - Output: "The weather is the weather is the weather is..."
  - Use for: predictable tasks

- **0.7**: Good balance (recommended)
  - Output: Coherent but somewhat varied
  - Use for: general text generation

- **1.5**: Very creative, random
  - Output: Often nonsensical or very creative
  - Use for: brainstorming ideas

Try different values and see what you like!

---

## 🚀 Summary: Quick Reference

### Complete Workflow

```bash
# 1. Prepare data (one-time)
python dataset_preparation.py

# 2. Train model
python train.py --epochs 20

# 3. Evaluate
python evaluate.py

# 4. Benchmark against ChatGPT/Claude
python benchmark.py

# 5. Generate text
python inference.py --prompt "Your prompt here"
```

### File Structure

```
llm-training-template/
├── README.md                      ← Full documentation
├── GETTING_STARTED.md             ← You are here
├── requirements.txt               ← Install: pip install -r requirements.txt
├── data/                          ← PUT YOUR TEXT FILES HERE
│   └── sample.txt
├── dataset_preparation.py         ← Step 1: Run this first
├── train.py                       ← Step 2: Training
├── evaluate.py                    ← Step 3: Testing
├── inference.py                   ← Step 4: Generate text
├── processed_data/                ← Auto-created by script
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   └── vocab.json
└── models/                        ← Auto-created by script
    ├── best_model.pt              ← Use this one!
    └── model_epoch_*.pt
```

---

## ❓ Common Questions

### Q: How long does training take?
**A:** 
- CPU: 30 min - 4 hours (depends on dataset size)
- GPU: 5 - 30 minutes
- Use smaller dataset/model for faster results

### Q: Do I need a GPU?
**A:** 
- No, CPU works fine for small datasets
- GPU is 5-10x faster
- Free GPU in Google Colab!

### Q: What if training seems stuck?
**A:** 
- Check if loss is decreasing (even slowly)
- If stuck, press Ctrl+C to stop
- Try smaller learning rate and more epochs

### Q: Can I use this for a different language?
**A:** 
- Yes! Provide text files in any language
- Works with English, Spanish, French, etc.
- Even works with code or poetry!

### Q: How do I improve results?
**A:** 
1. Add more training data (bigger is better)
2. Train for more epochs
3. Make the model bigger (higher hidden_dim)
4. Adjust learning rate based on loss curve

### Q: What's the difference between training loss and validation loss?
**A:** 
- **Training loss**: Calculated on data model saw during training
- **Validation loss**: Calculated on data model hasn't seen
- If validation loss >> training loss: model is overfitting
- Solution: reduce model size or use more data

---

## 🎓 Learning Path

After you complete this template:

1. **Try different datasets** - Books, Wikipedia, code, poetry
2. **Experiment with parameters** - See how they affect results
3. **Use pre-trained models** - Start from GPT-2 instead of training from scratch
4. **Deploy your model** - Make a website where people can use it
5. **Advanced topics** - Transformers, attention, fine-tuning

---

## 🤝 Need Help?

**Error: "No module named torch"**
- Solution: `pip install -r requirements.txt`

**Error: "data directory not found"**
- Solution: Create `data/` folder and add .txt files

**Error: "CUDA out of memory"**
- Solution: Reduce batch_size (e.g., --batch_size 8)

**Model generating random text**
- Solution: Train for more epochs, use more data

**Still stuck?**
- Check the full README.md
- Review each script's comments
- Try Google Colab (more resources)

---

Good luck! 🚀 You're about to train your first LLM!
