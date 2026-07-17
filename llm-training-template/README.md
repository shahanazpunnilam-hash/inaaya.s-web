# Small LLM Training Template

A beginner-friendly guide to train your own small language model! This template walks you through every step needed to create a working LLM.

## What You'll Learn

This template teaches you:
- How to prepare and load data
- How to tokenize text for models
- How to build and train a neural network
- How to evaluate and use your trained model

## What You Need

**Before starting, make sure you have:**
1. **Python** (version 3.8 or higher)
2. **A dataset** (text files, CSV, or JSON - see "Dataset Preparation" below)
3. **A computer with reasonable specs** (GPU recommended but CPU works for small models)

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all the libraries you need:
- `torch`: Deep learning framework
- `transformers`: Pre-built models from Hugging Face
- `numpy`: Data manipulation
- `pandas`: Data handling
- `tqdm`: Progress bars

### Step 2: Prepare Your Data

```bash
python dataset_preparation.py
```

This script will:
- Load your raw text data
- Clean and preprocess it
- Split it into training/validation/test sets
- Tokenize the text (convert words to numbers the model understands)

See **Dataset Preparation** section below for details.

### Step 3: Train Your Model

```bash
python train.py --epochs 10 --batch_size 32 --learning_rate 0.0001
```

The model will:
- Load your prepared dataset
- Build a neural network
- Train it on your data
- Save the best version

Training typically takes 30 minutes to several hours depending on dataset size.

### Step 4: Evaluate and Test

```bash
python evaluate.py
```

This will:
- Test your model's performance
- Show you example predictions
- Calculate accuracy metrics

### Step 5: Use Your Model

```bash
python inference.py --prompt "Once upon a time"
```

Your trained model will generate text starting with your prompt!

---

## Dataset Preparation

### What Data Do You Need?

Your model learns from text data. You can use:
- **Plain text files** (.txt) - one file or many
- **CSV files** - with a column containing text
- **JSON files** - with text in a specific field
- **Books, articles, or any written content**

### Recommended Dataset Sizes

- **Small model** (for learning): 50 MB - 500 MB of text
- **Medium model** (good results): 500 MB - 5 GB
- **Large model** (high quality): 5 GB+

### Data Sources

Here are some free, open datasets:

1. **Wikitext** - Wikipedia articles
   ```bash
   # Included in Hugging Face Datasets
   ```

2. **Common Crawl** - Web content
   https://commoncrawl.org/

3. **Project Gutenberg** - Free books
   https://www.gutenberg.org/

4. **ArXiv** - Scientific papers
   https://arxiv.org/

5. **Your Own Data** - Collect text files yourself

### How Data Preparation Works

```
Raw Text Files
       ↓
   Cleaning (remove extra spaces, special chars)
       ↓
   Tokenization (split text into tokens/words)
       ↓
   Encoding (convert tokens to numbers)
       ↓
   Split into Train/Validation/Test
       ↓
  Ready for Training!
```

### Preparing Your Data

1. **Create a `data/` directory** with your text files:
   ```
   data/
   ├── file1.txt
   ├── file2.txt
   └── file3.txt
   ```

2. **Run the preparation script**:
   ```bash
   python dataset_preparation.py
   ```

3. **It creates**:
   - `processed_data/train.pt` - training data (80%)
   - `processed_data/val.pt` - validation data (10%)
   - `processed_data/test.pt` - test data (10%)
   - `processed_data/vocab.json` - word mappings

---

## Training Your Model

### What Happens During Training?

Training is like teaching someone to predict the next word:

```
Input: "The quick brown"
Model predicts: "fox" (or similar word)
Actual: "fox"
Model learns from the difference
Repeat millions of times
```

### Training Parameters Explained

- **Epochs**: How many times to go through the entire dataset (10-50 typical)
- **Batch Size**: How many examples to process at once (16-64 typical)
- **Learning Rate**: How fast the model learns (0.0001-0.001 typical)
- **Embedding Dim**: Size of word representations (128-512)
- **Hidden Dim**: Size of neural network layers (256-1024)

### Example Training Run

```bash
python train.py \
  --epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --embedding_dim 256 \
  --hidden_dim 512
```

### Monitoring Training

During training, you'll see:
```
Epoch 1/20
Loss: 5.234  ← Lower is better
Val Loss: 4.892

Epoch 2/20
Loss: 4.156
Val Loss: 3.987

... model is learning!
```

If loss isn't decreasing:
- Increase learning rate slightly
- Try more epochs
- Check your data quality

---

## Evaluating Your Model

### What Gets Measured?

The evaluation script calculates:
- **Perplexity**: How "confused" the model is (lower is better, <100 is good)
- **Accuracy**: Percentage of correct next-word predictions
- **Loss**: Overall error metric

### Example Output

```
Test Perplexity: 45.23
Test Accuracy: 68.5%
Sample Generation:
Input: "Once upon a time"
Output: "Once upon a time there was a young girl..."
```

---

## Using Your Trained Model

### Generate Text

```python
from inference import generate_text

# Generate 50 tokens from a prompt
result = generate_text("The future of AI", max_tokens=50)
print(result)
```

### Fine-tune on New Data

```bash
python train.py --checkpoint model_epoch_10.pt --data new_data/
```

### Convert for Production

Your saved model can be:
- Deployed as a REST API (using Flask)
- Integrated into a web app
- Run on edge devices (with optimization)

---

## 🏆 Benchmark Your Model

After training and evaluating, test how your model compares to ChatGPT and Claude:

```bash
python benchmark.py
```

### What Gets Measured

1. **Perplexity** - How confused the model is (lower is better)
2. **Accuracy** - % of correctly predicted next tokens
3. **Diversity** - How varied the generated text is
4. **Coherence** - How confident the model is in predictions

### Example Output

```
📊 RESULTS COMPARISON:
Metric               Your Model           GPT-2                Claude-like
Perplexity           45.23                50.0                 25.0
Accuracy             62.34%               75.0%                85.0%
Diversity            0.42                 0.45                 0.55
Coherence            58.21%               65.0%                80.0%
```

### How It Works

- **Your Model**: Measured on your test set
- **GPT-2 (estimated)**: Based on published benchmarks
- **Claude-like (estimated)**: Simulated scores for comparison

These comparisons help you understand where your model stands relative to larger, more sophisticated models.

---

## Troubleshooting

### Problem: GPU out of memory
**Solution**: Reduce batch size (e.g., 8 instead of 32)

### Problem: Training is too slow
**Solution**: 
- Reduce dataset size
- Reduce model size (lower hidden_dim)
- Use GPU instead of CPU

### Problem: Model generates random text
**Solution**:
- Train for more epochs
- Use more/better quality data
- Increase model size

### Problem: Model overfits (great training loss, bad test loss)
**Solution**:
- Reduce model size
- Use more data
- Add dropout (regularization)

---

## Project Structure

```
llm-training-template/
├── README.md                 ← You are here
├── requirements.txt          ← Python dependencies
├── dataset_preparation.py    ← Data loading & preprocessing
├── train.py                  ← Training script
├── evaluate.py               ← Evaluation script
├── inference.py              ← Generate text with trained model
├── data/                     ← Put your raw text files here
│   └── sample.txt
├── processed_data/           ← Auto-created by preparation script
│   ├── train.pt
│   ├── val.pt
│   └── vocab.json
└── models/                   ← Auto-created during training
    └── model_epoch_1.pt
```

---

## Learning More

### Recommended Resources

1. **Hugging Face Course** - Free online course
   https://huggingface.co/course/

2. **Fast.ai** - Practical deep learning
   https://www.fast.ai/

3. **Papers with Code** - ML research papers
   https://paperswithcode.com/

4. **Stanford CS224N** - NLP course (free lectures)
   https://web.stanford.edu/class/cs224n/

### Key Concepts

- **Tokenization**: Converting text to numbers
- **Embeddings**: Representing words as vectors
- **Transformer**: Modern architecture for LLMs
- **Attention Mechanism**: How models focus on relevant words
- **Fine-tuning**: Training a pre-trained model on your data

---

## Next Steps

After mastering this template:

1. **Use Pre-trained Models** - Start from models like GPT-2 or BERT
2. **Scale Up** - Train on larger datasets
3. **Advanced Architectures** - Try Transformers and Attention
4. **Deployment** - Create a web API for your model
5. **Research** - Explore cutting-edge techniques

---

## Notes for Teachers

This template is designed for:
- High school students learning AI basics
- College CS/ML students
- Anyone curious about how LLMs work

**Teaching tips:**
- Start with a small dataset (< 100 MB)
- Have students modify hyperparameters to see effects
- Discuss why certain datasets produce certain outputs
- Show famous LLM limitations (bias, hallucinations)
- Compare generated text quality across datasets

---

## License

This template is free to use and modify for educational purposes.

Happy training! 🚀
