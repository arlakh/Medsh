# Phi-2 Fine-tuning for Creative Writing

A comprehensive fine-tuning setup for Microsoft's Phi-2 model to create a personalized writing assistant that learns from your book collection.

## 🚀 Features

- **LoRA/QLoRA Fine-tuning**: Efficient training with minimal memory requirements
- **Automatic Dataset Preparation**: Process .txt files from Project Gutenberg or your own books
- **Smart Text Processing**: Clean and chunk text for optimal training
- **Gradio Web UI**: Beautiful interface for text generation with multiple modes
- **Multiple Generation Modes**: Continue stories, write chapters, suggest characters
- **Model Management**: Save, load, and test trained models

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (for training)
- 8GB+ VRAM (for inference)

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create directories**:
   ```bash
   mkdir books models outputs
   ```

## 📚 Getting Started

### 1. Prepare Your Books

Place your `.txt` book files in the `books/` directory, or download from Project Gutenberg:

```bash
# Download sample books from Project Gutenberg
python phi2_finetuning.py --download_books 1342 11 1661
```

Popular Project Gutenberg IDs:
- `1342` - Pride and Prejudice
- `11` - Alice in Wonderland  
- `1661` - The Count of Monte Cristo
- `84` - Frankenstein
- `98` - A Tale of Two Cities

### 2. Train the Model

```bash
# Basic training
python phi2_finetuning.py --mode train

# Training with test after completion
python phi2_finetuning.py --mode train --test
```

### 3. Use the Writing Assistant

```bash
# Launch Gradio UI
python phi2_finetuning.py --mode ui

# Command-line inference
python phi2_finetuning.py --mode inference
```

## 🎯 Usage Examples

### Training Pipeline

```python
# The script automatically:
# 1. Loads and processes your books
# 2. Sets up Phi-2 with LoRA
# 3. Trains the model
# 4. Saves the fine-tuned model
```

### Web Interface Features

1. **Continue Story**: Provide a story beginning and let the AI continue it
2. **Write Chapter**: Generate new chapters with title and genre
3. **Character Suggestions**: Create character descriptions for your story world

### Command Line Usage

```bash
# Download books and train
python phi2_finetuning.py --download_books 1342 11 --mode train

# Launch UI after training
python phi2_finetuning.py --mode ui

# Test the model
python phi2_finetuning.py --mode inference
```

## ⚙️ Configuration

### LoRA Parameters (in `phi2_finetuning.py`)

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                    # Rank (higher = more parameters)
    lora_alpha=32,           # Alpha parameter
    lora_dropout=0.1,        # Dropout rate
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
)
```

### Training Parameters

```python
training_args = TrainingArguments(
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=2,   # Batch size per device
    gradient_accumulation_steps=4,   # Gradient accumulation
    learning_rate=2e-4,              # Learning rate
    warmup_steps=100,                # Warmup steps
)
```

## 📁 Project Structure

```
phi2-finetuning/
├── phi2_finetuning.py      # Main script
├── requirements.txt         # Dependencies
├── README.md              # This file
├── books/                 # Your book files (.txt)
├── models/                # Saved models
└── outputs/               # Training outputs
```

## 🎨 Generation Parameters

### Temperature
- **0.1-0.5**: More focused, deterministic text
- **0.8-1.2**: More creative, diverse text
- **Default**: 0.8

### Top-p (Nucleus Sampling)
- **0.1-0.5**: More focused on high-probability tokens
- **0.9**: Good balance of creativity and coherence
- **Default**: 0.9

### Max Length
- **100-300**: Short continuations
- **500-800**: Medium-length text
- **1000+**: Long-form content

## 🔧 Advanced Usage

### Custom Training

```python
# Modify training parameters in phi2_finetuning.py
training_args = TrainingArguments(
    num_train_epochs=5,              # More epochs
    per_device_train_batch_size=1,   # Smaller batch size
    learning_rate=1e-4,              # Lower learning rate
    # ... other parameters
)
```

### Different Model Sizes

The script uses Phi-2 (2.7B parameters). For different models:

```python
# Change model_name in setup_model_and_tokenizer()
model_name = "microsoft/phi-1_5"  # Smaller model
# or
model_name = "microsoft/phi-2"    # Current model
```

### Custom Text Processing

Modify the `clean_text()` function to handle your specific text format:

```python
def clean_text(text: str) -> str:
    # Add your custom cleaning logic here
    # Remove specific headers, footers, etc.
    return cleaned_text
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `per_device_train_batch_size`
   - Reduce `max_length` in tokenization
   - Use gradient accumulation

2. **Slow Training**
   - Enable mixed precision training
   - Use `flash-attn` for faster attention
   - Reduce dataset size for testing

3. **Poor Generation Quality**
   - Increase training epochs
   - Adjust LoRA rank (`r` parameter)
   - Use better quality training data

### Memory Requirements

- **Training**: 16GB+ RAM, 8GB+ VRAM
- **Inference**: 8GB+ RAM, 4GB+ VRAM
- **Model Size**: ~3GB (Phi-2)

## 📊 Performance Tips

1. **Data Quality**: Use high-quality, well-formatted text files
2. **Data Quantity**: More books = better results (aim for 10+ books)
3. **Training Time**: Expect 2-6 hours depending on dataset size
4. **GPU Usage**: Use CUDA for faster training

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational and personal use. Please respect the licenses of the base models and training data.

## 🙏 Acknowledgments

- Microsoft for Phi-2 model
- Hugging Face for transformers and datasets
- Project Gutenberg for public domain books
- Gradio for the web interface

---

**Happy Writing! 📝✨**
