# Phi-2 Fine-tuning for Creative Writing - Colab Version
# Run each section in order in Google Colab

# ============================================================================
# SECTION 1: Install Dependencies
# ============================================================================

!pip install -q transformers==4.36.0
!pip install -q accelerate==0.25.0
!pip install -q peft==0.7.1
!pip install -q datasets==2.14.0
!pip install -q bitsandbytes==0.41.1
!pip install -q gradio==4.7.1
!pip install -q torch==2.1.0
!pip install -q sentencepiece==0.1.99
!pip install -q protobuf==3.20.3

# Restart runtime after installation
import os
os.kill(os.getpid(), 9)

# ============================================================================
# SECTION 2: Import Libraries and Setup
# ============================================================================

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Hugging Face libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import gradio as gr

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("books", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================================
# SECTION 3: Dataset Preparation Functions
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and preprocess text from books."""
    # Remove Project Gutenberg headers/footers
    text = re.sub(r'\*\*\* START OF .*?\*\*\*', '', text)
    text = re.sub(r'\*\*\* END OF .*?\*\*\*', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?;:"\'\-()]', '', text)
    
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 2048, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for training."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def prepare_dataset_from_files(book_dir: str = "books", chunk_size: int = 2048) -> Dataset:
    """Prepare dataset from .txt files in the books directory."""
    all_texts = []
    
    # Find all .txt files
    txt_files = list(Path(book_dir).glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found in books directory!")
        print("Please upload your book files to the 'books' folder.")
        return None
    
    print(f"Found {len(txt_files)} book files:")
    for file in txt_files:
        print(f"  - {file.name}")
    
    # Process each file
    for file_path in txt_files:
        print(f"Processing {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Split into chunks
            chunks = split_into_chunks(cleaned_text, chunk_size)
            all_texts.extend(chunks)
            
            print(f"  Created {len(chunks)} chunks from {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nTotal chunks created: {len(all_texts)}")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": all_texts})
    return dataset

def download_books_from_gutenberg(book_ids: List[str]):
    """Download books from Project Gutenberg."""
    import requests
    
    for book_id in book_ids:
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url)
            
            if response.status_code == 200:
                filename = f"books/gutenberg_{book_id}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"Downloaded book {book_id} to {filename}")
            else:
                print(f"Failed to download book {book_id}")
                
        except Exception as e:
            print(f"Error downloading book {book_id}: {e}")

# ============================================================================
# SECTION 4: Download Sample Books (Optional)
# ============================================================================

# Uncomment and modify to download books from Project Gutenberg
# download_books_from_gutenberg(["1342", "11", "1661"])  # Pride and Prejudice, Alice in Wonderland, The Count of Monte Cristo

# ============================================================================
# SECTION 5: Prepare Dataset
# ============================================================================

print("Preparing dataset...")
dataset = prepare_dataset_from_files()
if dataset:
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sample text: {dataset[0]['text'][:200]}...")
else:
    print("No dataset available. Please add .txt files to the 'books' directory.")

# ============================================================================
# SECTION 6: Model and Tokenizer Setup
# ============================================================================

# Load Phi-2 model and tokenizer
model_name = "microsoft/phi-2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

# ============================================================================
# SECTION 7: LoRA Configuration and Model Preparation
# ============================================================================

# Prepare model for LoRA training
print("Preparing model for LoRA training...")

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Rank
    lora_alpha=32,  # Alpha parameter
    lora_dropout=0.1,  # Dropout
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

print("LoRA configuration applied successfully!")

# ============================================================================
# SECTION 8: Data Tokenization
# ============================================================================

def tokenize_function(examples):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="pt"
    )

# Tokenize dataset
if dataset:
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"Tokenized dataset created with {len(tokenized_dataset)} samples")
else:
    print("No dataset available for tokenization!")

# ============================================================================
# SECTION 9: Training Configuration and Execution
# ============================================================================

# Training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to=None,  # Disable wandb
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize trainer
if 'tokenized_dataset' in locals():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model("./models/phi2-writer")
    tokenizer.save_pretrained("./models/phi2-writer")
    
    print("Training completed and model saved!")
else:
    print("No tokenized dataset available for training!")

# ============================================================================
# SECTION 10: Model Loading and Generation Functions
# ============================================================================

def load_trained_model(model_path: str = "./models/phi2-writer"):
    """Load the trained model and tokenizer."""
    try:
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def generate_text(model, tokenizer, prompt: str, max_length: int = 500, temperature: float = 0.8, top_p: float = 0.9):
    """Generate text using the trained model."""
    if model is None or tokenizer is None:
        print("Model not loaded!")
        return ""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Load the trained model
model, tokenizer = load_trained_model()

# Test generation
if model and tokenizer:
    test_prompt = "The old mansion stood silently against the darkening sky"
    print("\nTest generation:")
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generate_text(model, tokenizer, test_prompt, max_length=200)}")

# ============================================================================
# SECTION 11: Gradio Web Interface
# ============================================================================

def create_writing_interface(model, tokenizer):
    """Create a Gradio interface for the writing assistant."""
    
    def continue_story(prompt, max_length, temperature, top_p):
        """Continue a story from a given prompt."""
        try:
            result = generate_text(model, tokenizer, prompt, max_length, temperature, top_p)
            return result
        except Exception as e:
            return f"Error generating text: {e}"
    
    def write_chapter(title, genre, max_length, temperature, top_p):
        """Write a new chapter based on title and genre."""
        prompt = f"Chapter: {title}\n\nGenre: {genre}\n\n"
        try:
            result = generate_text(model, tokenizer, prompt, max_length, temperature, top_p)
            return result
        except Exception as e:
            return f"Error generating chapter: {e}"
    
    def suggest_characters(description, num_characters, temperature, top_p):
        """Suggest character descriptions."""
        prompt = f"Create {num_characters} character descriptions for: {description}\n\n"
        try:
            result = generate_text(model, tokenizer, prompt, max_length=300, temperature=temperature, top_p=top_p)
            return result
        except Exception as e:
            return f"Error generating characters: {e}"
    
    # Create the interface
    with gr.Blocks(title="Phi-2 Writing Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📚 Phi-2 Writing Assistant
            
            Your AI writing companion, fine-tuned on your book collection!
            """
        )
        
        with gr.Tabs():
            # Tab 1: Continue Story
            with gr.TabItem("Continue Story"):
                gr.Markdown("### Continue your story from where you left off")
                
                with gr.Row():
                    with gr.Column():
                        story_prompt = gr.Textbox(
                            label="Story Prompt",
                            placeholder="Enter the beginning of your story...",
                            lines=5
                        )
                        
                        with gr.Row():
                            story_max_length = gr.Slider(
                                minimum=100, maximum=1000, value=500, step=50,
                                label="Max Length"
                            )
                            story_temperature = gr.Slider(
                                minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                                label="Temperature"
                            )
                            story_top_p = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                                label="Top-p"
                            )
                        
                        continue_btn = gr.Button("Continue Story", variant="primary")
                    
                    with gr.Column():
                        story_output = gr.Textbox(
                            label="Generated Story",
                            lines=15,
                            interactive=False
                        )
                
                continue_btn.click(
                    continue_story,
                    inputs=[story_prompt, story_max_length, story_temperature, story_top_p],
                    outputs=story_output
                )
            
            # Tab 2: Write Chapter
            with gr.TabItem("Write Chapter"):
                gr.Markdown("### Generate a new chapter")
                
                with gr.Row():
                    with gr.Column():
                        chapter_title = gr.Textbox(
                            label="Chapter Title",
                            placeholder="e.g., The Mysterious Arrival"
                        )
                        chapter_genre = gr.Textbox(
                            label="Genre",
                            placeholder="e.g., Mystery, Fantasy, Romance"
                        )
                        
                        with gr.Row():
                            chapter_max_length = gr.Slider(
                                minimum=200, maximum=1500, value=800, step=100,
                                label="Max Length"
                            )
                            chapter_temperature = gr.Slider(
                                minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                                label="Temperature"
                            )
                            chapter_top_p = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                                label="Top-p"
                            )
                        
                        chapter_btn = gr.Button("Write Chapter", variant="primary")
                    
                    with gr.Column():
                        chapter_output = gr.Textbox(
                            label="Generated Chapter",
                            lines=20,
                            interactive=False
                        )
                
                chapter_btn.click(
                    write_chapter,
                    inputs=[chapter_title, chapter_genre, chapter_max_length, chapter_temperature, chapter_top_p],
                    outputs=chapter_output
                )
            
            # Tab 3: Character Suggestions
            with gr.TabItem("Character Suggestions"):
                gr.Markdown("### Generate character descriptions")
                
                with gr.Row():
                    with gr.Column():
                        char_description = gr.Textbox(
                            label="Story/Setting Description",
                            placeholder="Describe your story world or setting...",
                            lines=3
                        )
                        char_count = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="Number of Characters"
                        )
                        
                        with gr.Row():
                            char_temperature = gr.Slider(
                                minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                                label="Temperature"
                            )
                            char_top_p = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                                label="Top-p"
                            )
                        
                        char_btn = gr.Button("Generate Characters", variant="primary")
                    
                    with gr.Column():
                        char_output = gr.Textbox(
                            label="Character Descriptions",
                            lines=15,
                            interactive=False
                        )
                
                char_btn.click(
                    suggest_characters,
                    inputs=[char_description, char_count, char_temperature, char_top_p],
                    outputs=char_output
                )
        
        gr.Markdown(
            """
            ---
            ### Tips for Best Results:
            - **Temperature**: Lower values (0.1-0.5) for more focused text, higher values (0.8-1.2) for more creative text
            - **Top-p**: Controls diversity; 0.9 is usually a good balance
            - **Max Length**: Adjust based on how much text you want generated
            - **Prompts**: Be specific and descriptive for better results
            """
        )
    
    return demo

# Create and launch the interface
if model and tokenizer:
    print("Creating Gradio interface...")
    demo = create_writing_interface(model, tokenizer)
    demo.launch(share=True, debug=True)
else:
    print("Model not loaded! Please run the model loading cell first.")

# ============================================================================
# SECTION 12: Quick Test and Demo
# ============================================================================

# Quick test of the trained model
if model and tokenizer:
    print("=== Quick Test ===")
    
    test_prompts = [
        "The detective entered the dimly lit room",
        "In a world where magic was real",
        "She looked out the window and saw",
        "The ancient castle stood on the hill"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7)}")
        print("-" * 50)
else:
    print("Model not available for testing!")