#!/usr/bin/env python3
"""
Example Usage of Phi-2 Fine-tuned Writing Assistant

This script demonstrates how to use the fine-tuned Phi-2 model
for creative writing tasks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./models/phi2-writer"):
    """Load the fine-tuned model and tokenizer."""
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
        print("Make sure you have trained the model first!")
        return None, None

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.8, top_p=0.9):
    """Generate text using the fine-tuned model."""
    if model is None or tokenizer is None:
        return "Model not loaded!"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

def demonstrate_writing_assistant():
    """Demonstrate various writing tasks."""
    print("=== Phi-2 Writing Assistant Demo ===\n")
    
    # Load the model
    model, tokenizer = load_model()
    
    if model is None:
        print("Please train the model first using phi2_finetuning.py")
        return
    
    # Example 1: Continue a story
    print("1. CONTINUING A STORY")
    print("-" * 40)
    story_prompt = "The old mansion stood silently against the darkening sky, its windows like hollow eyes watching the approaching storm."
    print(f"Prompt: {story_prompt}")
    print("\nGenerated continuation:")
    continuation = generate_text(model, tokenizer, story_prompt, max_length=300, temperature=0.8)
    print(continuation)
    print("\n" + "="*60 + "\n")
    
    # Example 2: Write a chapter
    print("2. WRITING A NEW CHAPTER")
    print("-" * 40)
    chapter_prompt = "Chapter: The Mysterious Arrival\n\nGenre: Mystery\n\n"
    print(f"Prompt: {chapter_prompt}")
    print("\nGenerated chapter:")
    chapter = generate_text(model, tokenizer, chapter_prompt, max_length=600, temperature=0.9)
    print(chapter)
    print("\n" + "="*60 + "\n")
    
    # Example 3: Character suggestions
    print("3. CHARACTER SUGGESTIONS")
    print("-" * 40)
    char_prompt = "Create 3 character descriptions for: A steampunk fantasy world with airships and magic"
    print(f"Prompt: {char_prompt}")
    print("\nGenerated characters:")
    characters = generate_text(model, tokenizer, char_prompt, max_length=400, temperature=0.7)
    print(characters)
    print("\n" + "="*60 + "\n")
    
    # Example 4: Different genres
    print("4. DIFFERENT GENRES")
    print("-" * 40)
    
    genres = [
        ("Romance", "She felt her heart skip a beat when he walked into the room"),
        ("Sci-Fi", "The quantum drive hummed softly as the ship prepared for hyperspace"),
        ("Horror", "The shadows in the corner seemed to move on their own"),
        ("Fantasy", "The dragon's scales shimmered like precious metals in the sunlight")
    ]
    
    for genre, prompt in genres:
        print(f"\n{genre.upper()}:")
        print(f"Prompt: {prompt}")
        result = generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8)
        print(f"Generated: {result}")
        print("-" * 30)
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Interactive mode
    print("5. INTERACTIVE MODE")
    print("-" * 40)
    print("Enter your own prompts (type 'quit' to exit):")
    
    while True:
        user_prompt = input("\nEnter a prompt: ")
        if user_prompt.lower() == 'quit':
            break
        
        if user_prompt.strip():
            print("\nGenerating...")
            result = generate_text(model, tokenizer, user_prompt, max_length=300, temperature=0.8)
            print(f"Generated: {result}")
        else:
            print("Please enter a valid prompt.")

def main():
    """Main function."""
    print("Phi-2 Writing Assistant - Example Usage")
    print("=" * 50)
    
    # Check if model exists
    import os
    if not os.path.exists("./models/phi2-writer"):
        print("No trained model found!")
        print("Please run the training first:")
        print("python phi2_finetuning.py --mode train")
        return
    
    demonstrate_writing_assistant()

if __name__ == "__main__":
    main()