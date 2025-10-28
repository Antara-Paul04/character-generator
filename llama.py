from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use device mapping to handle memory efficiently
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    # Load tokenizer and model with more aggressive memory settings
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        dtype=torch.float16,  # Use half precision
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="./offload"
    )
    
    # Create messages
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    print("Generating response...")
    
    # Generate response with more tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,  # Increased to get complete responses
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("Response:", response)
    
except Exception as e:
    print(f"Error: {e}")