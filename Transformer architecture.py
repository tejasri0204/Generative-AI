pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can also choose other GPT-2 variants, such as gpt2-medium, gpt2-large, etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text based on the input prompt
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, top_k=50)

    generated_text = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_text

prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=100, num_return_sequences=1)

for text in generated_text:
    print(text)
