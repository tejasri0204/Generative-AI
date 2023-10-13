pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(input_ids, max_length=50, num_return_sequences=1, 
                        pad_token_id=50256)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)