pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"  # You can change this to "gpt2-medium", "gpt2-large", etc. for larger models.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Hey there..."
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)