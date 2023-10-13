pip install transformers torch

!pip install transformers[torch]

# Install necessary libraries
!pip install transformers

# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Define your custom dataset (replace this with your own data)
dataset = [
    "Most high, most mighty, and most puissant Caesar",
    "Metellus Cimber throws before thy seat",
    "An humble heart",
]

# Initialize the GPT-2 tokenizer and model
model_name = "gpt2"  # You can choose another GPT-2 variant if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset
input_data = [tokenizer(text, return_tensors="pt") for text in dataset]

# Fine-tune the GPT-2 model on your dataset
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Fine-tune for a fixed number of epochs
num_epochs = 1
for epoch in range(num_epochs):
    for data in input_data:
        optimizer.zero_grad()
        outputs = model(**data, labels=data["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned")

# Load the fine-tuned model for text generation
generator = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
generator.eval()

# Generate text based on a prompt
prompt = "Most puissant Caesar"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = generator.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text with 7 words per line and multiple lines
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Define the number of words per line
words_per_line = 7

# Split the text into paragraphs with the specified number of words per line
formatted_text = ""
words = generated_text.split()
current_line = []

for word in words:
    if len(current_line) + len(word.split()) <= words_per_line:
        current_line.extend(word.split())
    else:
        formatted_text += " ".join(current_line) + "\n"
        current_line = word.split()

if current_line:
    formatted_text += " ".join(current_line)

# Print the formatted text
print(formatted_text)
