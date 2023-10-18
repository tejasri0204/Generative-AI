pip install transformers

pip install torch

!pip install transformers[torch]

!pip install accelerate

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Sample dataset for fine-tuning
sample_dataset = """
This is a sample sentence.
You can add more sentences.
Customize this dataset for your task.
"""

# Save the sample dataset to a file
with open("sample_dataset.txt", "w") as file:
    file.write(sample_dataset)

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize and prepare the dataset
def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dataset, data_collator

train_dataset, data_collator = load_dataset("sample_dataset.txt", tokenizer)

# Set training parameters
training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create a Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine-tuned-gpt2")

# Generate text with the fine-tuned model
from transformers import pipeline

generator = pipeline('text-generation', model='./fine-tuned-gpt2', tokenizer=model_name)

generated_text = generator("Generate text that follows a specific pattern: ", max_length=50, num_return_sequences=1)
print(generated_text[0]['generated_text'])