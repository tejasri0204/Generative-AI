pip install transformers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"  # You can replace this with the model you prefer

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import pipeline

summarizer = pipeline("summarization")

# Define the input text
input_text = """
In recent years, natural language processing (NLP) has seen significant advancements 
due to transformer-based models. Transformers like BERT and GPT have revolutionized 
text analysis and generation. In this hands-on task, we will use a transformer-based 
model to perform text summarization.
"""

# Print the input text
print("Input Text:")
print(input_text)

# Generate the summary
summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)

# Extract and print the generated summary
generated_summary = summary[0]['summary_text']
print("\nGenerated Summary:")
print(generated_summary)