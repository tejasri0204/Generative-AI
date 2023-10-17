import openai

api_key = "YOUR_API_KEY"
openai.api_key = api_key

prompt = "Translate the following English text to French: 'Hello, how are you?'"

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50,
    temperature=0.7
)

generated_text = response.choices[0].text
print("Generated Text:")
print(generated_text)
