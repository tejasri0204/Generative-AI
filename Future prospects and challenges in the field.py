pip install openai

import openai

openai.api_key = "sk-Tt8KNeujNqz0uEm3IqzVT3BlbkFtrjH9v34gVZvT7Gdn23w"

prompt = "Write a short story about a time-traveling explorer who discovers a hidden civilization in a remote jungle."

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=150,  # Adjust the length of the generated text
)

generated_text = response.choices[0].text

print(generated_text)