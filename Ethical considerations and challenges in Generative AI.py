pip install openai

import openai
import random

api_key = "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"

conversation_prompts = [
    "User: Hi there! Tell me a funny joke.",
    "User: What's your favorite book?",
    "User: Can you help me with my math homework?",
]

openai.api_key = "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"

# Function to generate chatbot response
def generate_chat_response(prompt, engine="text-davinci-003"):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def handle_ethical_concerns(response):
    sensitive_words = ["bias", "discrimination", "controversial_topic", "inappropriate_word"]  # Add sensitive words/topics
    for word in sensitive_words:
        if word in response:
            response = "I'm sorry, but I can't provide a response to that question due to ethical considerations."
            break
    return response

def test_chatbot():
    for prompt in conversation_prompts:
        print(prompt)
        response = generate_chat_response(prompt)
        modified_response = handle_ethical_concerns(response)
        print("Chatbot:", modified_response)
        print()

# Run the test_chatbot function
if __name__ == "__main__":
    test_chatbot()