!pip install nltk

import nltk
from nltk.chat.util import Chat, reflections

# Define pairs of patterns and responses
pairs = [
    ['my name is (.*)', ['Hello %1, how can I help you?']],
    ['(hi|hello|hey|holla)', ['Hey there, how can I assist you?']],
    ['(.*) (location|city) ?', ['Chennai, India']],
    ['(.*) created you ?', ['I was created by Tejasri.']],
    ['how is the weather in (.*)', ['The weather in %1 is usually nice.']],
    ['(.*) help (.*)', ['I can help you with various things.']],
    ['(.*) your name ?', ['I am a chatbot. You can call me ChatBot.']],
    ['Thank you', ['You are welcome!']],
    ['exit', ['Goodbye!']],
]

# Create a chatbot using the pairs defined above
chatbot = Chat(pairs, reflections)

# Chat interface
print("Welcome! Type 'exit' to end the conversation.")
chatbot.converse()