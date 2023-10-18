# Define a list of predefined inappropriate keywords
inappropriate_keywords = ["offensive_word1", "offensive_word2", "profanity", "harmful_phrase"]

# Create a function for content moderation
def content_moderation(user_input):
    # Convert the user input to lowercase for case-insensitive matching
    user_input = user_input.lower()
    
    # Check if any predefined keyword exists in the user input
    for keyword in inappropriate_keywords:
        if keyword in user_input:
            return "This content contains inappropriate language and is not allowed."
    
    # If no inappropriate keyword is found, the content is considered appropriate
    return "This content is appropriate."

# Get user input
user_input = input("Enter your text: ")

# Check the content and display the moderation result
moderation_result = content_moderation(user_input)
print(moderation_result)