import re

def clean_recipe_text(recipe_text):
    # Remove unnecessary characters and symbols using regular expressions
    cleaned_text = re.sub(r'\[.*?\]', '', recipe_text)  # Remove square brackets and their contents
    cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)  # Remove parentheses and their contents
    cleaned_text = re.sub(r'\{.*?\}', '', cleaned_text)  # Remove curly braces and their contents
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)  # Remove angle brackets and their contents
    cleaned_text = re.sub(r'&.*?;', '', cleaned_text)  # Remove HTML entities
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    
    # Remove leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# Example usage
recipe_text = "[1] This is a sample recipe (with unnecessary symbols) {and formatting} <and HTML entities>."
cleaned_text = clean_recipe_text(recipe_text)
print(cleaned_text)
