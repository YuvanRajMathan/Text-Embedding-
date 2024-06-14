import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from utils import clean_text, answer_question
import openai

# Directory containing the text files
text_directory = 'data/raw'

# Verify the directory
if not os.path.isdir(text_directory):
    raise ValueError(f"Directory {text_directory} does not exist. Please check the path.")

# List to hold the cleaned text from each file
cleaned_texts = []

# Loop through all text files in the directory
for filename in os.listdir(text_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(text_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokenized_text = ' '.join(word_tokenize(text))
            cleaned_text = clean_text(tokenized_text)
            cleaned_texts.append(cleaned_text)

# Create a Pandas DataFrame from the list of cleaned texts
df = pd.DataFrame(cleaned_texts, columns=['text'])
df['n_tokens'] = df['text'].apply(lambda x: len(word_tokenize(x)))

# Generate embeddings for each text and store in the DataFrame
df['embeddings'] = df['text'].apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

# Write the DataFrame to a CSV file
csv_file_path = 'data/processed/embeddings.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
df.to_csv(csv_file_path, index=False)

# Read the CSV file and process embeddings
df = pd.read_csv('data/processed/embeddings.csv')
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# Example question
question = input("Enter your question: ")

# Answer the question
answer = answer_question(df, question=question)
print("Answer:", answer)

