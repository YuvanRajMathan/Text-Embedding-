import os
import openai
import numpy as np
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from openai.embeddings_utils import distances_from_embeddings

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or include it in the script.")
else:
    openai.api_key = api_key

def clean_text(text):
    """
    Clean up text by removing extra spaces and blank lines.
    """
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = ' '.join(lines)
    return cleaned_text

def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe.
    """
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values.tolist(), distance_metric='cosine')

    returns = []
    cur_len = 0

    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)

def answer_question(df, model="gpt-3.5-turbo", question="", max_len=1800, size="ada", debug=False, max_tokens=150, stop_sequence=None):
    """
    Answer a question based on the most similar context from the dataframe texts.
    """
    context = create_context(question, df, max_len=max_len, size=size)
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
            {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_sequence,
    )
    return response.choices[0].message['content'].strip()

